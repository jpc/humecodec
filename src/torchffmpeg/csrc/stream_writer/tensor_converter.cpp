#include "torchffmpeg/csrc/stream_writer/tensor_converter.h"
#include "torchffmpeg/csrc/tensor_view.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace torchffmpeg {

namespace {

using InitFunc = TensorConverter::InitFunc;
using ConvertFunc = TensorConverter::ConvertFunc;

////////////////////////////////////////////////////////////////////////////////
// Helper: DLDataType comparison
////////////////////////////////////////////////////////////////////////////////

inline bool dtype_eq(DLDataType a, DLDataType b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

////////////////////////////////////////////////////////////////////////////////
// Helper: make a contiguous copy of a ManagedBuffer if it isn't already
////////////////////////////////////////////////////////////////////////////////

ManagedBuffer ensure_contiguous(const ManagedBuffer& buf) {
  // Check if already contiguous (row-major)
  int64_t expected = 1;
  bool contiguous = true;
  for (int i = buf.ndim() - 1; i >= 0; --i) {
    if (buf.shape()[i] != 1 && buf.strides()[i] != expected) {
      contiguous = false;
      break;
    }
    expected *= buf.shape()[i];
  }
  if (contiguous) {
    // We need to return a new buffer that owns a copy since ManagedBuffer
    // is move-only. For the contiguous case we do a full copy.
    ManagedBuffer out(buf.shape(), buf.dtype(), buf.device());
    if (buf.is_cpu()) {
      std::memcpy(out.data(), buf.data(), buf.nbytes());
    } else {
#ifdef USE_CUDA
      cudaMemcpy(out.data(), buf.data(), buf.nbytes(), cudaMemcpyDeviceToDevice);
#endif
    }
    return out;
  }

  // Non-contiguous: copy element-by-element with stride awareness
  ManagedBuffer out(buf.shape(), buf.dtype(), buf.device());
  size_t elem_sz = buf.element_size();

  if (buf.is_cpu()) {
    // Generic strided copy for CPU
    int64_t total = buf.numel();
    const auto& sh = buf.shape();
    const auto& st = buf.strides();
    int nd = buf.ndim();
    std::vector<int64_t> idx(nd, 0);

    const uint8_t* src_base = static_cast<const uint8_t*>(buf.data());
    uint8_t* dst = static_cast<uint8_t*>(out.data());

    for (int64_t flat = 0; flat < total; ++flat) {
      // Compute source offset from strides
      int64_t src_offset = 0;
      for (int d = 0; d < nd; ++d) {
        src_offset += idx[d] * st[d];
      }
      std::memcpy(dst + flat * elem_sz, src_base + src_offset * elem_sz, elem_sz);

      // Increment multi-index
      for (int d = nd - 1; d >= 0; --d) {
        if (++idx[d] < sh[d]) break;
        idx[d] = 0;
      }
    }
  } else {
#ifdef USE_CUDA
    // For CUDA non-contiguous, this is a rare path. Fall back to device copy
    // assuming the buffer was already made contiguous by the caller.
    cudaMemcpy(out.data(), buf.data(), buf.nbytes(), cudaMemcpyDeviceToDevice);
#endif
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Helper: NCHW to NHWC permutation (returns a new contiguous ManagedBuffer)
////////////////////////////////////////////////////////////////////////////////

ManagedBuffer nchw_to_nhwc(const ManagedBuffer& buf) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(buf.ndim() == 4);
  int64_t N = buf.size(0);
  int64_t C = buf.size(1);
  int64_t H = buf.size(2);
  int64_t W = buf.size(3);
  size_t elem_sz = buf.element_size();

  ManagedBuffer out({N, H, W, C}, buf.dtype(), buf.device());

  if (buf.is_cpu()) {
    const uint8_t* src = static_cast<const uint8_t*>(buf.data());
    uint8_t* dst = static_cast<uint8_t*>(out.data());

    // Source strides in elements (assuming contiguous NCHW)
    int64_t src_n_stride = C * H * W;
    int64_t src_c_stride = H * W;
    int64_t src_h_stride = W;

    for (int64_t n = 0; n < N; ++n) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          for (int64_t c = 0; c < C; ++c) {
            int64_t src_idx = n * src_n_stride + c * src_c_stride + h * src_h_stride + w;
            int64_t dst_idx = n * (H * W * C) + h * (W * C) + w * C + c;
            std::memcpy(dst + dst_idx * elem_sz, src + src_idx * elem_sz, elem_sz);
          }
        }
      }
    }
  } else {
#ifdef USE_CUDA
    // For CUDA: allocate a temp host buffer, do the permute on host, copy back.
    // This is the encoder path and typically not performance-critical.
    size_t total_bytes = buf.nbytes();
    std::vector<uint8_t> host_src(total_bytes);
    std::vector<uint8_t> host_dst(total_bytes);
    cudaMemcpy(host_src.data(), buf.data(), total_bytes, cudaMemcpyDeviceToHost);

    int64_t src_n_stride = C * H * W;
    int64_t src_c_stride = H * W;
    int64_t src_h_stride = W;

    for (int64_t n = 0; n < N; ++n) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          for (int64_t c = 0; c < C; ++c) {
            int64_t src_idx = n * src_n_stride + c * src_c_stride + h * src_h_stride + w;
            int64_t dst_idx = n * (H * W * C) + h * (W * C) + w * C + c;
            std::memcpy(host_dst.data() + dst_idx * elem_sz, host_src.data() + src_idx * elem_sz, elem_sz);
          }
        }
      }
    }
    cudaMemcpy(out.data(), host_dst.data(), total_bytes, cudaMemcpyHostToDevice);
#endif
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Helper: Slice a sub-range along dim 0 of a contiguous ManagedBuffer
// Returns a new ManagedBuffer owning a copy of buf[start:end] along dim 0.
////////////////////////////////////////////////////////////////////////////////

ManagedBuffer slice_dim0(const ManagedBuffer& buf, int64_t start, int64_t end) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(buf.ndim() >= 1);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(start >= 0 && end <= buf.size(0) && start < end);

  std::vector<int64_t> out_shape = buf.shape();
  out_shape[0] = end - start;

  // Compute bytes per "row" (one element along dim 0)
  size_t row_bytes = buf.element_size();
  for (int i = 1; i < buf.ndim(); ++i) {
    row_bytes *= buf.size(i);
  }

  ManagedBuffer out(out_shape, buf.dtype(), buf.device());
  size_t offset = start * row_bytes;
  size_t copy_bytes = (end - start) * row_bytes;

  if (buf.is_cpu()) {
    std::memcpy(out.data(), static_cast<const uint8_t*>(buf.data()) + offset, copy_bytes);
  } else {
#ifdef USE_CUDA
    cudaMemcpy(out.data(), static_cast<const uint8_t*>(buf.data()) + offset, copy_bytes, cudaMemcpyDeviceToDevice);
#endif
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////

inline int get_frame_channels(const AVFrame* frame) {
#if LIBAVUTIL_VERSION_MAJOR >= 59
  return frame->ch_layout.nb_channels;
#else
  return frame->channels;
#endif
}

void validate_audio_input(
    const ManagedBuffer& t,
    AVFrame* buffer,
    DLDataType expected_dtype) {
  int num_channels = get_frame_channels(buffer);
  TFMPEG_CHECK(
      dtype_eq(t.dtype(), expected_dtype),
      "Expected matching dtype for audio encoding.");
  TFMPEG_CHECK(t.is_cpu(), "Input buffer has to be on CPU.");
  TFMPEG_CHECK(t.ndim() == 2, "Input buffer has to be 2D.");
  TFMPEG_CHECK(
      t.size(1) == num_channels,
      "Expected waveform with ",
      num_channels,
      " channels. Found ",
      t.size(1));
}

// 2D (time, channel) and contiguous.
void convert_func_(const ManagedBuffer& chunk, AVFrame* buffer) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(chunk.ndim() == 2);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(chunk.size(1) == get_frame_channels(buffer));

  if (!av_frame_is_writable(buffer)) {
    int ret = av_frame_make_writable(buffer);
    TFMPEG_INTERNAL_ASSERT(
        ret >= 0, "Failed to make frame writable: ", av_err2string(ret));
  }

  auto byte_size = chunk.numel() * chunk.element_size();
  memcpy(buffer->data[0], chunk.data(), byte_size);
  buffer->nb_samples = static_cast<int>(chunk.size(0));
}

std::pair<InitFunc, ConvertFunc> get_audio_func(AVFrame* buffer) {
  auto expected_dtype = [&]() -> DLDataType {
    switch (static_cast<AVSampleFormat>(buffer->format)) {
      case AV_SAMPLE_FMT_U8:
        return dtype::uint8();
      case AV_SAMPLE_FMT_S16:
        return dtype::int16();
      case AV_SAMPLE_FMT_S32:
        return dtype::int32();
      case AV_SAMPLE_FMT_S64:
        return dtype::int64();
      case AV_SAMPLE_FMT_FLT:
        return dtype::float32();
      case AV_SAMPLE_FMT_DBL:
        return dtype::float64();
      default:
        TFMPEG_INTERNAL_ASSERT(
            false, "Audio encoding process is not properly configured.");
    }
  }();

  InitFunc init_func = [=](const ManagedBuffer& buf, AVFrame* buffer) {
    validate_audio_input(buf, buffer, expected_dtype);
    return ensure_contiguous(buf);
  };
  return {init_func, convert_func_};
}

////////////////////////////////////////////////////////////////////////////////
// Video
////////////////////////////////////////////////////////////////////////////////

void validate_video_input(
    const ManagedBuffer& t,
    AVFrame* buffer,
    int num_channels) {
  if (buffer->hw_frames_ctx) {
    TFMPEG_CHECK(t.is_cuda(), "Input buffer has to be on CUDA.");
  } else {
    TFMPEG_CHECK(t.is_cpu(), "Input buffer has to be on CPU.");
  }
  TFMPEG_CHECK(
      dtype_eq(t.dtype(), dtype::uint8()),
      "Expected buffer of uint8 type.");

  TFMPEG_CHECK(t.ndim() == 4, "Input buffer has to be 4D.");
  TFMPEG_CHECK(
      t.size(1) == num_channels && t.size(2) == buffer->height &&
          t.size(3) == buffer->width,
      "Expected buffer with shape (N, ",
      num_channels,
      ", ",
      buffer->height,
      ", ",
      buffer->width,
      ") (NCHW format).");
}

// Special case where encode pixel format is RGB0/BGR0 but the tensor is RGB/BGR
void validate_rgb0(const ManagedBuffer& t, AVFrame* buffer) {
  if (buffer->hw_frames_ctx) {
    TFMPEG_CHECK(t.is_cuda(), "Input buffer has to be on CUDA.");
  } else {
    TFMPEG_CHECK(t.is_cpu(), "Input buffer has to be on CPU.");
  }
  TFMPEG_CHECK(
      dtype_eq(t.dtype(), dtype::uint8()),
      "Expected buffer of uint8 type.");

  TFMPEG_CHECK(t.ndim() == 4, "Input buffer has to be 4D.");
  TFMPEG_CHECK(
      t.size(2) == buffer->height && t.size(3) == buffer->width,
      "Expected buffer with shape (N, 3, ",
      buffer->height,
      ", ",
      buffer->width,
      ") (NCHW format).");
}

// NCHW -> NHWC, ensure contiguous
ManagedBuffer init_interlaced(const ManagedBuffer& buf) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(buf.ndim() == 4);
  auto contiguous = ensure_contiguous(buf);
  return nchw_to_nhwc(contiguous);
}

// Keep NCHW, ensure contiguous
ManagedBuffer init_planar(const ManagedBuffer& buf) {
  return ensure_contiguous(buf);
}

// Convert RGB (3-channel NCHW) to RGB0/BGR0 (4-channel NHWC) by padding
ManagedBuffer rgb_to_rgb0_nhwc(const ManagedBuffer& buf) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(buf.ndim() == 4);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(buf.size(1) == 3);

  int64_t N = buf.size(0);
  int64_t H = buf.size(2);
  int64_t W = buf.size(3);

  // Ensure input is contiguous first
  auto src = ensure_contiguous(buf);

  // Allocate NHWC with 4 channels
  ManagedBuffer out({N, H, W, 4}, dtype::uint8(), buf.device());

  if (buf.is_cpu()) {
    const uint8_t* sp = src.data_ptr<uint8_t>();
    uint8_t* dp = out.data_ptr<uint8_t>();

    // Zero the output (sets alpha/padding channel to 0)
    std::memset(dp, 0, out.nbytes());

    // Source is NCHW: [N, 3, H, W]
    int64_t chw = 3 * H * W;
    int64_t hw = H * W;

    for (int64_t n = 0; n < N; ++n) {
      const uint8_t* src_n = sp + n * chw;
      uint8_t* dst_n = dp + n * H * W * 4;
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          int64_t pixel_idx = h * W + w;
          uint8_t* dst_pixel = dst_n + pixel_idx * 4;
          dst_pixel[0] = src_n[0 * hw + pixel_idx]; // R or B
          dst_pixel[1] = src_n[1 * hw + pixel_idx]; // G
          dst_pixel[2] = src_n[2 * hw + pixel_idx]; // B or R
          // dst_pixel[3] remains 0 (padding)
        }
      }
    }
  } else {
#ifdef USE_CUDA
    // Host-staged approach for CUDA
    size_t src_bytes = src.nbytes();
    size_t dst_bytes = out.nbytes();
    std::vector<uint8_t> host_src(src_bytes);
    std::vector<uint8_t> host_dst(dst_bytes, 0);
    cudaMemcpy(host_src.data(), src.data(), src_bytes, cudaMemcpyDeviceToHost);

    int64_t chw = 3 * H * W;
    int64_t hw = H * W;

    for (int64_t n = 0; n < N; ++n) {
      const uint8_t* src_n = host_src.data() + n * chw;
      uint8_t* dst_n = host_dst.data() + n * H * W * 4;
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          int64_t pixel_idx = h * W + w;
          uint8_t* dst_pixel = dst_n + pixel_idx * 4;
          dst_pixel[0] = src_n[0 * hw + pixel_idx];
          dst_pixel[1] = src_n[1 * hw + pixel_idx];
          dst_pixel[2] = src_n[2 * hw + pixel_idx];
        }
      }
    }
    cudaMemcpy(out.data(), host_dst.data(), dst_bytes, cudaMemcpyHostToDevice);
#endif
  }
  return out;
}

// Interlaced video write (NHWC buffer → AVFrame)
void write_interlaced_video(
    const ManagedBuffer& frame,
    AVFrame* buffer,
    int num_channels) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.ndim() == 4);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(0) == 1);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(1) == buffer->height);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(2) == buffer->width);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(3) == num_channels);

  if (!av_frame_is_writable(buffer)) {
    int ret = av_frame_make_writable(buffer);
    TFMPEG_INTERNAL_ASSERT(
        ret >= 0, "Failed to make frame writable: ", av_err2string(ret));
  }

  size_t stride = buffer->width * num_channels;
  const uint8_t* src = static_cast<const uint8_t*>(frame.data());
  uint8_t* dst = buffer->data[0];
  for (int h = 0; h < buffer->height; ++h) {
    std::memcpy(dst, src, stride);
    src += stride;
    dst += buffer->linesize[0];
  }
}

// Planar video write (NCHW buffer → AVFrame)
void write_planar_video(
    const ManagedBuffer& frame,
    AVFrame* buffer,
    int num_planes) {
  const auto num_colors =
      av_pix_fmt_desc_get((AVPixelFormat)buffer->format)->nb_components;
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.ndim() == 4);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(0) == 1);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(1) == num_colors);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(2) == buffer->height);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(3) == buffer->width);

  if (!av_frame_is_writable(buffer)) {
    int ret = av_frame_make_writable(buffer);
    TFMPEG_INTERNAL_ASSERT(
        ret >= 0, "Failed to make frame writable: ", av_err2string(ret));
  }

  const uint8_t* src_base = static_cast<const uint8_t*>(frame.data());
  int64_t plane_size = buffer->height * buffer->width;

  for (int j = 0; j < num_colors; ++j) {
    const uint8_t* src = src_base + j * plane_size;
    uint8_t* dst = buffer->data[j];
    for (int h = 0; h < buffer->height; ++h) {
      memcpy(dst, src, buffer->width);
      src += buffer->width;
      dst += buffer->linesize[j];
    }
  }
}

void write_interlaced_video_cuda(
    const ManagedBuffer& frame,
    AVFrame* buffer,
    int num_channels) {
#ifndef USE_CUDA
  TFMPEG_CHECK(
      false,
      "torchffmpeg is not compiled with CUDA support. Hardware acceleration is not available.");
#else
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.ndim() == 4);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(0) == 1);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(1) == buffer->height);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(2) == buffer->width);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(3) == num_channels);
  size_t spitch = buffer->width * num_channels;
  if (cudaSuccess !=
      cudaMemcpy2D(
          (void*)(buffer->data[0]),
          buffer->linesize[0],
          frame.data(),
          spitch,
          spitch,
          buffer->height,
          cudaMemcpyDeviceToDevice)) {
    TFMPEG_CHECK(false, "Failed to copy pixel data from CUDA buffer.");
  }
#endif
}

void write_planar_video_cuda(
    const ManagedBuffer& frame,
    AVFrame* buffer,
    int num_planes) {
#ifndef USE_CUDA
  TFMPEG_CHECK(
      false,
      "torchffmpeg is not compiled with CUDA support. Hardware acceleration is not available.");
#else
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.ndim() == 4);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(0) == 1);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(1) == num_planes);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(2) == buffer->height);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(frame.size(3) == buffer->width);

  const uint8_t* src_base = static_cast<const uint8_t*>(frame.data());
  int64_t plane_size = buffer->height * buffer->width;

  for (int j = 0; j < num_planes; ++j) {
    if (cudaSuccess !=
        cudaMemcpy2D(
            (void*)(buffer->data[j]),
            buffer->linesize[j],
            (const void*)(src_base + j * plane_size),
            buffer->width,
            buffer->width,
            buffer->height,
            cudaMemcpyDeviceToDevice)) {
      TFMPEG_CHECK(false, "Failed to copy pixel data from CUDA buffer.");
    }
  }
#endif
}

std::pair<InitFunc, ConvertFunc> get_video_func(AVFrame* buffer) {
  if (buffer->hw_frames_ctx) {
    auto frames_ctx = (AVHWFramesContext*)(buffer->hw_frames_ctx->data);
    auto sw_pix_fmt = frames_ctx->sw_format;
    switch (sw_pix_fmt) {
      case AV_PIX_FMT_RGB0:
      case AV_PIX_FMT_BGR0: {
        ConvertFunc convert_func = [](const ManagedBuffer& t, AVFrame* f) {
          write_interlaced_video_cuda(t, f, 4);
        };
        InitFunc init_func = [](const ManagedBuffer& t, AVFrame* f) -> ManagedBuffer {
          if (t.ndim() == 4 && t.size(1) == 3) {
            validate_rgb0(t, f);
            return rgb_to_rgb0_nhwc(t);
          }
          validate_video_input(t, f, 4);
          return init_interlaced(t);
        };
        return {init_func, convert_func};
      }
      case AV_PIX_FMT_GBRP:
      case AV_PIX_FMT_GBRP16LE:
      case AV_PIX_FMT_YUV444P:
      case AV_PIX_FMT_YUV444P16LE: {
        ConvertFunc convert_func = [](const ManagedBuffer& t, AVFrame* f) {
          write_planar_video_cuda(t, f, 3);
        };
        InitFunc init_func = [](const ManagedBuffer& t, AVFrame* f) -> ManagedBuffer {
          validate_video_input(t, f, 3);
          return init_planar(t);
        };
        return {init_func, convert_func};
      }
      default:
        TFMPEG_CHECK(
            false,
            "Unexpected pixel format for CUDA: ",
            av_get_pix_fmt_name(sw_pix_fmt));
    }
  }

  auto pix_fmt = static_cast<AVPixelFormat>(buffer->format);
  switch (pix_fmt) {
    case AV_PIX_FMT_GRAY8:
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24: {
      int channels = av_pix_fmt_desc_get(pix_fmt)->nb_components;
      InitFunc init_func = [=](const ManagedBuffer& t, AVFrame* f) -> ManagedBuffer {
        validate_video_input(t, f, channels);
        return init_interlaced(t);
      };
      ConvertFunc convert_func = [=](const ManagedBuffer& t, AVFrame* f) {
        write_interlaced_video(t, f, channels);
      };
      return {init_func, convert_func};
    }
    case AV_PIX_FMT_RGB0:
    case AV_PIX_FMT_BGR0: {
      InitFunc init_func = [](const ManagedBuffer& t, AVFrame* f) -> ManagedBuffer {
        if (t.ndim() == 4 && t.size(1) == 3) {
          validate_rgb0(t, f);
          return rgb_to_rgb0_nhwc(t);
        }
        validate_video_input(t, f, 4);
        return init_interlaced(t);
      };
      ConvertFunc convert_func = [](const ManagedBuffer& t, AVFrame* f) {
        write_interlaced_video(t, f, 4);
      };
      return {init_func, convert_func};
    }
    case AV_PIX_FMT_YUV444P: {
      InitFunc init_func = [](const ManagedBuffer& t, AVFrame* f) -> ManagedBuffer {
        validate_video_input(t, f, 3);
        return init_planar(t);
      };
      ConvertFunc convert_func = [](const ManagedBuffer& t, AVFrame* f) {
        write_planar_video(t, f, 3);
      };
      return {init_func, convert_func};
    }
    default:
      TFMPEG_CHECK(
          false, "Unexpected pixel format: ", av_get_pix_fmt_name(pix_fmt));
  }
}

////////////////////////////////////////////////////////////////////////////////
// Unknown (for supporting frame writing)
////////////////////////////////////////////////////////////////////////////////
std::pair<InitFunc, ConvertFunc> get_frame_func() {
  InitFunc init_func = [](const ManagedBuffer&,
                          AVFrame*) -> ManagedBuffer {
    TFMPEG_CHECK(
        false,
        "This shouldn't have been called. "
        "If you intended to write frames, please select a stream that supports doing so.");
  };
  ConvertFunc convert_func = [](const ManagedBuffer&, AVFrame*) {
    TFMPEG_CHECK(
        false,
        "This shouldn't have been called. "
        "If you intended to write frames, please select a stream that supports doing so.");
  };
  return {init_func, convert_func};
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// TensorConverter
////////////////////////////////////////////////////////////////////////////////

TensorConverter::TensorConverter(AVMediaType type, AVFrame* buf, int buf_size)
    : buffer(buf), buffer_size(buf_size) {
  switch (type) {
    case AVMEDIA_TYPE_AUDIO:
      std::tie(init_func, convert_func) = get_audio_func(buffer);
      break;
    case AVMEDIA_TYPE_VIDEO:
      std::tie(init_func, convert_func) = get_video_func(buffer);
      break;
    case AVMEDIA_TYPE_UNKNOWN:
      std::tie(init_func, convert_func) = get_frame_func();
      break;
    default:
      TFMPEG_INTERNAL_ASSERT(
          false, "Unsupported media type: ", av_get_media_type_string(type));
  }
}

using Generator = TensorConverter::Generator;

Generator TensorConverter::convert(const ManagedBuffer& t) {
  return Generator{init_func(t, buffer), buffer, convert_func, buffer_size};
}

////////////////////////////////////////////////////////////////////////////////
// Generator
////////////////////////////////////////////////////////////////////////////////

using Iterator = Generator::Iterator;

Generator::Generator(
    ManagedBuffer frames_,
    AVFrame* buff,
    ConvertFunc& func,
    int64_t step_)
    : frames(std::move(frames_)),
      buffer(buff),
      convert_func(func),
      step(step_) {}

Iterator Generator::begin() const {
  return Iterator{frames, buffer, convert_func, step};
}

int64_t Generator::end() const {
  return frames.size(0);
}

////////////////////////////////////////////////////////////////////////////////
// Iterator
////////////////////////////////////////////////////////////////////////////////

Iterator::Iterator(
    const ManagedBuffer& frames_,
    AVFrame* buffer_,
    ConvertFunc& convert_func_,
    int64_t step_)
    : frames(frames_),
      buffer(buffer_),
      convert_func(convert_func_),
      step(step_) {}

Iterator& Iterator::operator++() {
  i += step;
  return *this;
}

AVFrame* Iterator::operator*() const {
  auto chunk = slice_dim0(frames, i, i + step);
  convert_func(chunk, buffer);
  return buffer;
}

bool Iterator::operator!=(const int64_t end) const {
  return i < end;
}

} // namespace torchffmpeg
