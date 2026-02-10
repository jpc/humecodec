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
// Helper: make a contiguous copy of a ManagedBuffer
////////////////////////////////////////////////////////////////////////////////

ManagedBuffer copy_buffer(const ManagedBuffer& buf) {
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

std::pair<InitFunc, ConvertFunc> get_audio_func(AVFrame*) {
  // Python has already validated and made the tensor contiguous.
  InitFunc init_func = [](const ManagedBuffer& buf, AVFrame*) {
    return copy_buffer(buf);
  };
  return {init_func, convert_func_};
}

////////////////////////////////////////////////////////////////////////////////
// Video
////////////////////////////////////////////////////////////////////////////////

// Interlaced video write (NHWC buffer -> AVFrame)
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

// Planar video write (NCHW buffer -> AVFrame)
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

// Python has already transformed the tensor to the correct layout.
// InitFunc just copies the buffer for ownership.
std::pair<InitFunc, ConvertFunc> get_video_func(AVFrame* buffer) {
  InitFunc init_func = [](const ManagedBuffer& t, AVFrame*) -> ManagedBuffer {
    return copy_buffer(t);
  };

  if (buffer->hw_frames_ctx) {
    auto frames_ctx = (AVHWFramesContext*)(buffer->hw_frames_ctx->data);
    auto sw_pix_fmt = frames_ctx->sw_format;
    switch (sw_pix_fmt) {
      case AV_PIX_FMT_RGB0:
      case AV_PIX_FMT_BGR0: {
        ConvertFunc convert_func = [](const ManagedBuffer& t, AVFrame* f) {
          write_interlaced_video_cuda(t, f, 4);
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
      ConvertFunc convert_func = [=](const ManagedBuffer& t, AVFrame* f) {
        write_interlaced_video(t, f, channels);
      };
      return {init_func, convert_func};
    }
    case AV_PIX_FMT_RGB0:
    case AV_PIX_FMT_BGR0: {
      ConvertFunc convert_func = [](const ManagedBuffer& t, AVFrame* f) {
        write_interlaced_video(t, f, 4);
      };
      return {init_func, convert_func};
    }
    case AV_PIX_FMT_YUV444P: {
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
