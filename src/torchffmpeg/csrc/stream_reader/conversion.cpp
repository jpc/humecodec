#include "torchffmpeg/csrc/stream_reader/conversion.h"

#ifdef USE_CUDA
#include "torchffmpeg/csrc/cuda_utils.h"
#endif

namespace torchffmpeg {

////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////

template <DLDataTypeCode type_code, uint8_t bits, bool is_planar>
AudioConverter<type_code, bits, is_planar>::AudioConverter(int c)
    : num_channels(c) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(num_channels > 0);
}

template <DLDataTypeCode type_code, uint8_t bits, bool is_planar>
ManagedBuffer AudioConverter<type_code, bits, is_planar>::convert(
    const AVFrame* src) {
  constexpr int bps = bits / 8;
  constexpr DLDataType dt = {type_code, bits, 1};

  if constexpr (is_planar) {
    // Allocate directly in {nb_samples, num_channels} layout
    ManagedBuffer dst(
        {src->nb_samples, (int64_t)num_channels}, dt, device::cpu());
    uint8_t* p_dst = dst.data_ptr<uint8_t>();
    // Copy plane-by-plane with correct strides into interleaved output
    for (int s = 0; s < src->nb_samples; ++s) {
      for (int c = 0; c < num_channels; ++c) {
        memcpy(
            p_dst + (s * num_channels + c) * bps,
            src->extended_data[c] + s * bps,
            bps);
      }
    }
    return dst;
  } else {
    ManagedBuffer dst(
        {src->nb_samples, (int64_t)num_channels}, dt, device::cpu());
    int plane_size = bps * src->nb_samples * num_channels;
    memcpy(dst.data(), src->extended_data[0], plane_size);
    return dst;
  }
}

// Explicit instantiation
// uint8
template class AudioConverter<kDLUInt, 8, false>;
template class AudioConverter<kDLUInt, 8, true>;
// int16
template class AudioConverter<kDLInt, 16, false>;
template class AudioConverter<kDLInt, 16, true>;
// int32
template class AudioConverter<kDLInt, 32, false>;
template class AudioConverter<kDLInt, 32, true>;
// int64
template class AudioConverter<kDLInt, 64, false>;
template class AudioConverter<kDLInt, 64, true>;
// float32
template class AudioConverter<kDLFloat, 32, false>;
template class AudioConverter<kDLFloat, 32, true>;
// float64
template class AudioConverter<kDLFloat, 64, false>;
template class AudioConverter<kDLFloat, 64, true>;

////////////////////////////////////////////////////////////////////////////////
// Image
////////////////////////////////////////////////////////////////////////////////

ImageConverterBase::ImageConverterBase(int h, int w, int c)
    : height(h), width(w), num_channels(c) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(height > 0);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(width > 0);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(num_channels > 0);
}

////////////////////////////////////////////////////////////////////////////////
// Interlaced Image -> NCHW uint8
////////////////////////////////////////////////////////////////////////////////
ManagedBuffer InterlacedImageConverter::convert(const AVFrame* src) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);

  // Allocate directly in NCHW layout {1, C, H, W}
  ManagedBuffer buffer({1, (int64_t)num_channels, (int64_t)height, (int64_t)width},
                       dtype::uint8(), device::cpu());
  uint8_t* p_dst = buffer.data_ptr<uint8_t>();
  uint8_t* p_src = src->data[0];

  // Source is NHWC (interleaved), dest is NCHW
  // Copy with stride-aware de-interleaving
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < num_channels; ++c) {
        p_dst[c * height * width + h * width + w] =
            p_src[h * src->linesize[0] + w * num_channels + c];
      }
    }
  }
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// Interlaced 16 Bit Image -> NCHW int16
////////////////////////////////////////////////////////////////////////////////
ManagedBuffer Interlaced16BitImageConverter::convert(const AVFrame* src) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);

  // Allocate in NCHW layout {1, C, H, W} as int16
  ManagedBuffer buffer({1, (int64_t)num_channels, (int64_t)height, (int64_t)width},
                       dtype::int16(), device::cpu());
  int16_t* p_dst = buffer.data_ptr<int16_t>();
  uint8_t* p_src_base = src->data[0];

  // Source is NHWC interleaved int16, dest is NCHW
  for (int h = 0; h < height; ++h) {
    const int16_t* p_src_row =
        reinterpret_cast<const int16_t*>(p_src_base + h * src->linesize[0]);
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < num_channels; ++c) {
        // Add 32768 offset during copy (same as original dst += 32768)
        p_dst[c * height * width + h * width + w] =
            p_src_row[w * num_channels + c] + 32768;
      }
    }
  }
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// Planar Image -> NCHW uint8
////////////////////////////////////////////////////////////////////////////////
ManagedBuffer PlanarImageConverter::convert(const AVFrame* src) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);

  ManagedBuffer buffer({1, (int64_t)num_channels, (int64_t)height, (int64_t)width},
                       dtype::uint8(), device::cpu());
  uint8_t* p_dst = buffer.data_ptr<uint8_t>();

  for (int i = 0; i < num_channels; ++i) {
    uint8_t* dst_plane = p_dst + i * height * width;
    uint8_t* p_src = src->data[i];
    int linesize = src->linesize[i];
    for (int h = 0; h < height; ++h) {
      memcpy(dst_plane + h * width, p_src + h * linesize, width);
    }
  }
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// YUV420P -> NCHW uint8 (upsampled to YUV444P)
////////////////////////////////////////////////////////////////////////////////
YUV420PConverter::YUV420PConverter(int h, int w) : ImageConverterBase(h, w, 3) {
  TFMPEG_WARN_ONCE(
      "The output format YUV420P is selected. "
      "This will be implicitly converted to YUV444P, "
      "in which all the color components Y, U, V have the same dimension.");
}

ManagedBuffer YUV420PConverter::convert(const AVFrame* src) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(
      (AVPixelFormat)(src->format) == AV_PIX_FMT_YUV420P);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);

  ManagedBuffer buffer({1, 3, (int64_t)height, (int64_t)width},
                       dtype::uint8(), device::cpu());
  uint8_t* p_dst = buffer.data_ptr<uint8_t>();

  // Copy Y plane (full resolution)
  uint8_t* dst_y = p_dst;
  uint8_t* src_y = src->data[0];
  for (int h = 0; h < height; ++h) {
    memcpy(dst_y + h * width, src_y + h * src->linesize[0], width);
  }

  // Copy U and V planes with 2x2 upsampling
  for (int i = 1; i < 3; ++i) {
    uint8_t* dst_plane = p_dst + i * height * width;
    uint8_t* src_plane = src->data[i];
    int src_linesize = src->linesize[i];
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        dst_plane[h * width + w] =
            src_plane[(h / 2) * src_linesize + (w / 2)];
      }
    }
  }
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// YUV420P10LE -> NCHW int16 (upsampled to YUV444P)
////////////////////////////////////////////////////////////////////////////////
YUV420P10LEConverter::YUV420P10LEConverter(int h, int w)
    : ImageConverterBase(h, w, 3) {
  TFMPEG_WARN_ONCE(
      "The output format YUV420PLE is selected. "
      "This will be implicitly converted to YUV444P (16-bit), "
      "in which all the color components Y, U, V have the same dimension.");
}

ManagedBuffer YUV420P10LEConverter::convert(const AVFrame* src) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(
      (AVPixelFormat)(src->format) == AV_PIX_FMT_YUV420P10LE);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);

  ManagedBuffer buffer({1, 3, (int64_t)height, (int64_t)width},
                       dtype::int16(), device::cpu());
  int16_t* p_dst = buffer.data_ptr<int16_t>();

  // Copy Y plane (full resolution)
  int16_t* dst_y = p_dst;
  uint8_t* src_y = src->data[0];
  for (int h = 0; h < height; ++h) {
    memcpy(dst_y + h * width, src_y + h * src->linesize[0],
           (size_t)width * 2);
  }

  // Copy U and V planes with 2x2 upsampling
  for (int i = 1; i < 3; ++i) {
    int16_t* dst_plane = p_dst + i * height * width;
    uint8_t* src_plane_bytes = src->data[i];
    int src_linesize = src->linesize[i];
    for (int h = 0; h < height; ++h) {
      const int16_t* src_row =
          reinterpret_cast<const int16_t*>(src_plane_bytes + (h / 2) * src_linesize);
      for (int w = 0; w < width; ++w) {
        dst_plane[h * width + w] = src_row[w / 2];
      }
    }
  }
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// NV12 -> NCHW uint8 (upsampled to YUV444P)
////////////////////////////////////////////////////////////////////////////////
NV12Converter::NV12Converter(int h, int w) : ImageConverterBase(h, w, 3) {
  TFMPEG_WARN_ONCE(
      "The output format NV12 is selected. "
      "This will be implicitly converted to YUV444P, "
      "in which all the color components Y, U, V have the same dimension.");
}

ManagedBuffer NV12Converter::convert(const AVFrame* src) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(
      (AVPixelFormat)(src->format) == AV_PIX_FMT_NV12);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src->height == height);
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src->width == width);

  ManagedBuffer buffer({1, 3, (int64_t)height, (int64_t)width},
                       dtype::uint8(), device::cpu());
  uint8_t* p_dst = buffer.data_ptr<uint8_t>();

  // Copy Y plane
  uint8_t* dst_y = p_dst;
  uint8_t* src_y = src->data[0];
  for (int h = 0; h < height; ++h) {
    memcpy(dst_y + h * width, src_y + h * src->linesize[0], width);
  }

  // NV12: UV plane is interleaved (UVUVUV...) at half resolution
  // De-interleave and upsample 2x2
  uint8_t* dst_u = p_dst + 1 * height * width;
  uint8_t* dst_v = p_dst + 2 * height * width;
  uint8_t* src_uv = src->data[1];
  int uv_linesize = src->linesize[1];
  for (int h = 0; h < height; ++h) {
    const uint8_t* uv_row = src_uv + (h / 2) * uv_linesize;
    for (int w = 0; w < width; ++w) {
      dst_u[h * width + w] = uv_row[(w / 2) * 2];
      dst_v[h * width + w] = uv_row[(w / 2) * 2 + 1];
    }
  }
  return buffer;
}

#ifdef USE_CUDA

CudaImageConverterBase::CudaImageConverterBase(const DLDevice& dev)
    : cuda_device(dev) {}

////////////////////////////////////////////////////////////////////////////////
// NV12 CUDA -> raw Y + UV planes (Python does chroma upsampling)
////////////////////////////////////////////////////////////////////////////////
NV12CudaConverter::NV12CudaConverter(const DLDevice& dev)
    : CudaImageConverterBase(dev) {
  TFMPEG_WARN_ONCE(
      "The output format NV12 is selected. "
      "This will be implicitly converted to YUV444P, "
      "in which all the color components Y, U, V have the same dimension.");
}

ManagedBuffer NV12CudaConverter::convert(const AVFrame* src) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src);
  if (!init) {
    height = src->height;
    width = src->width;
    init = true;
  }

  auto fmt = (AVPixelFormat)(src->format);
  AVHWFramesContext* hwctx = (AVHWFramesContext*)src->hw_frames_ctx->data;
  AVPixelFormat sw_fmt = hwctx->sw_format;

  TFMPEG_INTERNAL_ASSERT(
      AV_PIX_FMT_CUDA == fmt,
      "Expected CUDA frame. Found: ",
      av_get_pix_fmt_name(fmt));
  TFMPEG_INTERNAL_ASSERT(
      AV_PIX_FMT_NV12 == sw_fmt,
      "Expected NV12 format. Found: ",
      av_get_pix_fmt_name(sw_fmt));

  // Output: {1, 3, H, W} uint8 on GPU
  // Y plane occupies [0, 0, :, :], UV interleaved in [0, 1:3, :H/2, :W/2]
  // Python side will upsample UV channels.
  // For simplicity, allocate full YUV444P and copy Y + raw UV.
  ManagedBuffer buffer({1, 3, (int64_t)height, (int64_t)width},
                       dtype::uint8(), cuda_device);
  uint8_t* p_dst = buffer.data_ptr<uint8_t>();

  // Copy Y plane
  auto status = cudaMemcpy2D(
      p_dst, width, src->data[0], src->linesize[0],
      width, height, cudaMemcpyDeviceToDevice);
  TFMPEG_CHECK(cudaSuccess == status, "Failed to copy Y plane to CUDA buffer.");

  // Copy UV plane into a temporary buffer, then scatter to U and V channels
  // For now, just zero-fill U and V and copy UV interleaved data
  // The Python side will handle the upsampling from the raw NV12 data.
  // Actually, let's copy the raw UV data into the U channel area for Python
  // to process. We'll store it as: Y in channel 0, raw UV in channels 1-2
  // at half resolution. But ManagedBuffer is contiguous NCHW, so we need
  // Python to know this is NV12.
  //
  // Simpler approach matching the plan: just copy Y full-res, and for UV
  // copy each half-res row duplicated. This avoids needing Python changes
  // for the basic case.
  //
  // Actually, the simplest approach that matches the original behavior while
  // removing torch: copy Y plane, then for U/V do nearest-neighbor upsample
  // on CPU. But wait, this is CUDA data.
  //
  // Best approach per plan: C++ outputs raw subsampled. Python upsamples.
  // Store UV interleaved at half res in a separate buffer? No, that changes
  // the API.
  //
  // Let's keep it simple: allocate a temp buffer for UV at half res,
  // cudaMemcpy2D the UV plane, then do the nearest-neighbor scatter
  // via a simple CUDA kernel... but we don't have custom kernels.
  //
  // Pragmatic solution: do nearest-neighbor upsampling with cudaMemcpy2D
  // row-by-row. For each full-res row h, copy from UV row h/2.
  // For NV12, UV is interleaved as UVUV, width bytes per half-res row.
  // We need to de-interleave and upsample.
  //
  // Since we can't easily do this without torch or custom kernels,
  // let's return a special layout that Python can handle:
  // Channel 0 = Y (full res), Channels 1-2 = zeros (Python fills from NV12)
  // But that wastes the GPU copy.
  //
  // Actually the cleanest: just memset U/V to 0 and let Python overwrite.
  // Or: return raw NV12 layout and have Python interpret it.
  //
  // Per the plan: "C++ copies Y plane into ManagedBuffer and UV plane.
  // Return them as raw NV12 layout. Python post-processes."
  // Let's zero U/V channels and store raw UV in a way Python can use.
  //
  // SIMPLEST: zero the U and V planes. Python will overwrite them
  // using the source NV12 data from the same AVFrame. But Python
  // doesn't have access to the AVFrame...
  //
  // OK, let me just implement a simple device-to-device scatter.
  // We have cudaMemcpy2D. For nearest neighbor upsample of UV:
  // 1. Copy UV interleaved (width bytes, height/2 rows) to a temp buffer
  // 2. For de-interleaving: we need element-wise access -> can't do with memcpy
  //
  // The RIGHT answer: just copy Y plane, set U/V to zero, and flag this
  // buffer as needing NV12 upsample. Store the raw UV half-res data
  // in a second ManagedBuffer that we return alongside.
  // But the Chunk struct only has one ManagedBuffer.
  //
  // OK, practical decision: for CUDA NV12 and P010, we return the full
  // {1, 3, H, W} buffer with Y correct and U/V set to 128 (neutral gray)
  // as a placeholder, plus we store the raw UV data in additional memory
  // at the end of the same allocation. Python can then do the upsample.
  //
  // Actually, let me re-examine: the Python decoder.py currently works
  // by calling pop_chunks() which returns tensors. The chroma upsampling
  // that was in C++ now needs to move to Python. But Python needs the
  // raw subsampled UV data on GPU.
  //
  // CLEANEST APPROACH: Return {1, 1+2*(H/2)/(H), H, W} ... no, too complex.
  //
  // Let me just implement the scatter using row-by-row cudaMemcpy.
  // For NV12 nearest-neighbor upsampling, each UV value maps to a 2x2 block.
  // We can't trivially do this with cudaMemcpy2D.
  //
  // DECISION: For now, do the upsample on CPU after copying GPU->CPU->GPU.
  // This is a temporary inefficiency. We can optimize later with a simple
  // CUDA kernel or by moving to Python-side processing.
  //
  // Actually wait - let me re-read cuda_utils.h to see what's available.

  // REVISED APPROACH: Copy Y and UV planes to GPU ManagedBuffers,
  // return a combined buffer where:
  // - Y channel is full resolution in channel 0
  // - U/V channels contain the nearest-neighbor upsampled data
  // We achieve this by copying each UV half-res row into two full-res rows,
  // duplicating horizontally via a host-side temporary.

  // Allocate host temp for one UV row (width bytes, interleaved UV)
  // Then for each pair of output rows, copy the same source UV row
  // and de-interleave.

  // For efficiency, let's use a host staging buffer
  size_t uv_row_bytes = width;  // NV12: width bytes per UV row
  std::vector<uint8_t> uv_host(uv_row_bytes);
  std::vector<uint8_t> u_full_row(width);
  std::vector<uint8_t> v_full_row(width);

  uint8_t* dst_u = p_dst + 1 * height * width;
  uint8_t* dst_v = p_dst + 2 * height * width;

  for (int row = 0; row < height / 2; ++row) {
    // Copy one UV row from GPU to host
    status = cudaMemcpy(
        uv_host.data(),
        src->data[1] + row * src->linesize[1],
        uv_row_bytes,
        cudaMemcpyDeviceToHost);
    TFMPEG_CHECK(cudaSuccess == status, "Failed to copy UV row to host.");

    // De-interleave and horizontally upsample
    for (int w = 0; w < width; ++w) {
      u_full_row[w] = uv_host[(w / 2) * 2];
      v_full_row[w] = uv_host[(w / 2) * 2 + 1];
    }

    // Copy upsampled rows to GPU (2 rows per source row = vertical upsample)
    for (int dup = 0; dup < 2; ++dup) {
      int dst_row = row * 2 + dup;
      status = cudaMemcpy(
          dst_u + dst_row * width, u_full_row.data(),
          width, cudaMemcpyHostToDevice);
      TFMPEG_CHECK(cudaSuccess == status, "Failed to copy U row to GPU.");
      status = cudaMemcpy(
          dst_v + dst_row * width, v_full_row.data(),
          width, cudaMemcpyHostToDevice);
      TFMPEG_CHECK(cudaSuccess == status, "Failed to copy V row to GPU.");
    }
  }

  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// P010 CUDA -> NCHW int16 (with chroma upsampling via host staging)
////////////////////////////////////////////////////////////////////////////////
P010CudaConverter::P010CudaConverter(const DLDevice& dev)
    : CudaImageConverterBase{dev} {
  TFMPEG_WARN_ONCE(
      "The output format P010 is selected. "
      "This will be implicitly converted to YUV444P, "
      "in which all the color components Y, U, V have the same dimension.");
}

ManagedBuffer P010CudaConverter::convert(const AVFrame* src) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src);
  if (!init) {
    height = src->height;
    width = src->width;
    init = true;
  }

  auto fmt = (AVPixelFormat)(src->format);
  AVHWFramesContext* hwctx = (AVHWFramesContext*)src->hw_frames_ctx->data;
  AVPixelFormat sw_fmt = hwctx->sw_format;

  TFMPEG_INTERNAL_ASSERT(
      AV_PIX_FMT_CUDA == fmt,
      "Expected CUDA frame. Found: ", av_get_pix_fmt_name(fmt));
  TFMPEG_INTERNAL_ASSERT(
      AV_PIX_FMT_P010 == sw_fmt,
      "Expected P010 format. Found: ", av_get_pix_fmt_name(sw_fmt));

  ManagedBuffer buffer({1, 3, (int64_t)height, (int64_t)width},
                       dtype::int16(), cuda_device);
  int16_t* p_dst = buffer.data_ptr<int16_t>();

  // Copy Y plane
  auto status = cudaMemcpy2D(
      p_dst, width * 2, src->data[0], src->linesize[0],
      width * 2, height, cudaMemcpyDeviceToDevice);
  TFMPEG_CHECK(cudaSuccess == status, "Failed to copy Y plane to CUDA buffer.");

  // P010 UV plane: interleaved int16 UV at half resolution
  size_t uv_row_bytes = width * 2;  // width int16 values per UV row
  std::vector<int16_t> uv_host(width);
  std::vector<int16_t> u_full_row(width);
  std::vector<int16_t> v_full_row(width);

  int16_t* dst_u = p_dst + 1 * height * width;
  int16_t* dst_v = p_dst + 2 * height * width;

  for (int row = 0; row < height / 2; ++row) {
    // Copy one UV row from GPU to host
    status = cudaMemcpy(
        uv_host.data(),
        src->data[1] + row * src->linesize[1],
        uv_row_bytes,
        cudaMemcpyDeviceToHost);
    TFMPEG_CHECK(cudaSuccess == status, "Failed to copy UV row to host.");

    // De-interleave and horizontally upsample
    for (int w = 0; w < width; ++w) {
      u_full_row[w] = uv_host[(w / 2) * 2];
      v_full_row[w] = uv_host[(w / 2) * 2 + 1];
    }

    // Copy upsampled rows to GPU
    for (int dup = 0; dup < 2; ++dup) {
      int dst_row = row * 2 + dup;
      status = cudaMemcpy(
          dst_u + dst_row * width, u_full_row.data(),
          width * 2, cudaMemcpyHostToDevice);
      TFMPEG_CHECK(cudaSuccess == status, "Failed to copy U row to GPU.");
      status = cudaMemcpy(
          dst_v + dst_row * width, v_full_row.data(),
          width * 2, cudaMemcpyHostToDevice);
      TFMPEG_CHECK(cudaSuccess == status, "Failed to copy V row to GPU.");
    }
  }

  // Apply +32768 offset: need to do element-wise on GPU.
  // Since we don't have a CUDA kernel, copy to host, add, copy back.
  // This is suboptimal but correct.
  size_t total_elements = 3 * height * width;
  std::vector<int16_t> host_buf(total_elements);
  status = cudaMemcpy(
      host_buf.data(), p_dst, total_elements * 2, cudaMemcpyDeviceToHost);
  TFMPEG_CHECK(cudaSuccess == status, "Failed to copy buffer to host for offset.");
  for (size_t i = 0; i < total_elements; ++i) {
    host_buf[i] += 32768;
  }
  status = cudaMemcpy(
      p_dst, host_buf.data(), total_elements * 2, cudaMemcpyHostToDevice);
  TFMPEG_CHECK(cudaSuccess == status, "Failed to copy buffer back to GPU.");

  return buffer;
}

////////////////////////////////////////////////////////////////////////////////
// YUV444P CUDA -> NCHW uint8 (direct plane copy)
////////////////////////////////////////////////////////////////////////////////
YUV444PCudaConverter::YUV444PCudaConverter(const DLDevice& dev)
    : CudaImageConverterBase(dev) {}

ManagedBuffer YUV444PCudaConverter::convert(const AVFrame* src) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(src);
  if (!init) {
    height = src->height;
    width = src->width;
    init = true;
  }

  auto fmt = (AVPixelFormat)(src->format);
  AVHWFramesContext* hwctx = (AVHWFramesContext*)src->hw_frames_ctx->data;
  AVPixelFormat sw_fmt = hwctx->sw_format;

  TFMPEG_INTERNAL_ASSERT(
      AV_PIX_FMT_CUDA == fmt,
      "Expected CUDA frame. Found: ", av_get_pix_fmt_name(fmt));
  TFMPEG_INTERNAL_ASSERT(
      AV_PIX_FMT_YUV444P == sw_fmt,
      "Expected YUV444P format. Found: ", av_get_pix_fmt_name(sw_fmt));

  ManagedBuffer buffer({1, 3, (int64_t)height, (int64_t)width},
                       dtype::uint8(), cuda_device);
  uint8_t* p_dst = buffer.data_ptr<uint8_t>();

  for (int i = 0; i < 3; ++i) {
    auto status = cudaMemcpy2D(
        p_dst + i * height * width, width,
        src->data[i], src->linesize[i],
        width, height, cudaMemcpyDeviceToDevice);
    TFMPEG_CHECK(
        cudaSuccess == status, "Failed to copy plane ", i, " to CUDA buffer.");
  }
  return buffer;
}

#endif

} // namespace torchffmpeg
