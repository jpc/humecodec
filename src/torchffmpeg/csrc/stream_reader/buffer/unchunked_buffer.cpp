#include "torchffmpeg/csrc/stream_reader/buffer/unchunked_buffer.h"

namespace torchffmpeg::detail {

UnchunkedBuffer::UnchunkedBuffer(AVRational time_base) : time_base(time_base){};

bool UnchunkedBuffer::is_ready() const {
  return chunks.size() > 0;
}

void UnchunkedBuffer::push_frame(ManagedBuffer frame, int64_t pts_) {
  if (chunks.size() == 0) {
    pts = double(pts_) * time_base.num / time_base.den;
  }
  chunks.push_back(std::move(frame));
}

std::optional<Chunk> UnchunkedBuffer::pop_chunk() {
  if (chunks.size() == 0) {
    return {};
  }

  // Concatenate all chunks along dimension 0
  // First compute total size along dim 0
  int64_t total_dim0 = 0;
  for (const auto& c : chunks) {
    total_dim0 += c.size(0);
  }

  // Build output shape: same as first chunk but with summed dim 0
  auto& first = chunks.front();
  std::vector<int64_t> shape = first.shape();
  shape[0] = total_dim0;

  ManagedBuffer result(shape, first.dtype(), first.device());

  // Copy data sequentially
  size_t elem_size = first.element_size();
  int64_t stride0_bytes = 1;
  for (int d = 1; d < first.ndim(); ++d) {
    stride0_bytes *= first.shape()[d];
  }
  stride0_bytes *= elem_size;

  uint8_t* dst = result.data_ptr<uint8_t>();
  for (auto& c : chunks) {
    size_t chunk_bytes = c.size(0) * stride0_bytes;
    if (c.is_cpu()) {
      memcpy(dst, c.data(), chunk_bytes);
    } else {
#ifdef USE_CUDA
      cudaMemcpy(dst, c.data(), chunk_bytes, cudaMemcpyDeviceToDevice);
#else
      TFMPEG_CHECK(false, "CUDA support not compiled.");
#endif
    }
    dst += chunk_bytes;
  }
  chunks.clear();
  return {Chunk{std::move(result), pts}};
}

void UnchunkedBuffer::flush() {
  chunks.clear();
}

} // namespace torchffmpeg::detail
