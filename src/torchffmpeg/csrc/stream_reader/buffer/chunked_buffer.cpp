#include "torchffmpeg/csrc/stream_reader/buffer/chunked_buffer.h"

#ifdef USE_CUDA
#include "torchffmpeg/csrc/cuda_utils.h"
#endif

namespace torchffmpeg::detail {

ChunkedBuffer::ChunkedBuffer(
    AVRational time_base,
    int frames_per_chunk_,
    int num_chunks_)
    : time_base(time_base),
      frames_per_chunk(frames_per_chunk_),
      num_chunks(num_chunks_){};

bool ChunkedBuffer::is_ready() const {
  return num_buffered_frames >= frames_per_chunk;
}

size_t ChunkedBuffer::frame_bytes(const ManagedBuffer& buf) const {
  // Bytes per "frame" = product of dims 1..N * element_size
  size_t bytes = buf.element_size();
  for (int d = 1; d < buf.ndim(); ++d) {
    bytes *= buf.shape()[d];
  }
  return bytes;
}

void ChunkedBuffer::push_frame(ManagedBuffer frame, int64_t pts_) {
  if (!has_cached_info) {
    cached_dtype = frame.dtype();
    cached_device = frame.device();
    has_cached_info = true;
  }

  size_t fbytes = frame_bytes(frame);
  int64_t num_frames = frame.size(0);

  auto copy_frames = [&](void* dst, const void* src, size_t n_frames) {
    size_t bytes = n_frames * fbytes;
    if (frame.is_cpu()) {
      memcpy(dst, src, bytes);
    } else {
#ifdef USE_CUDA
      cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
#else
      TFMPEG_CHECK(false, "CUDA support not compiled.");
#endif
    }
  };

  // 1. Check if the last chunk is partially filled. If so, fill it.
  if (int64_t filled = num_buffered_frames % frames_per_chunk) {
    TFMPEG_INTERNAL_ASSERT(
        chunks.size() > 0,
        "There is supposed to be left over frames, but the buffer dequeue is empty.");
    int64_t remain = frames_per_chunk - filled;
    int64_t append = remain < num_frames ? remain : num_frames;

    uint8_t* prev_data = chunks.back().data_ptr<uint8_t>();
    const uint8_t* src_data = frame.data_ptr<uint8_t>();
    copy_frames(prev_data + filled * fbytes, src_data, append);
    num_buffered_frames += append;

    // Advance frame pointer
    if (append >= num_frames) {
      return;  // All frames consumed
    }
    // Create a sub-buffer from remaining frames
    num_frames -= append;
    // We need to work with the remaining portion
    const uint8_t* remaining_src = src_data + append * fbytes;

    // Build remaining shape
    std::vector<int64_t> rem_shape = frame.shape();
    rem_shape[0] = num_frames;
    ManagedBuffer remaining(rem_shape, frame.dtype(), frame.device());
    copy_frames(remaining.data(), remaining_src, num_frames);
    frame = std::move(remaining);
    pts_ += append;
  }

  // 2. Return if no frames left
  if (frame.size(0) == 0) {
    return;
  }

  num_frames = frame.size(0);

  // 3. Now the existing buffer chunks are fully filled, start adding new chunks
  int64_t num_splits =
      num_frames / frames_per_chunk + (num_frames % frames_per_chunk ? 1 : 0);
  const uint8_t* src_data = frame.data_ptr<uint8_t>();

  for (int64_t i = 0; i < num_splits; ++i) {
    int64_t start = i * frames_per_chunk;
    int64_t chunk_size = std::min(frames_per_chunk, num_frames - start);
    int64_t pts_val = pts_ + start;

    // Allocate a full chunk (frames_per_chunk frames)
    std::vector<int64_t> chunk_shape = frame.shape();
    chunk_shape[0] = frames_per_chunk;
    ManagedBuffer chunk(chunk_shape, frame.dtype(), frame.device());

    // Copy actual frames
    copy_frames(chunk.data(), src_data + start * fbytes, chunk_size);
    // Remaining slots (if chunk_size < frames_per_chunk) are uninitialized
    // but that's fine — we track num_buffered_frames

    chunks.push_back(std::move(chunk));
    pts.push_back(pts_val);
    num_buffered_frames += chunk_size;

    // Trim if num_chunks > 0
    if (num_chunks > 0 && static_cast<int64_t>(chunks.size()) > num_chunks) {
      TFMPEG_WARN_ONCE(
          "The number of buffered frames exceeded the buffer size. "
          "Dropping the old frames. "
          "To avoid this, you can set a higher buffer_chunk_size value.");
      chunks.pop_front();
      pts.pop_front();
      num_buffered_frames -= frames_per_chunk;
    }
  }
}

std::optional<Chunk> ChunkedBuffer::pop_chunk() {
  if (!num_buffered_frames) {
    return {};
  }

  ManagedBuffer chunk = std::move(chunks.front());
  double pts_val = double(pts.front()) * time_base.num / time_base.den;
  chunks.pop_front();
  pts.pop_front();

  int64_t actual_frames = std::min(num_buffered_frames, frames_per_chunk);
  num_buffered_frames -= actual_frames;

  if (actual_frames < frames_per_chunk) {
    // Need to slice: create a smaller buffer with only the valid frames
    std::vector<int64_t> sliced_shape = chunk.shape();
    sliced_shape[0] = actual_frames;
    size_t fbytes = frame_bytes(chunk);

    ManagedBuffer sliced(sliced_shape, chunk.dtype(), chunk.device());
    size_t copy_bytes = actual_frames * fbytes;
    if (chunk.is_cpu()) {
      memcpy(sliced.data(), chunk.data(), copy_bytes);
    } else {
#ifdef USE_CUDA
      cudaMemcpy(sliced.data(), chunk.data(), copy_bytes, cudaMemcpyDeviceToDevice);
#else
      TFMPEG_CHECK(false, "CUDA support not compiled.");
#endif
    }
    return {Chunk{std::move(sliced), pts_val}};
  }

  return {Chunk{std::move(chunk), pts_val}};
}

void ChunkedBuffer::flush() {
  num_buffered_frames = 0;
  chunks.clear();
  pts.clear();
}

} // namespace torchffmpeg::detail
