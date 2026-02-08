#pragma once
#include "torchffmpeg/csrc/ffmpeg.h"
#include "torchffmpeg/csrc/managed_buffer.h"
#include "torchffmpeg/csrc/stream_reader/typedefs.h"
#include <deque>

namespace torchffmpeg::detail {

class ChunkedBuffer {
  // Each AVFrame is converted to a ManagedBuffer and stored here.
  std::deque<ManagedBuffer> chunks;
  // Time stamps corresponding the first frame of each chunk
  std::deque<int64_t> pts;
  AVRational time_base;

  // The number of frames to return as a chunk
  // If <0, then user wants to receive all the frames
  const int64_t frames_per_chunk;
  // The number of chunks to retain
  const int64_t num_chunks;
  // The number of currently stored chunks
  // For video, one ManagedBuffer corresponds to one frame, but for audio,
  // one ManagedBuffer contains multiple samples, so we track here.
  int64_t num_buffered_frames = 0;

  // Cache dtype and device from first frame for allocating new buffers
  DLDataType cached_dtype{};
  DLDevice cached_device{};
  bool has_cached_info = false;

  // Helper: bytes per frame (product of dims 1..N * element_size)
  size_t frame_bytes(const ManagedBuffer& buf) const;

 public:
  ChunkedBuffer(AVRational time_base, int frames_per_chunk, int num_chunks);

  bool is_ready() const;
  void flush();
  std::optional<Chunk> pop_chunk();
  void push_frame(ManagedBuffer frame, int64_t pts_);
};

} // namespace torchffmpeg::detail
