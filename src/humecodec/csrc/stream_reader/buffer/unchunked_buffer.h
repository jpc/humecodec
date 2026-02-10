#pragma once
#include "humecodec/csrc/ffmpeg.h"
#include "humecodec/csrc/managed_buffer.h"
#include "humecodec/csrc/stream_reader/typedefs.h"
#include <deque>

namespace humecodec::detail {

class UnchunkedBuffer {
  // Each AVFrame is converted to a ManagedBuffer and stored here.
  std::deque<ManagedBuffer> chunks;
  double pts = -1.;
  AVRational time_base;

 public:
  explicit UnchunkedBuffer(AVRational time_base);
  bool is_ready() const;
  void push_frame(ManagedBuffer frame, int64_t pts_);
  std::optional<Chunk> pop_chunk();
  void flush();
};

} // namespace humecodec::detail
