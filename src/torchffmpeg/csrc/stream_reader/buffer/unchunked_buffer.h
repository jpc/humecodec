#pragma once
#include "torchffmpeg/csrc/ffmpeg.h"
#include "torchffmpeg/csrc/stream_reader/typedefs.h"
#include <torch/types.h>
#include <deque>

namespace torchffmpeg::detail {

class UnchunkedBuffer {
  // Each AVFrame is converted to a Tensor and stored here.
  std::deque<torch::Tensor> chunks;
  double pts = -1.;
  AVRational time_base;

 public:
  explicit UnchunkedBuffer(AVRational time_base);
  bool is_ready() const;
  void push_frame(torch::Tensor frame, int64_t pts_);
  std::optional<Chunk> pop_chunk();
  void flush();
};

} // namespace torchffmpeg::detail
