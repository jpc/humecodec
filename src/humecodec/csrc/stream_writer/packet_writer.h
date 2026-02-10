#pragma once
#include "humecodec/csrc/ffmpeg.h"

namespace humecodec {
class PacketWriter {
  AVFormatContext* format_ctx;
  AVStream* stream;
  AVRational original_time_base;

 public:
  PacketWriter(
      AVFormatContext* format_ctx_,
      const StreamParams& stream_params_);
  void write_packet(const AVPacketPtr& packet);
};
} // namespace humecodec
