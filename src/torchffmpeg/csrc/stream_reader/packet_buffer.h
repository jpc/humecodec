#pragma once
#include "torchffmpeg/csrc/ffmpeg.h"

namespace torchffmpeg {

class PacketBuffer {
 public:
  void push_packet(AVPacket* packet);
  std::vector<AVPacketPtr> pop_packets();
  bool has_packets();

 private:
  std::deque<AVPacketPtr> packets;
};

} // namespace torchffmpeg
