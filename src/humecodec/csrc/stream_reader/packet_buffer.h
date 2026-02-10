#pragma once
#include "humecodec/csrc/ffmpeg.h"

namespace humecodec {

class PacketBuffer {
 public:
  void push_packet(AVPacket* packet);
  std::vector<AVPacketPtr> pop_packets();
  bool has_packets();

 private:
  std::deque<AVPacketPtr> packets;
};

} // namespace humecodec
