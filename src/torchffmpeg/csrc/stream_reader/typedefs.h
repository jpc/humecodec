#pragma once

#include "torchffmpeg/csrc/ffmpeg.h"
#include "torchffmpeg/csrc/managed_buffer.h"
#include <iostream>

namespace torchffmpeg {

/// Information about source stream found in the input media.
struct SrcStreamInfo {
  AVMediaType media_type;
  const char* codec_name = "N/A";
  const char* codec_long_name = "N/A";
  const char* fmt_name = "N/A";
  int64_t bit_rate = 0;
  int64_t num_frames = 0;
  int bits_per_sample = 0;
  OptionDict metadata{};
  int time_base_num = 0;
  int time_base_den = 1;

  // Audio
  double sample_rate = 0;
  int num_channels = 0;

  // Video
  int width = 0;
  int height = 0;
  double frame_rate = 0;
};

/// Information about output stream configured by user code
struct OutputStreamInfo {
  int source_index;
  AVMediaType media_type = AVMEDIA_TYPE_UNKNOWN;
  int format = -1;
  std::string filter_description{};

  // Audio
  double sample_rate = -1;
  int num_channels = -1;

  // Video
  int width = -1;
  int height = -1;
  AVRational frame_rate{0, 1};
};

/// Stores decoded frames and metadata
struct Chunk {
  ManagedBuffer frames;
  double pts;
};

/// Lightweight packet index entry for building a seek index.
/// Contains only the information needed to locate and seek to a packet
/// in the file without decoding.
struct PacketIndexEntry {
  int64_t pts;          // Raw presentation timestamp in stream time_base units
  double pts_seconds;   // Presentation timestamp in seconds
  int64_t pos;          // Byte offset of the packet in the file
  int size;             // Packet payload size in bytes
  bool is_key;          // Whether this is a keyframe
};

} // namespace torchffmpeg
