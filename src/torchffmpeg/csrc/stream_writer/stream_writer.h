#pragma once

extern "C" {
#include <libavformat/version.h>
}

#include "torchffmpeg/csrc/ffmpeg.h"
#include "torchffmpeg/csrc/filter_graph.h"
#include "torchffmpeg/csrc/managed_buffer.h"
#include "torchffmpeg/csrc/stream_writer/encode_process.h"
#include "torchffmpeg/csrc/stream_writer/packet_writer.h"
#include "torchffmpeg/csrc/stream_writer/types.h"

namespace torchffmpeg {

////////////////////////////////////////////////////////////////////////////////
// StreamingMediaEncoder
////////////////////////////////////////////////////////////////////////////////

class StreamingMediaEncoder {
  AVFormatOutputContextPtr format_ctx;
  std::map<int, EncodeProcess> processes;
  std::map<int, PacketWriter> packet_writers;

  AVPacketPtr pkt{alloc_avpacket()};
  bool is_open = false;
  int current_key = 0;

 private:
  explicit StreamingMediaEncoder(AVFormatContext*);

 protected:
  explicit StreamingMediaEncoder(
      AVIOContext* io_ctx,
      const std::optional<std::string>& format = std::nullopt);

 public:
  explicit StreamingMediaEncoder(
      const std::string& dst,
      const std::optional<std::string>& format = std::nullopt);

  // Non-copyable
  StreamingMediaEncoder(const StreamingMediaEncoder&) = delete;
  StreamingMediaEncoder& operator=(const StreamingMediaEncoder&) = delete;

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  void dump_format(int64_t i);

  //////////////////////////////////////////////////////////////////////////////
  // Configure methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  void add_audio_stream(
      int sample_rate,
      int num_channels,
      const std::string& format,
      const std::optional<std::string>& encoder = std::nullopt,
      const std::optional<OptionDict>& encoder_option = std::nullopt,
      const std::optional<std::string>& encoder_format = std::nullopt,
      const std::optional<int>& encoder_sample_rate = std::nullopt,
      const std::optional<int>& encoder_num_channels = std::nullopt,
      const std::optional<CodecConfig>& codec_config = std::nullopt,
      const std::optional<std::string>& filter_desc = std::nullopt);

  void add_video_stream(
      double frame_rate,
      int width,
      int height,
      const std::string& format,
      const std::optional<std::string>& encoder = std::nullopt,
      const std::optional<OptionDict>& encoder_option = std::nullopt,
      const std::optional<std::string>& encoder_format = std::nullopt,
      const std::optional<double>& encoder_frame_rate = std::nullopt,
      const std::optional<int>& encoder_width = std::nullopt,
      const std::optional<int>& encoder_height = std::nullopt,
      const std::optional<std::string>& hw_accel = std::nullopt,
      const std::optional<CodecConfig>& codec_config = std::nullopt,
      const std::optional<std::string>& filter_desc = std::nullopt);

  void add_audio_frame_stream(
      int sample_rate,
      int num_channels,
      const std::string& format,
      const std::optional<std::string>& encoder = std::nullopt,
      const std::optional<OptionDict>& encoder_option = std::nullopt,
      const std::optional<std::string>& encoder_format = std::nullopt,
      const std::optional<int>& encoder_sample_rate = std::nullopt,
      const std::optional<int>& encoder_num_channels = std::nullopt,
      const std::optional<CodecConfig>& codec_config = std::nullopt,
      const std::optional<std::string>& filter_desc = std::nullopt);

  void add_video_frame_stream(
      double frame_rate,
      int width,
      int height,
      const std::string& format,
      const std::optional<std::string>& encoder = std::nullopt,
      const std::optional<OptionDict>& encoder_option = std::nullopt,
      const std::optional<std::string>& encoder_format = std::nullopt,
      const std::optional<double>& encoder_frame_rate = std::nullopt,
      const std::optional<int>& encoder_width = std::nullopt,
      const std::optional<int>& encoder_height = std::nullopt,
      const std::optional<std::string>& hw_accel = std::nullopt,
      const std::optional<CodecConfig>& codec_config = std::nullopt,
      const std::optional<std::string>& filter_desc = std::nullopt);

  void add_packet_stream(const StreamParams& stream_params);

  void set_metadata(const OptionDict& metadata);

  //////////////////////////////////////////////////////////////////////////////
  // Write methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  void open(const std::optional<OptionDict>& opt = std::nullopt);
  void close();

  void write_audio_chunk(
      int i,
      const ManagedBuffer& frames,
      const std::optional<double>& pts = std::nullopt);
  void write_video_chunk(
      int i,
      const ManagedBuffer& frames,
      const std::optional<double>& pts = std::nullopt);
  void write_frame(int i, AVFrame* frame);
  void write_packet(const AVPacketPtr& packet);

  void flush();

 private:
  int num_output_streams();
};

////////////////////////////////////////////////////////////////////////////////
// StreamingMediaEncoderCustomIO
////////////////////////////////////////////////////////////////////////////////

namespace detail {
struct CustomOutput {
  AVIOContextPtr io_ctx;
#if LIBAVFORMAT_VERSION_MAJOR >= 61
  CustomOutput(
      void* opaque,
      int buffer_size,
      int (*write_packet)(void* opaque, const uint8_t* buf, int buf_size),
      int64_t (*seek)(void* opaque, int64_t offset, int whence));
#else
  CustomOutput(
      void* opaque,
      int buffer_size,
      int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
      int64_t (*seek)(void* opaque, int64_t offset, int whence));
#endif
};
} // namespace detail

class StreamingMediaEncoderCustomIO : private detail::CustomOutput,
                                      public StreamingMediaEncoder {
 public:
#if LIBAVFORMAT_VERSION_MAJOR >= 61
  StreamingMediaEncoderCustomIO(
      void* opaque,
      const std::optional<std::string>& format,
      int buffer_size,
      int (*write_packet)(void* opaque, const uint8_t* buf, int buf_size),
      int64_t (*seek)(void* opaque, int64_t offset, int whence) = nullptr);
#else
  StreamingMediaEncoderCustomIO(
      void* opaque,
      const std::optional<std::string>& format,
      int buffer_size,
      int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
      int64_t (*seek)(void* opaque, int64_t offset, int whence) = nullptr);
#endif
};

} // namespace torchffmpeg
