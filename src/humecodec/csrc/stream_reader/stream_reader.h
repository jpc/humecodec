#pragma once
#include "humecodec/csrc/ffmpeg.h"
#include "humecodec/csrc/stream_reader/packet_buffer.h"
#include "humecodec/csrc/stream_reader/stream_processor.h"
#include "humecodec/csrc/stream_reader/typedefs.h"
#include <vector>

namespace humecodec {

//////////////////////////////////////////////////////////////////////////////
// StreamingMediaDecoder
//////////////////////////////////////////////////////////////////////////////

class StreamingMediaDecoder {
  AVFormatInputContextPtr format_ctx;
  AVPacketPtr packet{alloc_avpacket()};

  std::vector<std::unique_ptr<StreamProcessor>> processors;
  std::vector<std::pair<int, int>> stream_indices;

  std::unique_ptr<PacketBuffer> packet_buffer;
  std::unordered_set<int> packet_stream_indices;

  int64_t seek_timestamp = 0;

 private:
  explicit StreamingMediaDecoder(AVFormatContext* format_ctx);

 protected:
  explicit StreamingMediaDecoder(
      AVIOContext* io_ctx,
      const std::optional<std::string>& format = std::nullopt,
      const std::optional<OptionDict>& option = std::nullopt);

 public:
  explicit StreamingMediaDecoder(
      const std::string& src,
      const std::optional<std::string>& format = std::nullopt,
      const std::optional<OptionDict>& option = std::nullopt);

  ~StreamingMediaDecoder() = default;
  // Non-copyable
  StreamingMediaDecoder(const StreamingMediaDecoder&) = delete;
  StreamingMediaDecoder& operator=(const StreamingMediaDecoder&) = delete;
  // Movable
  StreamingMediaDecoder(StreamingMediaDecoder&&) = default;
  StreamingMediaDecoder& operator=(StreamingMediaDecoder&&) = default;

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  int64_t find_best_audio_stream() const;
  int64_t find_best_video_stream() const;
  OptionDict get_metadata() const;
  int64_t num_src_streams() const;
  SrcStreamInfo get_src_stream_info(int i) const;
  int64_t num_out_streams() const;
  OutputStreamInfo get_out_stream_info(int i) const;
  bool is_buffer_ready() const;

  StreamParams get_src_stream_params(int i);

  //////////////////////////////////////////////////////////////////////////////
  // Configure methods
  //////////////////////////////////////////////////////////////////////////////
  void add_audio_stream(
      int64_t i,
      int64_t frames_per_chunk,
      int64_t num_chunks,
      const std::optional<std::string>& filter_desc = std::nullopt,
      const std::optional<std::string>& decoder = std::nullopt,
      const std::optional<OptionDict>& decoder_option = std::nullopt);

  void add_video_stream(
      int64_t i,
      int64_t frames_per_chunk,
      int64_t num_chunks,
      const std::optional<std::string>& filter_desc = std::nullopt,
      const std::optional<std::string>& decoder = std::nullopt,
      const std::optional<OptionDict>& decoder_option = std::nullopt,
      const std::optional<std::string>& hw_accel = std::nullopt);

  void add_packet_stream(int i);

  void remove_stream(int64_t i);

 private:
  void add_stream(
      int i,
      AVMediaType media_type,
      int frames_per_chunk,
      int num_chunks,
      const std::string& filter_desc,
      const std::optional<std::string>& decoder,
      const std::optional<OptionDict>& decoder_option,
      const DLDevice& device);

  //////////////////////////////////////////////////////////////////////////////
  // Stream methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  void seek(double timestamp, int64_t mode);

  /// Seek directly to a byte offset in the file.  This is intended for use
  /// with positions obtained from build_packet_index() and avoids any
  /// container-level scanning.
  void seek_to_byte_offset(int64_t offset);

  int process_packet();

  int process_packet_block(const double timeout, const double backoff);

  int process_packet(
      const std::optional<double>& timeout,
      const double backoff);

  void process_all_packets();

  int fill_buffer(
      const std::optional<double>& timeout = std::nullopt,
      const double backoff = 10.);

 private:
  int drain();

  //////////////////////////////////////////////////////////////////////////////
  // Retrieval
  //////////////////////////////////////////////////////////////////////////////
 public:
  std::vector<std::optional<Chunk>> pop_chunks();

  std::vector<AVPacketPtr> pop_packets();

  /// Scan all packets for the given source stream and return a lightweight
  /// seek index (pts in seconds, byte offset, size, keyframe flag) without
  /// decoding.  The format context is rewound to the beginning before
  /// scanning, and the caller should seek() afterwards if they intend to
  /// continue decoding from a specific position.
  std::vector<PacketIndexEntry> build_packet_index(int stream_index);
};

//////////////////////////////////////////////////////////////////////////////
// StreamingMediaDecoderCustomIO
//////////////////////////////////////////////////////////////////////////////

namespace detail {
struct CustomInput {
  AVIOContextPtr io_ctx;
  CustomInput(
      void* opaque,
      int buffer_size,
      int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
      int64_t (*seek)(void* opaque, int64_t offset, int whence));
};
} // namespace detail

class StreamingMediaDecoderCustomIO : private detail::CustomInput,
                                      public StreamingMediaDecoder {
 public:
  StreamingMediaDecoderCustomIO(
      void* opaque,
      const std::optional<std::string>& format,
      int buffer_size,
      int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
      int64_t (*seek)(void* opaque, int64_t offset, int whence) = nullptr,
      const std::optional<OptionDict>& option = std::nullopt);
};

} // namespace humecodec
