#pragma once

#include "humecodec/csrc/ffmpeg.h"
#include "humecodec/csrc/stream_reader/post_process.h"
#include "humecodec/csrc/stream_reader/typedefs.h"
#include "humecodec/csrc/dlpack.h"
#include <map>

namespace humecodec {

class StreamProcessor {
 public:
  using KeyType = int;

 private:
  AVRational stream_time_base;
  AVCodecContextPtr codec_ctx{nullptr};
  AVFramePtr frame{alloc_avframe()};

  KeyType current_key = 0;
  std::map<KeyType, std::unique_ptr<IPostDecodeProcess>> post_processes;

  int64_t discard_before_pts = 0;
  // Set on a seek-flush (not seek-to-start). Used to correct the pts of the
  // first emitted Vorbis frame after a seek: a block-size transition frame
  // emits (bs_prev+bs_cur)/4 samples but its granule only advances bs_cur/2,
  // so its leading (nb_samples - duration) samples belong *before* its pts.
  bool warmup_pending_ = false;

 public:
  explicit StreamProcessor(const AVRational& time_base);
  ~StreamProcessor() = default;
  // Non-copyable
  StreamProcessor(const StreamProcessor&) = delete;
  StreamProcessor& operator=(const StreamProcessor&) = delete;
  // Movable
  StreamProcessor(StreamProcessor&&) = default;
  StreamProcessor& operator=(StreamProcessor&&) = default;

  //////////////////////////////////////////////////////////////////////////////
  // Configurations
  //////////////////////////////////////////////////////////////////////////////
  KeyType add_stream(
      int frames_per_chunk,
      int num_chunks,
      AVRational frame_rate,
      const std::string& filter_description,
      const DLDevice& device);

  void remove_stream(KeyType key);

  void set_discard_timestamp(int64_t timestamp);

  void set_decoder(
      const AVCodecParameters* codecpar,
      const std::optional<std::string>& decoder_name,
      const std::optional<OptionDict>& decoder_option,
      const DLDevice& device);

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
  [[nodiscard]] std::string get_filter_description(KeyType key) const;
  [[nodiscard]] FilterGraphOutputInfo get_filter_output_info(KeyType key) const;

  bool is_buffer_ready() const;
  [[nodiscard]] bool is_decoder_set() const;

  //////////////////////////////////////////////////////////////////////////////
  // The streaming process
  //////////////////////////////////////////////////////////////////////////////
  int process_packet(AVPacket* packet);

  void flush(bool seeking_to_start = false);

 private:
  int send_frame(AVFrame* pFrame);

  //////////////////////////////////////////////////////////////////////////////
  // Retrieval
  //////////////////////////////////////////////////////////////////////////////
 public:
  std::optional<Chunk> pop_chunk(KeyType key);
};

} // namespace humecodec
