#include "humecodec/csrc/ffmpeg.h"
#include "humecodec/csrc/stream_reader/stream_reader.h"
#include "humecodec/csrc/tensor_view.h"
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace humecodec {

using KeyType = StreamProcessor::KeyType;

//////////////////////////////////////////////////////////////////////////////
// Initialization / resource allocations
//////////////////////////////////////////////////////////////////////////////
namespace {
AVFormatContext* get_input_format_context(
    const std::string& src,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& option,
    AVIOContext* io_ctx) {
  AVFormatContext* p = avformat_alloc_context();
  HCODEC_CHECK(p, "Failed to allocate AVFormatContext.");
  if (io_ctx) {
    p->pb = io_ctx;
  }

  auto* pInputFormat = [&format]() -> AVFORMAT_CONST AVInputFormat* {
    if (format.has_value()) {
      std::string format_str = format.value();
      AVFORMAT_CONST AVInputFormat* pInput =
          av_find_input_format(format_str.c_str());
      HCODEC_CHECK(pInput, "Unsupported device/format: \"", format_str, "\"");
      return pInput;
    }
    return nullptr;
  }();

  AVDictionary* opt = get_option_dict(option);
  int ret = avformat_open_input(&p, src.c_str(), pInputFormat, &opt);
  clean_up_dict(opt);

  HCODEC_CHECK(
      ret >= 0,
      "Failed to open the input \"",
      src,
      "\" (",
      av_err2string(ret),
      ").");
  return p;
}
} // namespace

StreamingMediaDecoder::StreamingMediaDecoder(AVFormatContext* p)
    : format_ctx(p) {
  int ret = avformat_find_stream_info(format_ctx, nullptr);
  HCODEC_CHECK(
      ret >= 0, "Failed to find stream information: ", av_err2string(ret));

  processors =
      std::vector<std::unique_ptr<StreamProcessor>>(format_ctx->nb_streams);
  for (int i = 0; i < static_cast<int>(format_ctx->nb_streams); ++i) {
    switch (format_ctx->streams[i]->codecpar->codec_type) {
      case AVMEDIA_TYPE_AUDIO:
      case AVMEDIA_TYPE_VIDEO:
        break;
      default:
        format_ctx->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

StreamingMediaDecoder::StreamingMediaDecoder(
    AVIOContext* io_ctx,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& option)
    : StreamingMediaDecoder(get_input_format_context(
          "Custom Input Context",
          format,
          option,
          io_ctx)) {}

StreamingMediaDecoder::StreamingMediaDecoder(
    const std::string& src,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& option)
    : StreamingMediaDecoder(
          get_input_format_context(src, format, option, nullptr)) {}

//////////////////////////////////////////////////////////////////////////////
// Helper methods
//////////////////////////////////////////////////////////////////////////////
void validate_open_stream(AVFormatContext* format_ctx) {
  HCODEC_CHECK(format_ctx, "Stream is not open.");
}

void validate_src_stream_index(AVFormatContext* format_ctx, int i) {
  validate_open_stream(format_ctx);
  HCODEC_CHECK(
      i >= 0 && i < static_cast<int>(format_ctx->nb_streams),
      "Source stream index out of range");
}

void validate_src_stream_type(
    AVFormatContext* format_ctx,
    int i,
    AVMediaType type) {
  validate_src_stream_index(format_ctx, i);
  HCODEC_CHECK(
      format_ctx->streams[i]->codecpar->codec_type == type,
      "Stream ",
      i,
      " is not ",
      av_get_media_type_string(type),
      " stream.");
}

////////////////////////////////////////////////////////////////////////////////
// Query methods
////////////////////////////////////////////////////////////////////////////////
int64_t StreamingMediaDecoder::num_src_streams() const {
  return format_ctx->nb_streams;
}

namespace {
OptionDict parse_metadata(const AVDictionary* metadata) {
  AVDictionaryEntry* tag = nullptr;
  OptionDict ret;
  while ((tag = av_dict_get(metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {
    ret.emplace(std::string(tag->key), std::string(tag->value));
  }
  return ret;
}
} // namespace

OptionDict StreamingMediaDecoder::get_metadata() const {
  return parse_metadata(format_ctx->metadata);
}

SrcStreamInfo StreamingMediaDecoder::get_src_stream_info(int i) const {
  validate_src_stream_index(format_ctx, i);

  AVStream* stream = format_ctx->streams[i];
  AVCodecParameters* codecpar = stream->codecpar;

  SrcStreamInfo ret;
  ret.media_type = codecpar->codec_type;
  ret.bit_rate = codecpar->bit_rate;
  ret.num_frames = stream->nb_frames;
  ret.bits_per_sample = codecpar->bits_per_raw_sample;
  ret.metadata = parse_metadata(stream->metadata);
  ret.time_base_num = stream->time_base.num;
  ret.time_base_den = stream->time_base.den;
  const AVCodecDescriptor* desc = avcodec_descriptor_get(codecpar->codec_id);
  if (desc) {
    ret.codec_name = desc->name;
    ret.codec_long_name = desc->long_name;
  }

  switch (codecpar->codec_type) {
    case AVMEDIA_TYPE_AUDIO: {
      AVSampleFormat smp_fmt = static_cast<AVSampleFormat>(codecpar->format);
      if (smp_fmt != AV_SAMPLE_FMT_NONE) {
        ret.fmt_name = av_get_sample_fmt_name(smp_fmt);
      }
      ret.sample_rate = static_cast<double>(codecpar->sample_rate);
#if LIBAVUTIL_VERSION_MAJOR >= 59
      ret.num_channels = codecpar->ch_layout.nb_channels;
#else
      ret.num_channels = codecpar->channels;
#endif
      break;
    }
    case AVMEDIA_TYPE_VIDEO: {
      AVPixelFormat pix_fmt = static_cast<AVPixelFormat>(codecpar->format);
      if (pix_fmt != AV_PIX_FMT_NONE) {
        ret.fmt_name = av_get_pix_fmt_name(pix_fmt);
      }
      ret.width = codecpar->width;
      ret.height = codecpar->height;
      ret.frame_rate = av_q2d(stream->r_frame_rate);
      break;
    }
    default:;
  }
  return ret;
}

namespace {
AVCodecParameters* get_codecpar() {
  AVCodecParameters* ptr = avcodec_parameters_alloc();
  HCODEC_CHECK(ptr, "Failed to allocate resource.");
  return ptr;
}
} // namespace

StreamParams StreamingMediaDecoder::get_src_stream_params(int i) {
  validate_src_stream_index(format_ctx, i);
  AVStream* stream = format_ctx->streams[i];

  AVCodecParametersPtr codec_params(get_codecpar());
  int ret = avcodec_parameters_copy(codec_params, stream->codecpar);
  HCODEC_CHECK(
      ret >= 0,
      "Failed to copy the stream's codec parameters. (",
      av_err2string(ret),
      ")");
  return {std::move(codec_params), stream->time_base, i};
}

int64_t StreamingMediaDecoder::num_out_streams() const {
  return static_cast<int64_t>(stream_indices.size());
}

OutputStreamInfo StreamingMediaDecoder::get_out_stream_info(int i) const {
  HCODEC_CHECK(
      i >= 0 && static_cast<size_t>(i) < stream_indices.size(),
      "Output stream index out of range");
  int i_src = stream_indices[i].first;
  KeyType key = stream_indices[i].second;
  FilterGraphOutputInfo info = processors[i_src]->get_filter_output_info(key);

  OutputStreamInfo ret;
  ret.source_index = i_src;
  ret.filter_description = processors[i_src]->get_filter_description(key);
  ret.media_type = info.type;
  ret.format = info.format;
  switch (info.type) {
    case AVMEDIA_TYPE_AUDIO:
      ret.sample_rate = info.sample_rate;
      ret.num_channels = info.num_channels;
      break;
    case AVMEDIA_TYPE_VIDEO:
      ret.width = info.width;
      ret.height = info.height;
      ret.frame_rate = info.frame_rate;
      break;
    default:;
  }
  return ret;
}

int64_t StreamingMediaDecoder::find_best_audio_stream() const {
  return av_find_best_stream(
      format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
}

int64_t StreamingMediaDecoder::find_best_video_stream() const {
  return av_find_best_stream(
      format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
}

bool StreamingMediaDecoder::is_buffer_ready() const {
  if (processors.empty()) {
    return packet_buffer->has_packets();
  } else {
    for (const auto& it : processors) {
      if (it && !it->is_buffer_ready()) {
        return false;
      }
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Configure methods
////////////////////////////////////////////////////////////////////////////////
void StreamingMediaDecoder::seek(double timestamp_s, int64_t mode) {
  HCODEC_CHECK(timestamp_s >= 0, "timestamp must be non-negative.");
  HCODEC_CHECK(
      format_ctx->nb_streams > 0,
      "At least one stream must exist in this context");

  int64_t timestamp_av_tb = static_cast<int64_t>(timestamp_s * AV_TIME_BASE);

  int flag = AVSEEK_FLAG_BACKWARD;
  switch (mode) {
    case 0:
      seek_timestamp = 0;
      break;
    case 1:
      flag |= AVSEEK_FLAG_ANY;
      seek_timestamp = 0;
      break;
    case 2:
      seek_timestamp = timestamp_av_tb;
      break;
    default:
      HCODEC_CHECK(false, "Invalid mode value: ", mode);
  }

  int ret = av_seek_frame(format_ctx, -1, timestamp_av_tb, flag);

  if (ret < 0) {
    seek_timestamp = 0;
    HCODEC_CHECK(false, "Failed to seek. (" + av_err2string(ret) + ".)");
  }
  for (const auto& it : processors) {
    if (it) {
      it->flush();
      it->set_discard_timestamp(seek_timestamp);
    }
  }
}

void StreamingMediaDecoder::seek_to_byte_offset(int64_t offset) {
  HCODEC_CHECK(offset >= 0, "offset must be non-negative.");

  int ret = av_seek_frame(format_ctx, -1, offset, AVSEEK_FLAG_BYTE);
  HCODEC_CHECK(
      ret >= 0,
      "Failed to seek to byte offset ",
      offset,
      ". (",
      av_err2string(ret),
      ")");

  seek_timestamp = 0;
  for (const auto& it : processors) {
    if (it) {
      it->flush();
      it->set_discard_timestamp(0);
    }
  }
}

void StreamingMediaDecoder::add_audio_stream(
    int64_t i,
    int64_t frames_per_chunk,
    int64_t num_chunks,
    const std::optional<std::string>& filter_desc,
    const std::optional<std::string>& decoder,
    const std::optional<OptionDict>& decoder_option) {
  add_stream(
      static_cast<int>(i),
      AVMEDIA_TYPE_AUDIO,
      static_cast<int>(frames_per_chunk),
      static_cast<int>(num_chunks),
      filter_desc.value_or("anull"),
      decoder,
      decoder_option,
      device::cpu());
}

void StreamingMediaDecoder::add_video_stream(
    int64_t i,
    int64_t frames_per_chunk,
    int64_t num_chunks,
    const std::optional<std::string>& filter_desc,
    const std::optional<std::string>& decoder,
    const std::optional<OptionDict>& decoder_option,
    const std::optional<std::string>& hw_accel) {
  const DLDevice dev = [&]() -> DLDevice {
    if (!hw_accel) {
      return device::cpu();
    }
#ifdef USE_CUDA
    // Parse "cuda:N" or "cuda" string
    const std::string& accel = hw_accel.value();
    HCODEC_CHECK(
        accel.find("cuda") == 0,
        "Only CUDA is supported for HW acceleration. Found: ", accel);
    int device_id = 0;
    if (accel.size() > 5 && accel[4] == ':') {
      device_id = std::stoi(accel.substr(5));
    }
    return device::cuda(device_id);
#else
    HCODEC_CHECK(
        false,
        "humecodec is not compiled with CUDA support. Hardware acceleration is not available.");
#endif
  }();

  add_stream(
      static_cast<int>(i),
      AVMEDIA_TYPE_VIDEO,
      static_cast<int>(frames_per_chunk),
      static_cast<int>(num_chunks),
      filter_desc.value_or("null"),
      decoder,
      decoder_option,
      dev);
}

void StreamingMediaDecoder::add_packet_stream(int i) {
  validate_src_stream_index(format_ctx, i);
  if (!packet_buffer) {
    packet_buffer = std::make_unique<PacketBuffer>();
  }
  packet_stream_indices.emplace(i);
}

void StreamingMediaDecoder::add_stream(
    int i,
    AVMediaType media_type,
    int frames_per_chunk,
    int num_chunks,
    const std::string& filter_desc,
    const std::optional<std::string>& decoder,
    const std::optional<OptionDict>& decoder_option,
    const DLDevice& dev) {
  validate_src_stream_type(format_ctx, i, media_type);

  AVStream* stream = format_ctx->streams[i];
  HCODEC_CHECK(
      stream->codecpar->format != -1,
      "Failed to detect the source stream format.");

  if (!processors[i]) {
    processors[i] = std::make_unique<StreamProcessor>(stream->time_base);
    processors[i]->set_discard_timestamp(seek_timestamp);
  }
  if (!processors[i]->is_decoder_set()) {
    processors[i]->set_decoder(
        stream->codecpar, decoder, decoder_option, dev);
  } else {
    HCODEC_CHECK(
        !decoder && (!decoder_option || decoder_option.value().size() == 0),
        "Decoder options were provided, but the decoder has already been initialized.");
  }

  stream->discard = AVDISCARD_DEFAULT;

  auto frame_rate = [&]() -> AVRational {
    switch (media_type) {
      case AVMEDIA_TYPE_AUDIO:
        return AVRational{0, 1};
      case AVMEDIA_TYPE_VIDEO:
        return av_guess_frame_rate(format_ctx, stream, nullptr);
      default:
        HCODEC_INTERNAL_ASSERT(
            false,
            "Unexpected media type is given: ",
            av_get_media_type_string(media_type));
    }
  }();
  int key = processors[i]->add_stream(
      frames_per_chunk, num_chunks, frame_rate, filter_desc, dev);
  stream_indices.push_back(std::make_pair<>(i, key));
}

void StreamingMediaDecoder::remove_stream(int64_t i) {
  HCODEC_CHECK(
      i >= 0 && static_cast<size_t>(i) < stream_indices.size(),
      "Output stream index out of range");
  auto it = stream_indices.begin() + i;
  int iP = it->first;
  processors[iP]->remove_stream(it->second);
  stream_indices.erase(it);

  bool still_used = false;
  for (auto& p : stream_indices) {
    still_used |= (iP == p.first);
    if (still_used) {
      break;
    }
  }
  if (!still_used) {
    processors[iP].reset(nullptr);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Stream methods
////////////////////////////////////////////////////////////////////////////////
int StreamingMediaDecoder::process_packet() {
  int ret = av_read_frame(format_ctx, packet);
  if (ret == AVERROR_EOF) {
    ret = drain();
    return (ret < 0) ? ret : 1;
  }
  if (ret < 0) {
    return ret;
  }
  AutoPacketUnref auto_unref{packet};

  int stream_index = packet->stream_index;

  if (packet_stream_indices.count(stream_index)) {
    packet_buffer->push_packet(packet);
  }

  auto& processor = processors[stream_index];
  if (!processor) {
    return 0;
  }

  ret = processor->process_packet(packet);

  return (ret < 0) ? ret : 0;
}

int StreamingMediaDecoder::process_packet_block(
    double timeout,
    double backoff) {
  auto dead_line = [&]() {
    if (timeout < 0) {
      return std::chrono::time_point<std::chrono::steady_clock>::max();
    }
    auto timeout_ = static_cast<int64_t>(1000 * timeout);
    return std::chrono::steady_clock::now() +
        std::chrono::microseconds{timeout_};
  }();

  std::chrono::microseconds sleep{static_cast<int64_t>(1000 * backoff)};

  while (true) {
    int ret = process_packet();
    if (ret != AVERROR(EAGAIN)) {
      return ret;
    }
    if (dead_line < std::chrono::steady_clock::now()) {
      return ret;
    }
    std::this_thread::sleep_for(sleep);
  }
}

void StreamingMediaDecoder::process_all_packets() {
  int64_t ret = 0;
  do {
    ret = process_packet();
  } while (!ret);
}

int StreamingMediaDecoder::process_packet(
    const std::optional<double>& timeout,
    const double backoff) {
  int code = [&]() -> int {
    if (timeout.has_value()) {
      return process_packet_block(timeout.value(), backoff);
    }
    return process_packet();
  }();
  HCODEC_CHECK(
      code >= 0, "Failed to process a packet. (" + av_err2string(code) + "). ");
  return code;
}

int StreamingMediaDecoder::fill_buffer(
    const std::optional<double>& timeout,
    const double backoff) {
  while (!is_buffer_ready()) {
    int code = process_packet(timeout, backoff);
    if (code != 0) {
      return code;
    }
  }
  return 0;
}

int StreamingMediaDecoder::drain() {
  int ret = 0, tmp = 0;
  for (auto& p : processors) {
    if (p) {
      tmp = p->process_packet(nullptr);
      if (tmp < 0) {
        ret = tmp;
      }
    }
  }
  return ret;
}

std::vector<std::optional<Chunk>> StreamingMediaDecoder::pop_chunks() {
  std::vector<std::optional<Chunk>> ret;
  ret.reserve(static_cast<size_t>(num_out_streams()));
  for (auto& i : stream_indices) {
    ret.emplace_back(processors[i.first]->pop_chunk(i.second));
  }
  return ret;
}

std::vector<AVPacketPtr> StreamingMediaDecoder::pop_packets() {
  return packet_buffer->pop_packets();
}

std::vector<PacketIndexEntry> StreamingMediaDecoder::build_packet_index(
    int stream_index, int64_t resolution_bytes) {
  validate_src_stream_index(format_ctx, stream_index);

  AVStream* stream = format_ctx->streams[stream_index];
  double time_base = av_q2d(stream->time_base);

  // Seek to the beginning of the file
  int ret = av_seek_frame(format_ctx, stream_index, 0, AVSEEK_FLAG_BACKWARD);
  HCODEC_CHECK(
      ret >= 0,
      "Failed to seek to beginning of stream. (",
      av_err2string(ret),
      ")");

  AVPacketPtr pkt{alloc_avpacket()};
  std::vector<PacketIndexEntry> index;
  int64_t last_emitted_pos = -resolution_bytes; // ensures first entry is always emitted

  while (av_read_frame(format_ctx, pkt) >= 0) {
    if (pkt->stream_index == stream_index) {
      if (pkt->pos - last_emitted_pos >= resolution_bytes) {
        PacketIndexEntry entry;
        entry.pts = pkt->pts;
        entry.pts_seconds =
            (pkt->pts != AV_NOPTS_VALUE) ? pkt->pts * time_base : -1.0;
        entry.pos = pkt->pos;
        entry.size = pkt->size;
        entry.is_key = (pkt->flags & AV_PKT_FLAG_KEY) != 0;
        index.push_back(entry);
        last_emitted_pos = pkt->pos;
      }
    }
    av_packet_unref(pkt);
  }

  return index;
}

//////////////////////////////////////////////////////////////////////////////
// StreamingMediaDecoderCustomIO
//////////////////////////////////////////////////////////////////////////////

namespace detail {
namespace {
AVIOContext* get_io_context(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence)) {
  unsigned char* buffer = static_cast<unsigned char*>(av_malloc(buffer_size));
  HCODEC_CHECK(buffer, "Failed to allocate buffer.");
  AVIOContext* io_ctx = avio_alloc_context(
      buffer, buffer_size, 0, opaque, read_packet, nullptr, seek);
  if (!io_ctx) {
    av_freep(&buffer);
    HCODEC_CHECK(false, "Failed to allocate AVIOContext.");
  }
  return io_ctx;
}
} // namespace

CustomInput::CustomInput(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence))
    : io_ctx(get_io_context(opaque, buffer_size, read_packet, seek)) {}
} // namespace detail

StreamingMediaDecoderCustomIO::StreamingMediaDecoderCustomIO(
    void* opaque,
    const std::optional<std::string>& format,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence),
    const std::optional<OptionDict>& option)
    : CustomInput(opaque, buffer_size, read_packet, seek),
      StreamingMediaDecoder(io_ctx, format, option) {}

} // namespace humecodec
