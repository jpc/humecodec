#include "humecodec/csrc/hw_context.h"
#include "humecodec/csrc/stream_writer/encode_process.h"
#include "humecodec/csrc/tensor_view.h"
#include <cmath>
#include <sstream>

namespace humecodec {

////////////////////////////////////////////////////////////////////////////////
// EncodeProcess Logic Implementation
////////////////////////////////////////////////////////////////////////////////

EncodeProcess::EncodeProcess(
    TensorConverter&& converter,
    AVFramePtr&& frame,
    FilterGraph&& filter_graph,
    Encoder&& encoder,
    AVCodecContextPtr&& codec_ctx) noexcept
    : converter(std::move(converter)),
      src_frame(std::move(frame)),
      filter(std::move(filter_graph)),
      encoder(std::move(encoder)),
      codec_ctx(std::move(codec_ctx)) {}

void EncodeProcess::process(
    const ManagedBuffer& buf,
    const std::optional<double>& pts) {
  if (pts) {
    const double& pts_val = pts.value();
    HCODEC_CHECK(
        std::isfinite(pts_val) && pts_val >= 0.0,
        "The value of PTS must be positive and finite. Found: ",
        pts_val);
    AVRational tb = codec_ctx->time_base;
    auto val = static_cast<int64_t>(std::round(pts_val * tb.den / tb.num));
    if (src_frame->pts > val) {
      HCODEC_WARN_ONCE(
          "The provided PTS value is smaller than the next expected value.");
    }
    src_frame->pts = val;
  }
  for (const auto& frame : converter.convert(buf)) {
    process_frame(frame);
    frame->pts += frame->nb_samples;
  }
}

void EncodeProcess::process_frame(AVFrame* src) {
  int ret = filter.add_frame(src);
  while (ret >= 0) {
    ret = filter.get_frame(dst_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      if (ret == AVERROR_EOF) {
        encoder.encode(nullptr);
      }
      break;
    }
    if (ret >= 0) {
      encoder.encode(dst_frame);
    }
    av_frame_unref(dst_frame);
  }
}

void EncodeProcess::flush() {
  process_frame(nullptr);
}

////////////////////////////////////////////////////////////////////////////////
// EncodeProcess Initialization helper functions
////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////
// Local string join helper (replaces c10::Join)
////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::string join_str(const std::string& sep, const std::vector<T>& items) {
  std::ostringstream oss;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) oss << sep;
    oss << items[i];
  }
  return oss.str();
}

// Specialization for std::array<int, 2>
std::string join_ints(const std::string& sep, int a, int b) {
  std::ostringstream oss;
  oss << a << sep << b;
  return oss.str();
}

enum AVSampleFormat get_src_sample_fmt(const std::string& src) {
  auto fmt = av_get_sample_fmt(src.c_str());
  if (fmt != AV_SAMPLE_FMT_NONE && !av_sample_fmt_is_planar(fmt)) {
    return fmt;
  }
  HCODEC_CHECK(
      false,
      "Unsupported sample fotmat (",
      src,
      ") was provided. Valid values are ",
      []() -> std::string {
        std::vector<std::string> ret;
        for (const auto& fmt :
             {AV_SAMPLE_FMT_U8,
              AV_SAMPLE_FMT_S16,
              AV_SAMPLE_FMT_S32,
              AV_SAMPLE_FMT_S64,
              AV_SAMPLE_FMT_FLT,
              AV_SAMPLE_FMT_DBL}) {
          ret.emplace_back(av_get_sample_fmt_name(fmt));
        }
        return join_str(", ", ret);
      }(),
      ".");
}

const std::set<AVPixelFormat> SUPPORTED_PIX_FMTS{
    AV_PIX_FMT_GRAY8,
    AV_PIX_FMT_RGB0,
    AV_PIX_FMT_BGR0,
    AV_PIX_FMT_RGB24,
    AV_PIX_FMT_BGR24,
    AV_PIX_FMT_YUV444P};

enum AVPixelFormat get_src_pix_fmt(const std::string& src) {
  AVPixelFormat fmt = av_get_pix_fmt(src.c_str());
  HCODEC_CHECK(
      SUPPORTED_PIX_FMTS.count(fmt),
      "Unsupported pixel format (",
      src,
      ") was provided. Valid values are ",
      []() -> std::string {
        std::vector<std::string> ret;
        for (const auto& fmt : SUPPORTED_PIX_FMTS) {
          ret.emplace_back(av_get_pix_fmt_name(fmt));
        }
        return join_str(", ", ret);
      }(),
      ".");
  return fmt;
}

////////////////////////////////////////////////////////////////////////////////
// Codec & Codec context
////////////////////////////////////////////////////////////////////////////////
const AVCodec* get_codec(
    AVCodecID default_codec,
    const std::optional<std::string>& encoder) {
  if (encoder) {
    const AVCodec* c = avcodec_find_encoder_by_name(encoder.value().c_str());
    HCODEC_CHECK(c, "Unexpected codec: ", encoder.value());
    return c;
  }
  const AVCodec* c = avcodec_find_encoder(default_codec);
  HCODEC_CHECK(
      c, "Encoder not found for codec: ", avcodec_get_name(default_codec));
  return c;
}

AVCodecContextPtr get_codec_ctx(const AVCodec* codec, int flags) {
  AVCodecContext* ctx = avcodec_alloc_context3(codec);
  HCODEC_CHECK(ctx, "Failed to allocate CodecContext.");

  if (flags & AVFMT_GLOBALHEADER) {
    ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
  return AVCodecContextPtr(ctx);
}

void open_codec(
    AVCodecContext* codec_ctx,
    const std::optional<OptionDict>& option) {
  AVDictionary* opt = get_option_dict(option);

  if (std::strcmp(codec_ctx->codec->name, "vorbis") == 0) {
    if (!av_dict_get(opt, "strict", nullptr, 0)) {
      HCODEC_WARN_ONCE(
          "\"vorbis\" encoder is selected. Enabling '-strict experimental'. ",
          "If this is not desired, please provide \"strict\" encoder option ",
          "with desired value.");
      av_dict_set(&opt, "strict", "experimental", 0);
    }
  }
  if (std::strcmp(codec_ctx->codec->name, "opus") == 0) {
    if (!av_dict_get(opt, "strict", nullptr, 0)) {
      HCODEC_WARN_ONCE(
          "\"opus\" encoder is selected. Enabling '-strict experimental'. ",
          "If this is not desired, please provide \"strict\" encoder option ",
          "with desired value.");
      av_dict_set(&opt, "strict", "experimental", 0);
    }
  }

  // Default to single thread execution.
  if (!av_dict_get(opt, "threads", nullptr, 0)) {
    av_dict_set(&opt, "threads", "1", 0);
  }

  int ret = avcodec_open2(codec_ctx, codec_ctx->codec, &opt);
  clean_up_dict(opt);
  HCODEC_CHECK(ret >= 0, "Failed to open codec: (", av_err2string(ret), ")");
}

////////////////////////////////////////////////////////////////////////////////
// Audio codec
////////////////////////////////////////////////////////////////////////////////

bool supported_sample_fmt(
    const AVSampleFormat fmt,
    const AVSampleFormat* sample_fmts) {
  if (!sample_fmts) {
    return true;
  }
  while (*sample_fmts != AV_SAMPLE_FMT_NONE) {
    if (fmt == *sample_fmts) {
      return true;
    }
    ++sample_fmts;
  }
  return false;
}

std::string get_supported_formats(const AVSampleFormat* sample_fmts) {
  std::vector<std::string> ret;
  while (*sample_fmts != AV_SAMPLE_FMT_NONE) {
    ret.emplace_back(av_get_sample_fmt_name(*sample_fmts));
    ++sample_fmts;
  }
  return join_str(", ", ret);
}

AVSampleFormat get_enc_fmt(
    AVSampleFormat src_fmt,
    const std::optional<std::string>& encoder_format,
    const AVCodec* codec) {
  if (encoder_format) {
    auto& enc_fmt_val = encoder_format.value();
    auto fmt = av_get_sample_fmt(enc_fmt_val.c_str());
    HCODEC_CHECK(
        fmt != AV_SAMPLE_FMT_NONE, "Unknown sample format: ", enc_fmt_val);
    HCODEC_CHECK(
        supported_sample_fmt(fmt, codec->sample_fmts),
        codec->name,
        " does not support ",
        encoder_format.value(),
        " format. Supported values are; ",
        get_supported_formats(codec->sample_fmts));
    return fmt;
  }
  if (codec->sample_fmts) {
    return codec->sample_fmts[0];
  }
  return src_fmt;
};

bool supported_sample_rate(const int sample_rate, const AVCodec* codec) {
  if (!codec->supported_samplerates) {
    return true;
  }
  const int* it = codec->supported_samplerates;
  while (*it) {
    if (sample_rate == *it) {
      return true;
    }
    ++it;
  }
  return false;
}

std::string get_supported_samplerates(const int* supported_samplerates) {
  std::vector<int> ret;
  if (supported_samplerates) {
    while (*supported_samplerates) {
      ret.push_back(*supported_samplerates);
      ++supported_samplerates;
    }
  }
  return join_str(", ", ret);
}

int get_enc_sr(
    int src_sample_rate,
    const std::optional<int>& encoder_sample_rate,
    const AVCodec* codec) {
  // G.722 only supports 16000 Hz, but it does not list the sample rate in
  // supported_samplerates so we hard code it here.
  if (codec->id == AV_CODEC_ID_ADPCM_G722) {
    if (encoder_sample_rate) {
      auto val = encoder_sample_rate.value();
      HCODEC_CHECK(
          val == 16'000,
          codec->name,
          " does not support sample rate ",
          val,
          ". Supported values are; 16000.");
    }
    return 16'000;
  }
  if (encoder_sample_rate) {
    const int& encoder_sr = encoder_sample_rate.value();
    HCODEC_CHECK(
        encoder_sr > 0,
        "Encoder sample rate must be positive. Found: ",
        encoder_sr);
    HCODEC_CHECK(
        supported_sample_rate(encoder_sr, codec),
        codec->name,
        " does not support sample rate ",
        encoder_sr,
        ". Supported values are; ",
        get_supported_samplerates(codec->supported_samplerates));
    return encoder_sr;
  }
  if (codec->supported_samplerates &&
      !supported_sample_rate(src_sample_rate, codec)) {
    return codec->supported_samplerates[0];
  }
  return src_sample_rate;
}

#if LIBAVUTIL_VERSION_MAJOR >= 59
std::string get_supported_channels(const AVChannelLayout* ch_layouts) {
  std::vector<std::string> names;
  for (const AVChannelLayout* it = ch_layouts; it->nb_channels != 0; ++it) {
    std::stringstream ss;
    ss << it->nb_channels;
    char buf[64];
    if (av_channel_layout_describe(it, buf, sizeof(buf)) > 0) {
      ss << " (" << buf << ")";
    }
    names.emplace_back(ss.str());
  }
  return join_str(", ", names);
}
#else
std::string get_supported_channels(const uint64_t* channel_layouts) {
  std::vector<std::string> names;
  while (*channel_layouts) {
    std::stringstream ss;
    ss << av_get_channel_layout_nb_channels(*channel_layouts);
    ss << " (" << av_get_channel_name(*channel_layouts) << ")";
    names.emplace_back(ss.str());
    ++channel_layouts;
  }
  return join_str(", ", names);
}
#endif

#if LIBAVUTIL_VERSION_MAJOR >= 59
uint64_t get_channel_layout(
    const uint64_t src_ch_layout,
    const std::optional<int> enc_num_channels,
    const AVCodec* codec) {
  if (enc_num_channels) {
    const int& val = enc_num_channels.value();
    HCODEC_CHECK(
        val > 0, "The number of channels must be greater than 0. Found: ", val);
    if (!codec->ch_layouts || codec->ch_layouts[0].nb_channels == 0) {
      AVChannelLayout layout = {};
      av_channel_layout_default(&layout, val);
      uint64_t mask = layout.u.mask;
      av_channel_layout_uninit(&layout);
      return mask;
    }
    for (const AVChannelLayout* it = codec->ch_layouts; it->nb_channels != 0; ++it) {
      if (it->nb_channels == val) {
        return it->u.mask;
      }
    }
    HCODEC_CHECK(
        false,
        "Codec ",
        codec->name,
        " does not support a channel layout consists of ",
        val,
        " channels. Supported values are: ",
        get_supported_channels(codec->ch_layouts));
  }
  if (!codec->ch_layouts || codec->ch_layouts[0].nb_channels == 0) {
    return src_ch_layout;
  }
  for (const AVChannelLayout* it = codec->ch_layouts; it->nb_channels != 0; ++it) {
    if (it->u.mask == src_ch_layout) {
      return src_ch_layout;
    }
  }
  return codec->ch_layouts[0].u.mask;
}
#else
uint64_t get_channel_layout(
    const uint64_t src_ch_layout,
    const std::optional<int> enc_num_channels,
    const AVCodec* codec) {
  if (enc_num_channels) {
    const int& val = enc_num_channels.value();
    HCODEC_CHECK(
        val > 0, "The number of channels must be greater than 0. Found: ", val);
    if (!codec->channel_layouts) {
      return static_cast<uint64_t>(av_get_default_channel_layout(val));
    }
    for (const uint64_t* it = codec->channel_layouts; *it; ++it) {
      if (av_get_channel_layout_nb_channels(*it) == val) {
        return *it;
      }
    }
    HCODEC_CHECK(
        false,
        "Codec ",
        codec->name,
        " does not support a channel layout consists of ",
        val,
        " channels. Supported values are: ",
        get_supported_channels(codec->channel_layouts));
  }
  if (!codec->channel_layouts) {
    return src_ch_layout;
  }
  for (const uint64_t* it = codec->channel_layouts; *it; ++it) {
    if (*it == src_ch_layout) {
      return src_ch_layout;
    }
  }
  return codec->channel_layouts[0];
}
#endif

void configure_audio_codec_ctx(
    AVCodecContext* codec_ctx,
    AVSampleFormat format,
    int sample_rate,
    uint64_t channel_layout,
    const std::optional<CodecConfig>& codec_config) {
  codec_ctx->sample_fmt = format;
  codec_ctx->sample_rate = sample_rate;
  codec_ctx->time_base = av_inv_q(av_d2q(sample_rate, 1 << 24));
#if LIBAVUTIL_VERSION_MAJOR >= 59
  av_channel_layout_from_mask(&codec_ctx->ch_layout, channel_layout);
#else
  codec_ctx->channels = av_get_channel_layout_nb_channels(channel_layout);
  codec_ctx->channel_layout = channel_layout;
#endif

  if (codec_config) {
    auto& cfg = codec_config.value();
    if (cfg.bit_rate > 0) {
      codec_ctx->bit_rate = cfg.bit_rate;
    }
    if (cfg.compression_level != -1) {
      codec_ctx->compression_level = cfg.compression_level;
    }
    if (cfg.qscale) {
      codec_ctx->flags |= AV_CODEC_FLAG_QSCALE;
      codec_ctx->global_quality = FF_QP2LAMBDA * cfg.qscale.value();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Video codec
////////////////////////////////////////////////////////////////////////////////

bool supported_pix_fmt(const AVPixelFormat fmt, const AVPixelFormat* pix_fmts) {
  if (!pix_fmts) {
    return true;
  }
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    if (fmt == *pix_fmts) {
      return true;
    }
    ++pix_fmts;
  }
  return false;
}

std::string get_supported_formats(const AVPixelFormat* pix_fmts) {
  std::vector<std::string> ret;
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    ret.emplace_back(av_get_pix_fmt_name(*pix_fmts));
    ++pix_fmts;
  }
  return join_str(", ", ret);
}

AVPixelFormat get_enc_fmt(
    AVPixelFormat src_fmt,
    const std::optional<std::string>& encoder_format,
    const AVCodec* codec) {
  if (encoder_format) {
    const auto& val = encoder_format.value();
    auto fmt = av_get_pix_fmt(val.c_str());
    HCODEC_CHECK(
        supported_pix_fmt(fmt, codec->pix_fmts),
        codec->name,
        " does not support ",
        val,
        " format. Supported values are; ",
        get_supported_formats(codec->pix_fmts));
    return fmt;
  }
  if (codec->pix_fmts) {
    return codec->pix_fmts[0];
  }
  return src_fmt;
}

bool supported_frame_rate(AVRational rate, const AVRational* rates) {
  if (!rates) {
    return true;
  }
  for (; !(rates->num == 0 && rates->den == 0); ++rates) {
    if (av_cmp_q(rate, *rates) == 0) {
      return true;
    }
  }
  return false;
}

AVRational get_enc_rate(
    AVRational src_rate,
    const std::optional<double>& encoder_sample_rate,
    const AVCodec* codec) {
  if (encoder_sample_rate) {
    const double& enc_rate = encoder_sample_rate.value();
    HCODEC_CHECK(
        std::isfinite(enc_rate) && enc_rate > 0,
        "Encoder sample rate must be positive and fininte. Found: ",
        enc_rate);
    AVRational rate = av_d2q(enc_rate, 1 << 24);
    HCODEC_CHECK(
        supported_frame_rate(rate, codec->supported_framerates),
        codec->name,
        " does not support frame rate: ",
        enc_rate,
        ". Supported values are; ",
        [&]() {
          std::vector<std::string> ret;
          for (auto r = codec->supported_framerates;
               !(r->num == 0 && r->den == 0);
               ++r) {
            ret.push_back(join_ints("/", r->num, r->den));
          }
          return join_str(", ", ret);
        }());
    return rate;
  }
  if (codec->supported_framerates &&
      !supported_frame_rate(src_rate, codec->supported_framerates)) {
    return codec->supported_framerates[0];
  }
  return src_rate;
}

void configure_video_codec_ctx(
    AVCodecContextPtr& ctx,
    AVPixelFormat format,
    AVRational frame_rate,
    int width,
    int height,
    const std::optional<CodecConfig>& codec_config) {
  ctx->pix_fmt = format;
  ctx->width = width;
  ctx->height = height;
  ctx->time_base = av_inv_q(frame_rate);

  if (codec_config) {
    auto& cfg = codec_config.value();
    if (cfg.bit_rate > 0) {
      ctx->bit_rate = cfg.bit_rate;
    }
    if (cfg.compression_level != -1) {
      ctx->compression_level = cfg.compression_level;
    }
    if (cfg.gop_size != -1) {
      ctx->gop_size = cfg.gop_size;
    }
    if (cfg.max_b_frames != -1) {
      ctx->max_b_frames = cfg.max_b_frames;
    }
    if (cfg.qscale) {
      ctx->flags |= AV_CODEC_FLAG_QSCALE;
      ctx->global_quality = FF_QP2LAMBDA * cfg.qscale.value();
    }
  }
}

#ifdef USE_CUDA
void configure_hw_accel(AVCodecContext* ctx, const std::string& hw_accel) {
  // Parse "cuda" or "cuda:N" to get device_id
  int device_id = 0;
  HCODEC_CHECK(
      hw_accel.substr(0, 4) == "cuda",
      "Only CUDA is supported for hardware acceleration. Found: ",
      hw_accel);
  if (hw_accel.size() > 5 && hw_accel[4] == ':') {
    device_id = std::stoi(hw_accel.substr(5));
  }

  ctx->hw_device_ctx = av_buffer_ref(get_cuda_context(device_id));
  HCODEC_INTERNAL_ASSERT(
      ctx->hw_device_ctx, "Failed to reference HW device context.");

  ctx->sw_pix_fmt = ctx->pix_fmt;
  ctx->pix_fmt = AV_PIX_FMT_CUDA;

  ctx->hw_frames_ctx = av_hwframe_ctx_alloc(ctx->hw_device_ctx);
  HCODEC_CHECK(ctx->hw_frames_ctx, "Failed to create CUDA frame context.");

  auto frames_ctx = (AVHWFramesContext*)(ctx->hw_frames_ctx->data);
  frames_ctx->format = ctx->pix_fmt;
  frames_ctx->sw_format = ctx->sw_pix_fmt;
  frames_ctx->width = ctx->width;
  frames_ctx->height = ctx->height;
  frames_ctx->initial_pool_size = 5;

  int ret = av_hwframe_ctx_init(ctx->hw_frames_ctx);
  HCODEC_CHECK(
      ret >= 0,
      "Failed to initialize CUDA frame context: ",
      av_err2string(ret));
}
#endif // USE_CUDA

////////////////////////////////////////////////////////////////////////////////
// AVStream
////////////////////////////////////////////////////////////////////////////////

AVStream* get_stream(AVFormatContext* format_ctx, AVCodecContext* codec_ctx) {
  AVStream* stream = avformat_new_stream(format_ctx, nullptr);
  HCODEC_CHECK(stream, "Failed to allocate stream.");

  stream->time_base = codec_ctx->time_base;
  int ret = avcodec_parameters_from_context(stream->codecpar, codec_ctx);
  HCODEC_CHECK(
      ret >= 0, "Failed to copy the stream parameter: ", av_err2string(ret));
  return stream;
}

////////////////////////////////////////////////////////////////////////////////
// FilterGraph
////////////////////////////////////////////////////////////////////////////////

FilterGraph get_audio_filter_graph(
    AVSampleFormat src_fmt,
    int src_sample_rate,
    uint64_t src_ch_layout,
    const std::optional<std::string>& filter_desc,
    AVSampleFormat enc_fmt,
    int enc_sample_rate,
    uint64_t enc_ch_layout,
    int nb_samples) {
  const auto desc = [&]() -> const std::string {
    std::vector<std::string> parts;
    if (filter_desc) {
      parts.push_back(filter_desc.value());
    }
    if (filter_desc || src_fmt != enc_fmt ||
        src_sample_rate != enc_sample_rate || src_ch_layout != enc_ch_layout) {
      std::stringstream ss;
      ss << "aformat=sample_fmts=" << av_get_sample_fmt_name(enc_fmt)
         << ":sample_rates=" << enc_sample_rate << ":channel_layouts=0x"
         << std::hex << enc_ch_layout;
      parts.push_back(ss.str());
    }
    if (nb_samples > 0) {
      std::stringstream ss;
      ss << "asetnsamples=n=" << nb_samples << ":p=0";
      parts.push_back(ss.str());
    }
    if (parts.size()) {
      return join_str(",", parts);
    }
    return "anull";
  }();

  FilterGraph f;
  f.add_audio_src(
      src_fmt, {1, src_sample_rate}, src_sample_rate, src_ch_layout);
  f.add_audio_sink();
  f.add_process(desc);
  f.create_filter();
  return f;
}

FilterGraph get_video_filter_graph(
    AVPixelFormat src_fmt,
    AVRational src_rate,
    int src_width,
    int src_height,
    const std::optional<std::string>& filter_desc,
    AVPixelFormat enc_fmt,
    AVRational enc_rate,
    int enc_width,
    int enc_height,
    bool is_cuda) {
  const auto desc = [&]() -> const std::string {
    if (is_cuda) {
      return filter_desc.value_or("null");
    }
    std::vector<std::string> parts;
    if (filter_desc) {
      parts.push_back(filter_desc.value());
    }
    if (filter_desc || (src_width != enc_width || src_height != enc_height)) {
      std::stringstream ss;
      ss << "scale=" << enc_width << ":" << enc_height;
      parts.emplace_back(ss.str());
    }
    if (filter_desc || src_fmt != enc_fmt) {
      std::stringstream ss;
      ss << "format=" << av_get_pix_fmt_name(enc_fmt);
      parts.emplace_back(ss.str());
    }
    if (filter_desc ||
        (src_rate.num != enc_rate.num || src_rate.den != enc_rate.den)) {
      std::stringstream ss;
      ss << "fps=" << enc_rate.num << "/" << enc_rate.den;
      parts.emplace_back(ss.str());
    }
    if (parts.size()) {
      return join_str(",", parts);
    }
    return "null";
  }();

  FilterGraph f;
  f.add_video_src(
      is_cuda ? AV_PIX_FMT_CUDA : src_fmt,
      av_inv_q(src_rate),
      src_rate,
      src_width,
      src_height,
      {1, 1});
  f.add_video_sink();
  f.add_process(desc);
  f.create_filter();
  return f;
}

////////////////////////////////////////////////////////////////////////////////
// Source frame
////////////////////////////////////////////////////////////////////////////////

AVFramePtr get_audio_frame(
    AVSampleFormat format,
    int sample_rate,
    int num_channels,
    uint64_t channel_layout,
    int nb_samples) {
  AVFramePtr frame{alloc_avframe()};
  frame->format = format;
#if LIBAVUTIL_VERSION_MAJOR >= 59
  av_channel_layout_from_mask(&frame->ch_layout, channel_layout);
#else
  frame->channel_layout = channel_layout;
  frame->channels = num_channels;
#endif
  frame->sample_rate = sample_rate;
  frame->nb_samples = nb_samples;
  int ret = av_frame_get_buffer(frame, 0);
  HCODEC_CHECK(
      ret >= 0, "Error allocating the source audio frame:", av_err2string(ret));

  frame->pts = 0;
  return frame;
}

AVFramePtr get_video_frame(AVPixelFormat src_fmt, int width, int height) {
  AVFramePtr frame{alloc_avframe()};
  frame->format = src_fmt;
  frame->width = width;
  frame->height = height;
  int ret = av_frame_get_buffer(frame, 0);
  HCODEC_CHECK(
      ret >= 0, "Error allocating a video buffer :", av_err2string(ret));

  frame->nb_samples = 1;
  frame->pts = 0;
  return frame;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Finally, the extern-facing API
////////////////////////////////////////////////////////////////////////////////

EncodeProcess get_audio_encode_process(
    AVFormatContext* format_ctx,
    int src_sample_rate,
    int src_num_channels,
    const std::string& format,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_option,
    const std::optional<std::string>& encoder_format,
    const std::optional<int>& encoder_sample_rate,
    const std::optional<int>& encoder_num_channels,
    const std::optional<CodecConfig>& codec_config,
    const std::optional<std::string>& filter_desc,
    bool disable_converter) {
  HCODEC_CHECK(
      src_sample_rate > 0,
      "Sample rate must be positive. Found: ",
      src_sample_rate);
  HCODEC_CHECK(
      src_num_channels > 0,
      "The number of channels must be positive. Found: ",
      src_num_channels);
  const AVSampleFormat src_fmt = (disable_converter)
      ? av_get_sample_fmt(format.c_str())
      : get_src_sample_fmt(format);
#if LIBAVUTIL_VERSION_MAJOR >= 59
  AVChannelLayout tmp_layout = {};
  av_channel_layout_default(&tmp_layout, src_num_channels);
  const auto src_ch_layout = tmp_layout.u.mask;
  av_channel_layout_uninit(&tmp_layout);
#else
  const auto src_ch_layout =
      static_cast<uint64_t>(av_get_default_channel_layout(src_num_channels));
#endif

  HCODEC_CHECK(
      format_ctx->oformat->audio_codec != AV_CODEC_ID_NONE,
      format_ctx->oformat->name,
      " does not support audio.");
  const AVCodec* codec = get_codec(format_ctx->oformat->audio_codec, encoder);

  const AVSampleFormat enc_fmt = get_enc_fmt(src_fmt, encoder_format, codec);
  const int enc_sr = get_enc_sr(src_sample_rate, encoder_sample_rate, codec);
  const uint64_t enc_ch_layout = [&]() -> uint64_t {
    if (std::strcmp(codec->name, "vorbis") == 0) {
#if LIBAVUTIL_VERSION_MAJOR >= 59
      AVChannelLayout stereo_layout = {};
      av_channel_layout_default(&stereo_layout, 2);
      uint64_t result = stereo_layout.u.mask;
      av_channel_layout_uninit(&stereo_layout);
      return result;
#else
      return static_cast<uint64_t>(av_get_default_channel_layout(2));
#endif
    }
    return get_channel_layout(src_ch_layout, encoder_num_channels, codec);
  }();

  AVCodecContextPtr codec_ctx =
      get_codec_ctx(codec, format_ctx->oformat->flags);
  configure_audio_codec_ctx(
      codec_ctx, enc_fmt, enc_sr, enc_ch_layout, codec_config);
  open_codec(codec_ctx, encoder_option);

  FilterGraph filter_graph = get_audio_filter_graph(
      src_fmt,
      src_sample_rate,
      src_ch_layout,
      filter_desc,
      enc_fmt,
      enc_sr,
      enc_ch_layout,
      codec_ctx->frame_size);

  AVFramePtr src_frame = get_audio_frame(
      src_fmt,
      src_sample_rate,
      src_num_channels,
      src_ch_layout,
      codec_ctx->frame_size > 0 ? codec_ctx->frame_size : 256);

  TensorConverter converter{
      (disable_converter) ? AVMEDIA_TYPE_UNKNOWN : AVMEDIA_TYPE_AUDIO,
      src_frame,
      src_frame->nb_samples};

  Encoder enc{format_ctx, codec_ctx, get_stream(format_ctx, codec_ctx)};

  return EncodeProcess{
      std::move(converter),
      std::move(src_frame),
      std::move(filter_graph),
      std::move(enc),
      std::move(codec_ctx)};
}

namespace {

bool ends_with(std::string_view str, std::string_view suffix) {
  return str.size() >= suffix.size() &&
      0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

} // namespace

EncodeProcess get_video_encode_process(
    AVFormatContext* format_ctx,
    double frame_rate,
    int src_width,
    int src_height,
    const std::string& format,
    const std::optional<std::string>& encoder,
    const std::optional<OptionDict>& encoder_option,
    const std::optional<std::string>& encoder_format,
    const std::optional<double>& encoder_frame_rate,
    const std::optional<int>& encoder_width,
    const std::optional<int>& encoder_height,
    const std::optional<std::string>& hw_accel,
    const std::optional<CodecConfig>& codec_config,
    const std::optional<std::string>& filter_desc,
    bool disable_converter) {
  HCODEC_CHECK(
      std::isfinite(frame_rate) && frame_rate > 0,
      "Frame rate must be positive and finite. Found: ",
      frame_rate);
  HCODEC_CHECK(src_width > 0, "width must be positive. Found: ", src_width);
  HCODEC_CHECK(src_height > 0, "height must be positive. Found: ", src_height);
  const AVPixelFormat src_fmt = (disable_converter)
      ? av_get_pix_fmt(format.c_str())
      : get_src_pix_fmt(format);
  const AVRational src_rate = av_d2q(frame_rate, 1 << 24);

  HCODEC_CHECK(
      format_ctx->oformat->video_codec != AV_CODEC_ID_NONE,
      format_ctx->oformat->name,
      " does not support video.");
  const AVCodec* codec = get_codec(format_ctx->oformat->video_codec, encoder);

  const AVPixelFormat enc_fmt = get_enc_fmt(src_fmt, encoder_format, codec);
  const AVRational enc_rate = get_enc_rate(src_rate, encoder_frame_rate, codec);
  const int enc_width = [&]() -> int {
    if (!encoder_width) {
      return src_width;
    }
    const int& val = encoder_width.value();
    HCODEC_CHECK(val > 0, "Encoder width must be positive. Found: ", val);
    return val;
  }();
  const int enc_height = [&]() -> int {
    if (!encoder_height) {
      return src_height;
    }
    const int& val = encoder_height.value();
    HCODEC_CHECK(val > 0, "Encoder height must be positive. Found: ", val);
    return val;
  }();

  AVCodecContextPtr codec_ctx =
      get_codec_ctx(codec, format_ctx->oformat->flags);
  configure_video_codec_ctx(
      codec_ctx, enc_fmt, enc_rate, enc_width, enc_height, codec_config);
  if (hw_accel) {
#ifdef USE_CUDA
    configure_hw_accel(codec_ctx, hw_accel.value());
#else
    HCODEC_CHECK(
        false,
        "humecodec is not compiled with CUDA support. ",
        "Hardware acceleration is not available.");
#endif
  }
  open_codec(codec_ctx, encoder_option);

  FilterGraph filter_graph = get_video_filter_graph(
      src_fmt,
      src_rate,
      src_width,
      src_height,
      filter_desc,
      enc_fmt,
      enc_rate,
      enc_width,
      enc_height,
      hw_accel.has_value());

  AVFramePtr src_frame = [&]() {
    if (codec_ctx->hw_frames_ctx) {
      AVFramePtr frame{alloc_avframe()};
      int ret = av_hwframe_get_buffer(codec_ctx->hw_frames_ctx, frame, 0);
      HCODEC_CHECK(ret >= 0, "Failed to fetch CUDA frame: ", av_err2string(ret));
      frame->nb_samples = 1;
      frame->pts = 0;
      return frame;
    }
    return get_video_frame(src_fmt, src_width, src_height);
  }();

  TensorConverter converter{
      (disable_converter) ? AVMEDIA_TYPE_UNKNOWN : AVMEDIA_TYPE_VIDEO,
      src_frame};

  Encoder enc{format_ctx, codec_ctx, get_stream(format_ctx, codec_ctx)};

  return EncodeProcess{
      std::move(converter),
      std::move(src_frame),
      std::move(filter_graph),
      std::move(enc),
      std::move(codec_ctx)};
}

} // namespace humecodec
