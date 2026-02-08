#include "torchffmpeg/csrc/stream_reader/buffer/chunked_buffer.h"
#include "torchffmpeg/csrc/stream_reader/buffer/unchunked_buffer.h"
#include "torchffmpeg/csrc/stream_reader/conversion.h"
#include "torchffmpeg/csrc/stream_reader/post_process.h"
#include "torchffmpeg/csrc/tensor_view.h"

namespace torchffmpeg {
namespace detail {
namespace {

///////////////////////////////////////////////////////////////////////////////
// FilterGraphWrapper (FilterGraph + reset feature)
///////////////////////////////////////////////////////////////////////////////
using FilterGraphFactory = std::function<FilterGraph(const std::string&)>;

FilterGraphFactory get_audio_factory(
    AVRational time_base,
    AVCodecContext* codec_ctx) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(codec_ctx->codec_type == AVMEDIA_TYPE_AUDIO);
#if LIBAVUTIL_VERSION_MAJOR >= 59
  // FFmpeg 7+ uses AVChannelLayout - convert to uint64_t for filter
  uint64_t channel_layout_val = 0;
  if (codec_ctx->ch_layout.order == AV_CHANNEL_ORDER_NATIVE) {
    channel_layout_val = codec_ctx->ch_layout.u.mask;
  } else {
    // For non-native layouts, use default layout for channel count
    AVChannelLayout default_layout = {};
    av_channel_layout_default(&default_layout, codec_ctx->ch_layout.nb_channels);
    channel_layout_val = default_layout.u.mask;
    av_channel_layout_uninit(&default_layout);
  }
  return [fmt = codec_ctx->sample_fmt,
          time_base,
          rate = codec_ctx->sample_rate,
          channel_layout = channel_layout_val](
             const std::string& filter_desc) -> FilterGraph {
    FilterGraph f;
    f.add_audio_src(fmt, time_base, rate, channel_layout);
    f.add_audio_sink();
    f.add_process(filter_desc);
    f.create_filter();
    return f;
  };
#else
  return [fmt = codec_ctx->sample_fmt,
          time_base,
          rate = codec_ctx->sample_rate,
          channel_layout = codec_ctx->channel_layout](
             const std::string& filter_desc) -> FilterGraph {
    FilterGraph f;
    f.add_audio_src(fmt, time_base, rate, channel_layout);
    f.add_audio_sink();
    f.add_process(filter_desc);
    f.create_filter();
    return f;
  };
#endif
}

FilterGraphFactory get_video_factory(
    AVRational time_base,
    AVRational frame_rate,
    AVCodecContext* codec_ctx) {
  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO);
  return [fmt = codec_ctx->pix_fmt,
          time_base,
          frame_rate,
          w = codec_ctx->width,
          h = codec_ctx->height,
          ratio = codec_ctx->sample_aspect_ratio,
          hw_frames_ctx = codec_ctx->hw_frames_ctx](
             const std::string& filter_desc) -> FilterGraph {
    FilterGraph f;
    f.add_video_src(fmt, time_base, frame_rate, w, h, ratio);
    f.add_video_sink();
    f.add_process(filter_desc);
    if (hw_frames_ctx) {
      f.create_filter(av_buffer_ref(hw_frames_ctx));
    } else {
      f.create_filter();
    }
    return f;
  };
}

struct FilterGraphWrapper {
  const std::string desc;

 private:
  FilterGraphFactory factory;

 public:
  FilterGraph filter;

  // Constructor for audio input
  FilterGraphWrapper(
      AVRational input_time_base,
      AVCodecContext* codec_ctx,
      const std::string& desc)
      : desc(desc),
        factory(get_audio_factory(input_time_base, codec_ctx)),
        filter(factory(desc)) {}

  // Constructor for video input
  FilterGraphWrapper(
      AVRational input_time_base,
      AVRational frame_rate,
      AVCodecContext* codec_ctx,
      const std::string& desc)
      : desc(desc),
        factory(get_video_factory(input_time_base, frame_rate, codec_ctx)),
        filter(factory(desc)) {}

  void reset() {
    filter = factory(desc);
  }
};

///////////////////////////////////////////////////////////////////////////////
// ProcessImpl
///////////////////////////////////////////////////////////////////////////////
template <typename Converter, typename Buffer>
struct ProcessImpl : public IPostDecodeProcess {
 private:
  AVFramePtr frame{alloc_avframe()};
  FilterGraphWrapper filter_wrapper;

 public:
  Converter converter;
  Buffer buffer;

  ProcessImpl(
      FilterGraphWrapper&& filter_wrapper,
      Converter&& converter,
      Buffer&& buffer)
      : filter_wrapper(std::move(filter_wrapper)),
        converter(std::move(converter)),
        buffer(std::move(buffer)) {}

  bool is_buffer_ready() const override {
    return buffer.is_ready();
  }

  const std::string& get_filter_desc() const override {
    return filter_wrapper.desc;
  }

  FilterGraphOutputInfo get_filter_output_info() const override {
    return filter_wrapper.filter.get_output_info();
  }

  void flush() override {
    filter_wrapper.reset();
    buffer.flush();
  }

  int process_frame(AVFrame* in_frame) override {
    int ret = filter_wrapper.filter.add_frame(in_frame);
    while (ret >= 0) {
      ret = filter_wrapper.filter.get_frame(frame);
      //  AVERROR(EAGAIN) means that new input data is required to return new
      //  output.
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return 0;
      }
      if (ret >= 0) {
        buffer.push_frame(converter.convert(frame), frame->pts);
      }
      av_frame_unref(frame);
    }
    return ret;
  }

  std::optional<Chunk> pop_chunk() override {
    return buffer.pop_chunk();
  }
};

///////////////////////////////////////////////////////////////////////////////
// Audio
///////////////////////////////////////////////////////////////////////////////
std::unique_ptr<IPostDecodeProcess> get_unchunked_audio_process(
    FilterGraphWrapper&& filter) {
  auto i = filter.filter.get_output_info();

  TFMPEG_INTERNAL_ASSERT(
      i.type == AVMEDIA_TYPE_AUDIO,
      "Unsupported media type found: ",
      av_get_media_type_string(i.type));

  using B = UnchunkedBuffer;

  switch (auto fmt = (AVSampleFormat)i.format; fmt) {
    case AV_SAMPLE_FMT_U8: {
      using C = AudioConverter<kDLUInt, 8, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_S16: {
      using C = AudioConverter<kDLInt, 16, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_S32: {
      using C = AudioConverter<kDLInt, 32, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_S64: {
      using C = AudioConverter<kDLInt, 64, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_FLT: {
      using C = AudioConverter<kDLFloat, 32, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_DBL: {
      using C = AudioConverter<kDLFloat, 64, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_U8P: {
      using C = AudioConverter<kDLUInt, 8, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_S16P: {
      using C = AudioConverter<kDLInt, 16, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_S32P: {
      using C = AudioConverter<kDLInt, 32, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_S64P: {
      using C = AudioConverter<kDLInt, 64, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_FLTP: {
      using C = AudioConverter<kDLFloat, 32, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    case AV_SAMPLE_FMT_DBLP: {
      using C = AudioConverter<kDLFloat, 64, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, B{i.time_base});
    }
    default:
      TFMPEG_INTERNAL_ASSERT(
          false, "Unexpected audio type:", av_get_sample_fmt_name(fmt));
  }
}

std::unique_ptr<IPostDecodeProcess> get_chunked_audio_process(
    FilterGraphWrapper&& filter,
    int frames_per_chunk,
    int num_chunks) {
  auto i = filter.filter.get_output_info();

  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(
      i.type == AVMEDIA_TYPE_AUDIO,
      "Unsupported media type found: ",
      av_get_media_type_string(i.type));

  using B = ChunkedBuffer;
  B buffer{i.time_base, frames_per_chunk, num_chunks};

  switch (auto fmt = (AVSampleFormat)i.format; fmt) {
    case AV_SAMPLE_FMT_U8: {
      using C = AudioConverter<kDLUInt, 8, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_S16: {
      using C = AudioConverter<kDLInt, 16, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_S32: {
      using C = AudioConverter<kDLInt, 32, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_S64: {
      using C = AudioConverter<kDLInt, 64, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_FLT: {
      using C = AudioConverter<kDLFloat, 32, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_DBL: {
      using C = AudioConverter<kDLFloat, 64, false>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_U8P: {
      using C = AudioConverter<kDLUInt, 8, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_S16P: {
      using C = AudioConverter<kDLInt, 16, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_S32P: {
      using C = AudioConverter<kDLInt, 32, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_S64P: {
      using C = AudioConverter<kDLInt, 64, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_FLTP: {
      using C = AudioConverter<kDLFloat, 32, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    case AV_SAMPLE_FMT_DBLP: {
      using C = AudioConverter<kDLFloat, 64, true>;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{i.num_channels}, std::move(buffer));
    }
    default:
      TFMPEG_INTERNAL_ASSERT(
          false, "Unexpected audio type:", av_get_sample_fmt_name(fmt));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Video
///////////////////////////////////////////////////////////////////////////////
std::unique_ptr<IPostDecodeProcess> get_unchunked_video_process(
    FilterGraphWrapper&& filter) {
  auto i = filter.filter.get_output_info();

  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(
      i.type == AVMEDIA_TYPE_VIDEO,
      "Unsupported media type found: ",
      av_get_media_type_string(i.type));

  auto h = i.height;
  auto w = i.width;
  auto tb = i.time_base;

  using B = UnchunkedBuffer;
  switch (auto fmt = (AVPixelFormat)i.format; fmt) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24: {
      using C = InterlacedImageConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w, 3}, B{tb});
    }
    case AV_PIX_FMT_ARGB:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_ABGR:
    case AV_PIX_FMT_BGRA: {
      using C = InterlacedImageConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w, 4}, B{tb});
    }
    case AV_PIX_FMT_GRAY8: {
      using C = InterlacedImageConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w, 1}, B{tb});
    }
    case AV_PIX_FMT_RGB48LE: {
      using C = Interlaced16BitImageConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w, 3}, B{tb});
    }
    case AV_PIX_FMT_YUV444P: {
      using C = PlanarImageConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w, 3}, B{tb});
    }
    case AV_PIX_FMT_YUV420P: {
      using C = YUV420PConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w}, B{tb});
    }
    case AV_PIX_FMT_YUV420P10LE: {
      using C = YUV420P10LEConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w}, B{tb});
    }
    case AV_PIX_FMT_NV12: {
      using C = NV12Converter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w}, B{tb});
    }
    default: {
      TFMPEG_INTERNAL_ASSERT(
          false, "Unexpected video format found: ", av_get_pix_fmt_name(fmt));
    }
  }
}

std::unique_ptr<IPostDecodeProcess> get_unchunked_cuda_video_process(
    FilterGraphWrapper&& filter,
    const DLDevice& dev) {
#ifndef USE_CUDA
  TFMPEG_INTERNAL_ASSERT(
      false,
      "USE_CUDA is not defined, but CUDA decoding process was requested.");
#else
  auto i = filter.filter.get_output_info();

  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(
      i.type == AVMEDIA_TYPE_VIDEO,
      "Unsupported media type found: ",
      av_get_media_type_string(i.type));

  using B = UnchunkedBuffer;
  switch (auto fmt = (AVPixelFormat)i.format; fmt) {
    case AV_PIX_FMT_NV12: {
      using C = NV12CudaConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{dev}, B{i.time_base});
    }
    case AV_PIX_FMT_P010: {
      using C = P010CudaConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{dev}, B{i.time_base});
    }
    case AV_PIX_FMT_YUV444P: {
      using C = YUV444PCudaConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{dev}, B{i.time_base});
    }
    case AV_PIX_FMT_P016: {
      TFMPEG_CHECK(
          false,
          "Unsupported video format found in CUDA HW: ",
          av_get_pix_fmt_name(fmt));
    }
    default: {
      TFMPEG_CHECK(
          false,
          "Unexpected video format found in CUDA HW: ",
          av_get_pix_fmt_name(fmt));
    }
  }
#endif
}

std::unique_ptr<IPostDecodeProcess> get_chunked_video_process(
    FilterGraphWrapper&& filter,
    int frames_per_chunk,
    int num_chunks) {
  auto i = filter.filter.get_output_info();

  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(
      i.type == AVMEDIA_TYPE_VIDEO,
      "Unsupported media type found: ",
      av_get_media_type_string(i.type));

  auto h = i.height;
  auto w = i.width;
  auto tb = i.time_base;

  using B = ChunkedBuffer;
  switch (auto fmt = (AVPixelFormat)i.format; fmt) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24: {
      using C = InterlacedImageConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w, 3}, B{tb, frames_per_chunk, num_chunks});
    }
    case AV_PIX_FMT_ARGB:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_ABGR:
    case AV_PIX_FMT_BGRA: {
      using C = InterlacedImageConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w, 4}, B{tb, frames_per_chunk, num_chunks});
    }
    case AV_PIX_FMT_GRAY8: {
      using C = InterlacedImageConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w, 1}, B{tb, frames_per_chunk, num_chunks});
    }
    case AV_PIX_FMT_RGB48LE: {
      using C = Interlaced16BitImageConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w, 3}, B{tb, frames_per_chunk, num_chunks});
    }
    case AV_PIX_FMT_YUV444P: {
      using C = PlanarImageConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w, 3}, B{tb, frames_per_chunk, num_chunks});
    }
    case AV_PIX_FMT_YUV420P: {
      using C = YUV420PConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w}, B{tb, frames_per_chunk, num_chunks});
    }
    case AV_PIX_FMT_YUV420P10LE: {
      using C = YUV420P10LEConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w}, B{tb, frames_per_chunk, num_chunks});
    }
    case AV_PIX_FMT_NV12: {
      using C = NV12Converter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter), C{h, w}, B{tb, frames_per_chunk, num_chunks});
    }
    default: {
      TFMPEG_INTERNAL_ASSERT(
          false, "Unexpected video format found: ", av_get_pix_fmt_name(fmt));
    }
  }
}

std::unique_ptr<IPostDecodeProcess> get_chunked_cuda_video_process(
    FilterGraphWrapper&& filter,
    int frames_per_chunk,
    int num_chunks,
    const DLDevice& dev) {
#ifndef USE_CUDA
  TFMPEG_INTERNAL_ASSERT(
      false,
      "USE_CUDA is not defined, but CUDA decoding process was requested.");
#else
  auto i = filter.filter.get_output_info();

  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(
      i.type == AVMEDIA_TYPE_VIDEO,
      "Unsupported media type found: ",
      av_get_media_type_string(i.type));

  using B = ChunkedBuffer;
  switch (auto fmt = (AVPixelFormat)i.format; fmt) {
    case AV_PIX_FMT_NV12: {
      using C = NV12CudaConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter),
          C{dev},
          B{i.time_base, frames_per_chunk, num_chunks});
    }
    case AV_PIX_FMT_P010: {
      using C = P010CudaConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter),
          C{dev},
          B{i.time_base, frames_per_chunk, num_chunks});
    }
    case AV_PIX_FMT_YUV444P: {
      using C = YUV444PCudaConverter;
      return std::make_unique<ProcessImpl<C, B>>(
          std::move(filter),
          C{dev},
          B{i.time_base, frames_per_chunk, num_chunks});
    }
    case AV_PIX_FMT_P016: {
      TFMPEG_CHECK(
          false,
          "Unsupported video format found in CUDA HW: ",
          av_get_pix_fmt_name(fmt));
    }
    default: {
      TFMPEG_CHECK(
          false,
          "Unexpected video format found in CUDA HW: ",
          av_get_pix_fmt_name(fmt));
    }
  }
#endif
}
} // namespace
} // namespace detail

std::unique_ptr<IPostDecodeProcess> get_audio_process(
    AVRational input_time_base,
    AVCodecContext* codec_ctx,
    const std::string& desc,
    int frames_per_chunk,
    int num_chunks) {
  TFMPEG_CHECK(
      frames_per_chunk > 0 || frames_per_chunk == -1,
      "`frames_per_chunk` must be positive or -1. Found: ",
      frames_per_chunk);

  TFMPEG_CHECK(
      num_chunks > 0 || num_chunks == -1,
      "`num_chunks` must be positive or -1. Found: ",
      num_chunks);

  detail::FilterGraphWrapper filter{input_time_base, codec_ctx, desc};

  if (frames_per_chunk == -1) {
    return detail::get_unchunked_audio_process(std::move(filter));
  }
  return detail::get_chunked_audio_process(
      std::move(filter), frames_per_chunk, num_chunks);
}

std::unique_ptr<IPostDecodeProcess> get_video_process(
    AVRational input_time_base,
    AVRational frame_rate,
    AVCodecContext* codec_ctx,
    const std::string& desc,
    int frames_per_chunk,
    int num_chunks,
    const DLDevice& dev) {
  TFMPEG_CHECK(
      frames_per_chunk > 0 || frames_per_chunk == -1,
      "`frames_per_chunk` must be positive or -1. Found: ",
      frames_per_chunk);

  TFMPEG_CHECK(
      num_chunks > 0 || num_chunks == -1,
      "`num_chunks` must be positive or -1. Found: ",
      num_chunks);

  TFMPEG_INTERNAL_ASSERT_DEBUG_ONLY(
      dev.device_type == kDLCUDA || dev.device_type == kDLCPU,
      "Unexpected device type: ", dev.device_type);

  detail::FilterGraphWrapper filter{
      input_time_base, frame_rate, codec_ctx, desc};

  if (frames_per_chunk == -1) {
    if (dev.device_type == kDLCUDA) {
      return detail::get_unchunked_cuda_video_process(
          std::move(filter), dev);
    }
    return detail::get_unchunked_video_process(std::move(filter));
  }
  if (dev.device_type == kDLCUDA) {
    return detail::get_chunked_cuda_video_process(
        std::move(filter), frames_per_chunk, num_chunks, dev);
  }
  return detail::get_chunked_video_process(
      std::move(filter), frames_per_chunk, num_chunks);
}
} // namespace torchffmpeg
