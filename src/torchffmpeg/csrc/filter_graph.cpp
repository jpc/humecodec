#include "torchffmpeg/csrc/filter_graph.h"
#include <stdexcept>

namespace torchffmpeg {

namespace {
AVFilterGraph* get_filter_graph() {
  AVFilterGraph* ptr = avfilter_graph_alloc();
  TORCH_CHECK(ptr, "Failed to allocate resource.");
  ptr->nb_threads = 1;
  return ptr;
}
} // namespace

FilterGraph::FilterGraph() : graph(get_filter_graph()) {}

////////////////////////////////////////////////////////////////////////////////
// Configuration methods
////////////////////////////////////////////////////////////////////////////////
namespace {

#if LIBAVUTIL_VERSION_MAJOR >= 58
// FFmpeg 6+ uses AVChannelLayout
std::string get_audio_src_args(
    AVSampleFormat format,
    AVRational time_base,
    int sample_rate,
    uint64_t channel_layout) {
  char args[512];
  // Use channel_layout for filter args - ffmpeg will convert
  std::snprintf(
      args,
      sizeof(args),
      "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=0x%" PRIx64,
      time_base.num,
      time_base.den,
      sample_rate,
      av_get_sample_fmt_name(format),
      channel_layout);
  return std::string(args);
}
#else
std::string get_audio_src_args(
    AVSampleFormat format,
    AVRational time_base,
    int sample_rate,
    uint64_t channel_layout) {
  char args[512];
  std::snprintf(
      args,
      sizeof(args),
      "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=0x%" PRIx64,
      time_base.num,
      time_base.den,
      sample_rate,
      av_get_sample_fmt_name(format),
      channel_layout);
  return std::string(args);
}
#endif

std::string get_video_src_args(
    AVPixelFormat format,
    AVRational time_base,
    AVRational frame_rate,
    int width,
    int height,
    AVRational sample_aspect_ratio) {
  char args[512];
  std::snprintf(
      args,
      sizeof(args),
      "video_size=%dx%d:pix_fmt=%s:time_base=%d/%d:frame_rate=%d/%d:pixel_aspect=%d/%d",
      width,
      height,
      av_get_pix_fmt_name(format),
      time_base.num,
      time_base.den,
      frame_rate.num,
      frame_rate.den,
      sample_aspect_ratio.num,
      sample_aspect_ratio.den);
  return std::string(args);
}

} // namespace

void FilterGraph::add_audio_src(
    AVSampleFormat format,
    AVRational time_base,
    int sample_rate,
    uint64_t channel_layout) {
  add_src(
      avfilter_get_by_name("abuffer"),
      get_audio_src_args(format, time_base, sample_rate, channel_layout));
}

void FilterGraph::add_video_src(
    AVPixelFormat format,
    AVRational time_base,
    AVRational frame_rate,
    int width,
    int height,
    AVRational sample_aspect_ratio) {
  add_src(
      avfilter_get_by_name("buffer"),
      get_video_src_args(
          format, time_base, frame_rate, width, height, sample_aspect_ratio));
}

void FilterGraph::add_src(const AVFilter* buffersrc, const std::string& args) {
  int ret = avfilter_graph_create_filter(
      &buffersrc_ctx, buffersrc, "in", args.c_str(), nullptr, graph);
  TORCH_CHECK(
      ret >= 0,
      "Failed to create input filter: \"" + args + "\" (" + av_err2string(ret) +
          ")");
}

void FilterGraph::add_audio_sink() {
  add_sink(avfilter_get_by_name("abuffersink"));
}

void FilterGraph::add_video_sink() {
  add_sink(avfilter_get_by_name("buffersink"));
}

void FilterGraph::add_sink(const AVFilter* buffersink) {
  TORCH_CHECK(!buffersink_ctx, "Sink buffer is already allocated.");
  int ret = avfilter_graph_create_filter(
      &buffersink_ctx, buffersink, "out", nullptr, nullptr, graph);
  TORCH_CHECK(ret >= 0, "Failed to create output filter.");
}

namespace {

class InOuts {
  AVFilterInOut* p = nullptr;
  InOuts(const InOuts&) = delete;
  InOuts& operator=(const InOuts&) = delete;

 public:
  InOuts(const char* name, AVFilterContext* pCtx) {
    p = avfilter_inout_alloc();
    TORCH_CHECK(p, "Failed to allocate AVFilterInOut.");
    p->name = av_strdup(name);
    p->filter_ctx = pCtx;
    p->pad_idx = 0;
    p->next = nullptr;
  }
  ~InOuts() {
    avfilter_inout_free(&p);
  }
  operator AVFilterInOut**() {
    return &p;
  }
};

} // namespace

void FilterGraph::add_process(const std::string& filter_description) {
  InOuts in{"in", buffersrc_ctx}, out{"out", buffersink_ctx};

  int ret = avfilter_graph_parse_ptr(
      graph, filter_description.c_str(), out, in, nullptr);

  TORCH_CHECK(
      ret >= 0,
      "Failed to create the filter from \"" + filter_description + "\" (" +
          av_err2string(ret) + ".)");
}

void FilterGraph::create_filter(AVBufferRef* hw_frames_ctx) {
#if LIBAVFILTER_VERSION_MAJOR < 10
  if (hw_frames_ctx) {
    buffersrc_ctx->outputs[0]->hw_frames_ctx = hw_frames_ctx;
  }
#else
  // FFmpeg 7+ - hw_frames_ctx handled differently
  // For now, we don't set hw_frames_ctx directly on the filter link
  (void)hw_frames_ctx;
#endif
  int ret = avfilter_graph_config(graph, nullptr);
  TORCH_CHECK(ret >= 0, "Failed to configure the graph: " + av_err2string(ret));
}

//////////////////////////////////////////////////////////////////////////////
// Query methods
//////////////////////////////////////////////////////////////////////////////
FilterGraphOutputInfo FilterGraph::get_output_info() const {
  TORCH_INTERNAL_ASSERT(buffersink_ctx, "FilterGraph is not initialized.");
  AVFilterLink* l = buffersink_ctx->inputs[0];
  FilterGraphOutputInfo ret{};
  ret.type = l->type;
  ret.format = l->format;
  ret.time_base = l->time_base;
  switch (l->type) {
    case AVMEDIA_TYPE_AUDIO: {
      ret.sample_rate = l->sample_rate;
#if LIBAVFILTER_VERSION_MAJOR >= 10
      // FFmpeg 7+
      ret.num_channels = l->ch_layout.nb_channels;
#elif LIBAVFILTER_VERSION_MAJOR >= 8 && LIBAVFILTER_VERSION_MINOR >= 44
      // FFmpeg 5.1+
      ret.num_channels = l->ch_layout.nb_channels;
#else
      // Before FFmpeg 5.1
      ret.num_channels = av_get_channel_layout_nb_channels(l->channel_layout);
#endif
      break;
    }
    case AVMEDIA_TYPE_VIDEO: {
#if LIBAVFILTER_VERSION_MAJOR >= 10
      // FFmpeg 7+ - hw_frames_ctx not directly on link
      // Just use format as-is for now
      if (l->format == AV_PIX_FMT_CUDA) {
        // Try to get sw_format from buffersink
        ret.format = l->format;  // Keep CUDA format, let caller handle
      }
      ret.frame_rate = av_buffersink_get_frame_rate(buffersink_ctx);
#else
      if (l->format == AV_PIX_FMT_CUDA) {
        auto frames_ctx = [&]() -> AVHWFramesContext* {
          if (l->hw_frames_ctx) {
            return (AVHWFramesContext*)(l->hw_frames_ctx->data);
          }
          return (AVHWFramesContext*)(buffersrc_ctx->outputs[0]
                                          ->hw_frames_ctx->data);
        }();
        ret.format = frames_ctx->sw_format;
      }
      ret.frame_rate = l->frame_rate;
#endif
      ret.height = l->h;
      ret.width = l->w;
      break;
    }
    default:;
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Streaming process
//////////////////////////////////////////////////////////////////////////////
int FilterGraph::add_frame(AVFrame* pInputFrame) {
  return av_buffersrc_add_frame_flags(
      buffersrc_ctx, pInputFrame, AV_BUFFERSRC_FLAG_KEEP_REF);
}

int FilterGraph::get_frame(AVFrame* pOutputFrame) {
  return av_buffersink_get_frame(buffersink_ctx, pOutputFrame);
}

} // namespace torchffmpeg
