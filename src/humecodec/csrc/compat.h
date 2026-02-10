#pragma once

// FFmpeg version compatibility macros
// FFmpeg 7.0+ has significant API changes for channel layouts

extern "C" {
#include <libavutil/version.h>
#include <libavutil/channel_layout.h>
#include <libavcodec/avcodec.h>
}

// FFmpeg 7.0+ uses AVChannelLayout instead of uint64_t channel_layout
#if LIBAVUTIL_VERSION_MAJOR >= 59

// Helper to get number of channels from AVChannelLayout
inline int get_nb_channels(const AVChannelLayout* ch_layout) {
  return ch_layout->nb_channels;
}

// Helper to set default channel layout
inline void set_default_channel_layout(AVChannelLayout* ch_layout, int nb_channels) {
  av_channel_layout_default(ch_layout, nb_channels);
}

// Helper to get number of channels from codec context
inline int get_codec_ctx_channels(const AVCodecContext* ctx) {
  return ctx->ch_layout.nb_channels;
}

// Helper to get number of channels from codec parameters
inline int get_codecpar_channels(const AVCodecParameters* par) {
  return par->ch_layout.nb_channels;
}

// Helper to get number of channels from frame
inline int get_frame_channels(const AVFrame* frame) {
  return frame->ch_layout.nb_channels;
}

// Helper to copy channel layout
inline void copy_channel_layout(AVChannelLayout* dst, const AVChannelLayout* src) {
  av_channel_layout_copy(dst, src);
}

// Helper to compare channel layouts
inline bool channel_layouts_equal(const AVChannelLayout* a, const AVChannelLayout* b) {
  return av_channel_layout_compare(a, b) == 0;
}

// Helper for default stereo layout
inline AVChannelLayout get_default_stereo_layout() {
  AVChannelLayout layout = {};
  av_channel_layout_default(&layout, 2);
  return layout;
}

#else

// Legacy FFmpeg 4-6 compatibility

inline int get_nb_channels(const uint64_t* channel_layout) {
  return av_get_channel_layout_nb_channels(*channel_layout);
}

inline void set_default_channel_layout(uint64_t* channel_layout, int nb_channels) {
  *channel_layout = av_get_default_channel_layout(nb_channels);
}

inline int get_codec_ctx_channels(const AVCodecContext* ctx) {
  return ctx->channels;
}

inline int get_codecpar_channels(const AVCodecParameters* par) {
  return par->channels;
}

inline int get_frame_channels(const AVFrame* frame) {
  return frame->channels;
}

#endif

// AVFilterLink changes in FFmpeg 7
#if LIBAVFILTER_VERSION_MAJOR >= 10

#include <libavfilter/avfilter.h>

inline AVBufferRef* get_filter_link_hw_frames_ctx(AVFilterLink* link) {
  // In FFmpeg 7+, hw_frames_ctx is accessed differently
  // Use avfilter_link_get_hw_frames_ctx() if available
  return nullptr;  // TODO: implement if hw accel is needed
}

inline AVRational get_filter_link_frame_rate(AVFilterLink* link) {
  return av_buffersink_get_frame_rate(link->dst);
}

#else

inline AVBufferRef* get_filter_link_hw_frames_ctx(AVFilterLink* link) {
  return link->hw_frames_ctx;
}

inline AVRational get_filter_link_frame_rate(AVFilterLink* link) {
  return link->frame_rate;
}

#endif

// Write callback signature changed in FFmpeg 7
#if LIBAVFORMAT_VERSION_MAJOR >= 61

using WritePacketFunc = int (*)(void* opaque, const uint8_t* buf, int buf_size);

#else

using WritePacketFunc = int (*)(void* opaque, uint8_t* buf, int buf_size);

#endif
