#include "torchffmpeg/csrc/stream_writer/encoder.h"

namespace torchffmpeg {

Encoder::Encoder(
    AVFormatContext* format_ctx,
    AVCodecContext* codec_ctx,
    AVStream* stream) noexcept
    : format_ctx(format_ctx), codec_ctx(codec_ctx), stream(stream) {}

///
/// Encode the given AVFrame data
///
/// @param frame Frame data to encode
void Encoder::encode(AVFrame* frame) {
  int ret = avcodec_send_frame(codec_ctx, frame);
  TORCH_CHECK(ret >= 0, "Failed to encode frame (", av_err2string(ret), ").");
  while (ret >= 0) {
    ret = avcodec_receive_packet(codec_ctx, packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      if (ret == AVERROR_EOF) {
        ret = av_interleaved_write_frame(format_ctx, nullptr);
        TORCH_CHECK(
            ret >= 0, "Failed to flush packet (", av_err2string(ret), ").");
      }
      break;
    } else {
      TORCH_CHECK(
          ret >= 0,
          "Failed to fetch encoded packet (",
          av_err2string(ret),
          ").");
    }
    // https://github.com/pytorch/audio/issues/2790
    if (packet->duration == 0 && codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
      packet->duration = 1;
    }
    av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
    packet->stream_index = stream->index;

    ret = av_interleaved_write_frame(format_ctx, packet);
    TORCH_CHECK(ret >= 0, "Failed to write packet (", av_err2string(ret), ").");
  }
}

} // namespace torchffmpeg
