#include "humecodec/csrc/avdevice_loader.h"
#include "humecodec/csrc/hw_context.h"
#include "humecodec/csrc/stream_reader/stream_reader.h"
#include "humecodec/csrc/stream_writer/stream_writer.h"
#include "humecodec/csrc/dlpack.h"
#include "humecodec/csrc/managed_buffer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cassert>
#include <cstring>

extern "C" {
#include <libavformat/version.h>
}

namespace py = pybind11;

namespace humecodec {
namespace {

// DLPack capsule utilities

// Export a ManagedBuffer as a DLPack PyCapsule.
// This transfers ownership of the underlying data to the capsule.
py::capsule managed_buffer_to_capsule(ManagedBuffer& buf) {
  DLManagedTensor* managed = buf.to_dlpack();
  return py::capsule(managed, "dltensor", [](PyObject* obj) {
    const char* name = PyCapsule_GetName(obj);
    if (name && std::strcmp(name, "used_dltensor") == 0) {
      return;
    }
    auto* ptr = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(obj, "dltensor"));
    if (ptr && ptr->deleter) {
      ptr->deleter(ptr);
    }
  });
}

// Create a ManagedBuffer from a DLPack PyCapsule.
// Follows the DLPack protocol (renames capsule to "used_dltensor").
ManagedBuffer capsule_to_managed_buffer(py::capsule capsule) {
  const char* name = PyCapsule_GetName(capsule.ptr());
  if (name && std::strcmp(name, "dltensor") != 0) {
    throw std::runtime_error("Expected capsule with name 'dltensor'");
  }

  DLManagedTensor* managed = static_cast<DLManagedTensor*>(
      PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
  if (!managed) {
    throw std::runtime_error("Invalid DLPack capsule");
  }

  PyCapsule_SetName(capsule.ptr(), "used_dltensor");
  return ManagedBuffer::from_dlpack(managed);
}

std::map<std::string, std::tuple<int64_t, int64_t, int64_t>> get_versions() {
  std::map<std::string, std::tuple<int64_t, int64_t, int64_t>> ret;

#define add_version(NAME)            \
  {                                  \
    int ver = NAME##_version();      \
    ret.emplace(                     \
        "lib" #NAME,                 \
        std::make_tuple<>(           \
            AV_VERSION_MAJOR(ver),   \
            AV_VERSION_MINOR(ver),   \
            AV_VERSION_MICRO(ver))); \
  }

  add_version(avutil);
  add_version(avcodec);
  add_version(avformat);
  add_version(avfilter);

#undef add_version

  // libavdevice is loaded at runtime via dlopen (to avoid macOS ObjC class
  // conflicts when the user already has FFmpeg installed).  Report its
  // version only when available.
  int avdevice_ver = avdevice_loader::version();
  if (avdevice_ver >= 0) {
    ret.emplace(
        "libavdevice",
        std::make_tuple<>(
            AV_VERSION_MAJOR(avdevice_ver),
            AV_VERSION_MINOR(avdevice_ver),
            AV_VERSION_MICRO(avdevice_ver)));
  }
  return ret;
}

std::map<std::string, std::string> get_demuxers(bool req_device) {
  std::map<std::string, std::string> ret;
  const AVInputFormat* fmt = nullptr;
  void* i = nullptr;
  while ((fmt = av_demuxer_iterate(&i))) {
    assert(fmt);
    bool is_device = [&]() {
      const AVClass* avclass = fmt->priv_class;
      return avclass && AV_IS_INPUT_DEVICE(avclass->category);
    }();
    if (req_device == is_device) {
      ret.emplace(fmt->name, fmt->long_name);
    }
  }
  return ret;
}

std::map<std::string, std::string> get_muxers(bool req_device) {
  std::map<std::string, std::string> ret;
  const AVOutputFormat* fmt = nullptr;
  void* i = nullptr;
  while ((fmt = av_muxer_iterate(&i))) {
    assert(fmt);
    bool is_device = [&]() {
      const AVClass* avclass = fmt->priv_class;
      return avclass && AV_IS_OUTPUT_DEVICE(avclass->category);
    }();
    if (req_device == is_device) {
      ret.emplace(fmt->name, fmt->long_name);
    }
  }
  return ret;
}

std::map<std::string, std::string> get_codecs(
    AVMediaType type,
    bool req_encoder) {
  const AVCodec* c = nullptr;
  void* i = nullptr;
  std::map<std::string, std::string> ret;
  while ((c = av_codec_iterate(&i))) {
    assert(c);
    if ((req_encoder && av_codec_is_encoder(c)) ||
        (!req_encoder && av_codec_is_decoder(c))) {
      if (c->type == type && c->name) {
        ret.emplace(c->name, c->long_name ? c->long_name : "");
      }
    }
  }
  return ret;
}

std::vector<std::string> get_protocols(bool output) {
  void* opaque = nullptr;
  const char* name = nullptr;
  std::vector<std::string> ret;
  while ((name = avio_enum_protocols(&opaque, output))) {
    assert(name);
    ret.emplace_back(name);
  }
  return ret;
}

std::string get_build_config() {
  return avcodec_configuration();
}

//////////////////////////////////////////////////////////////////////////////
// StreamingMediaDecoder/Encoder FileObj
//////////////////////////////////////////////////////////////////////////////

struct FileObj {
  py::object fileobj;
  int buffer_size;
};

namespace {

static int read_func(void* opaque, uint8_t* buf, int buf_size) {
  FileObj* fileobj = static_cast<FileObj*>(opaque);
  buf_size = FFMIN(buf_size, fileobj->buffer_size);

  int num_read = 0;
  while (num_read < buf_size) {
    int request = buf_size - num_read;
    auto chunk = static_cast<std::string>(
        static_cast<py::bytes>(fileobj->fileobj.attr("read")(request)));
    auto chunk_len = chunk.length();
    if (chunk_len == 0) {
      break;
    }
    HCODEC_CHECK(
        chunk_len <= request,
        "Requested up to ",
        request,
        " bytes but, received ",
        chunk_len,
        " bytes. The given object does not confirm to read protocol of file object.");
    memcpy(buf, chunk.data(), chunk_len);
    buf += chunk_len;
    num_read += static_cast<int>(chunk_len);
  }
  return num_read == 0 ? AVERROR_EOF : num_read;
}

#if LIBAVFORMAT_VERSION_MAJOR >= 61
static int write_func(void* opaque, const uint8_t* buf, int buf_size) {
#else
static int write_func(void* opaque, uint8_t* buf, int buf_size) {
#endif
  FileObj* fileobj = static_cast<FileObj*>(opaque);
  buf_size = FFMIN(buf_size, fileobj->buffer_size);

  py::bytes b(reinterpret_cast<const char*>(buf), buf_size);
  fileobj->fileobj.attr("write")(b);
  return buf_size;
}

static int64_t seek_func(void* opaque, int64_t offset, int whence) {
  if (whence == AVSEEK_SIZE) {
    return AVERROR(EIO);
  }
  FileObj* fileobj = static_cast<FileObj*>(opaque);
  return py::cast<int64_t>(fileobj->fileobj.attr("seek")(offset, whence));
}

} // namespace

struct StreamingMediaDecoderFileObj : private FileObj,
                                      public StreamingMediaDecoderCustomIO {
  StreamingMediaDecoderFileObj(
      py::object fileobj,
      const std::optional<std::string>& format,
      const std::optional<std::map<std::string, std::string>>& option,
      int buffer_size)
      : FileObj{fileobj, buffer_size},
        StreamingMediaDecoderCustomIO(
            this,
            format,
            buffer_size,
            read_func,
            py::hasattr(fileobj, "seek") ? &seek_func : nullptr,
            option) {}
};

struct StreamingMediaEncoderFileObj : private FileObj,
                                      public StreamingMediaEncoderCustomIO {
  StreamingMediaEncoderFileObj(
      py::object fileobj,
      const std::optional<std::string>& format,
      int buffer_size)
      : FileObj{fileobj, buffer_size},
        StreamingMediaEncoderCustomIO(
            this,
            format,
            buffer_size,
            write_func,
            py::hasattr(fileobj, "seek") ? &seek_func : nullptr) {}
};

//////////////////////////////////////////////////////////////////////////////
// StreamingMediaDecoder/Encoder Bytes
//////////////////////////////////////////////////////////////////////////////
struct BytesWrapper {
  std::string_view src;
  size_t index = 0;
};

static int read_bytes(void* opaque, uint8_t* buf, int buf_size) {
  BytesWrapper* wrapper = static_cast<BytesWrapper*>(opaque);

  auto num_read = FFMIN(wrapper->src.size() - wrapper->index, buf_size);
  if (num_read == 0) {
    return AVERROR_EOF;
  }
  auto head = wrapper->src.data() + wrapper->index;
  memcpy(buf, head, num_read);
  wrapper->index += num_read;
  return num_read;
}

static int64_t seek_bytes(void* opaque, int64_t offset, int whence) {
  BytesWrapper* wrapper = static_cast<BytesWrapper*>(opaque);
  if (whence == AVSEEK_SIZE) {
    return wrapper->src.size();
  }

  if (whence == SEEK_SET) {
    wrapper->index = offset;
  } else if (whence == SEEK_CUR) {
    wrapper->index += offset;
  } else if (whence == SEEK_END) {
    wrapper->index = wrapper->src.size() + offset;
  } else {
    HCODEC_INTERNAL_ASSERT(false, "Unexpected whence value: ", whence);
  }
  return static_cast<int64_t>(wrapper->index);
}

struct StreamingMediaDecoderBytes : private BytesWrapper,
                                    public StreamingMediaDecoderCustomIO {
  StreamingMediaDecoderBytes(
      std::string_view src,
      const std::optional<std::string>& format,
      const std::optional<std::map<std::string, std::string>>& option,
      int64_t buffer_size)
      : BytesWrapper{src},
        StreamingMediaDecoderCustomIO(
            this,
            format,
            buffer_size,
            read_bytes,
            seek_bytes,
            option) {}
};

//////////////////////////////////////////////////////////////////////////////
// Helper: pop_chunks_dlpack for any decoder type
//////////////////////////////////////////////////////////////////////////////
template <typename DecoderT>
py::list pop_chunks_dlpack_impl(DecoderT& self) {
  py::list result;
  auto chunks = self.pop_chunks();
  for (auto& chunk : chunks) {
    if (chunk.has_value()) {
      py::tuple item = py::make_tuple(
          managed_buffer_to_capsule(chunk->frames),
          chunk->pts);
      result.append(item);
    } else {
      result.append(py::none());
    }
  }
  return result;
}

//////////////////////////////////////////////////////////////////////////////
// Helper: write_*_chunk_dlpack for any encoder type
//////////////////////////////////////////////////////////////////////////////
template <typename EncoderT>
void write_audio_chunk_dlpack_impl(
    EncoderT& self,
    int i,
    py::capsule capsule,
    const std::optional<double>& pts) {
  auto buf = capsule_to_managed_buffer(std::move(capsule));
  self.write_audio_chunk(i, buf, pts);
}

template <typename EncoderT>
void write_video_chunk_dlpack_impl(
    EncoderT& self,
    int i,
    py::capsule capsule,
    const std::optional<double>& pts) {
  auto buf = capsule_to_managed_buffer(std::move(capsule));
  self.write_video_chunk(i, buf, pts);
}

PYBIND11_MODULE(_humecodec, m) {
  m.def("init", []() { avdevice_loader::register_all(); });
  m.def("get_log_level", []() { return av_log_get_level(); });
  m.def("set_log_level", [](int level) { av_log_set_level(level); });
  m.def("get_versions", &get_versions);
  m.def("get_muxers", []() { return get_muxers(false); });
  m.def("get_demuxers", []() { return get_demuxers(false); });
  m.def("get_input_devices", []() { return get_demuxers(true); });
  m.def("get_build_config", &get_build_config);
  m.def("get_output_devices", []() { return get_muxers(true); });
  m.def("get_audio_decoders", []() {
    return get_codecs(AVMEDIA_TYPE_AUDIO, false);
  });
  m.def("get_audio_encoders", []() {
    return get_codecs(AVMEDIA_TYPE_AUDIO, true);
  });
  m.def("get_video_decoders", []() {
    return get_codecs(AVMEDIA_TYPE_VIDEO, false);
  });
  m.def("get_video_encoders", []() {
    return get_codecs(AVMEDIA_TYPE_VIDEO, true);
  });
  m.def("get_input_protocols", []() { return get_protocols(false); });
  m.def("get_output_protocols", []() { return get_protocols(true); });
  m.def("clear_cuda_context_cache", &clear_cuda_context_cache);

  py::class_<PacketIndexEntry>(m, "PacketIndexEntry", py::module_local())
      .def_readonly("pts", &PacketIndexEntry::pts)
      .def_readonly("pts_seconds", &PacketIndexEntry::pts_seconds)
      .def_readonly("pos", &PacketIndexEntry::pos)
      .def_readonly("size", &PacketIndexEntry::size)
      .def_readonly("is_key", &PacketIndexEntry::is_key);
  py::class_<CodecConfig>(m, "CodecConfig", py::module_local())
      .def(py::init<int, int, const std::optional<int>&, int, int>());
  py::class_<StreamingMediaEncoder>(
      m, "StreamingMediaEncoder", py::module_local())
      .def(py::init<const std::string&, const std::optional<std::string>&>())
      .def("set_metadata", &StreamingMediaEncoder::set_metadata)
      .def("add_audio_stream", &StreamingMediaEncoder::add_audio_stream)
      .def("add_video_stream", &StreamingMediaEncoder::add_video_stream)
      .def("dump_format", &StreamingMediaEncoder::dump_format)
      .def("open", &StreamingMediaEncoder::open)
      .def("write_audio_chunk_dlpack", &write_audio_chunk_dlpack_impl<StreamingMediaEncoder>)
      .def("write_video_chunk_dlpack", &write_video_chunk_dlpack_impl<StreamingMediaEncoder>)
      .def("flush", &StreamingMediaEncoder::flush)
      .def("close", &StreamingMediaEncoder::close);
  py::class_<StreamingMediaEncoderFileObj>(
      m, "StreamingMediaEncoderFileObj", py::module_local())
      .def(py::init<py::object, const std::optional<std::string>&, int64_t>())
      .def("set_metadata", &StreamingMediaEncoderFileObj::set_metadata)
      .def("add_audio_stream", &StreamingMediaEncoderFileObj::add_audio_stream)
      .def("add_video_stream", &StreamingMediaEncoderFileObj::add_video_stream)
      .def("dump_format", &StreamingMediaEncoderFileObj::dump_format)
      .def("open", &StreamingMediaEncoderFileObj::open)
      .def("write_audio_chunk_dlpack", &write_audio_chunk_dlpack_impl<StreamingMediaEncoderFileObj>)
      .def("write_video_chunk_dlpack", &write_video_chunk_dlpack_impl<StreamingMediaEncoderFileObj>)
      .def("flush", &StreamingMediaEncoderFileObj::flush)
      .def("close", &StreamingMediaEncoderFileObj::close);
  py::class_<OutputStreamInfo>(m, "OutputStreamInfo", py::module_local())
      .def_readonly("source_index", &OutputStreamInfo::source_index)
      .def_readonly("filter_description", &OutputStreamInfo::filter_description)
      .def_property_readonly(
          "media_type",
          [](const OutputStreamInfo& o) -> std::string {
            return av_get_media_type_string(o.media_type);
          })
      .def_property_readonly(
          "format",
          [](const OutputStreamInfo& o) -> std::string {
            switch (o.media_type) {
              case AVMEDIA_TYPE_AUDIO:
                return av_get_sample_fmt_name((AVSampleFormat)(o.format));
              case AVMEDIA_TYPE_VIDEO:
                return av_get_pix_fmt_name((AVPixelFormat)(o.format));
              default:
                HCODEC_INTERNAL_ASSERT(
                    false,
                    "FilterGraph is returning unexpected media type: ",
                    av_get_media_type_string(o.media_type));
            }
          })
      .def_readonly("sample_rate", &OutputStreamInfo::sample_rate)
      .def_readonly("num_channels", &OutputStreamInfo::num_channels)
      .def_readonly("width", &OutputStreamInfo::width)
      .def_readonly("height", &OutputStreamInfo::height)
      .def_property_readonly(
          "frame_rate", [](const OutputStreamInfo& o) -> double {
            if (o.frame_rate.den == 0) {
              HCODEC_WARN(
                  "Invalid frame rate is found: ",
                  o.frame_rate.num,
                  "/",
                  o.frame_rate.den);
              return -1;
            }
            return static_cast<double>(o.frame_rate.num) / o.frame_rate.den;
          });
  py::class_<SrcStreamInfo>(m, "SourceStreamInfo", py::module_local())
      .def_property_readonly(
          "media_type",
          [](const SrcStreamInfo& s) {
            return av_get_media_type_string(s.media_type);
          })
      .def_readonly("codec_name", &SrcStreamInfo::codec_name)
      .def_readonly("codec_long_name", &SrcStreamInfo::codec_long_name)
      .def_readonly("format", &SrcStreamInfo::fmt_name)
      .def_readonly("bit_rate", &SrcStreamInfo::bit_rate)
      .def_readonly("num_frames", &SrcStreamInfo::num_frames)
      .def_readonly("bits_per_sample", &SrcStreamInfo::bits_per_sample)
      .def_readonly("metadata", &SrcStreamInfo::metadata)
      .def_readonly("time_base_num", &SrcStreamInfo::time_base_num)
      .def_readonly("time_base_den", &SrcStreamInfo::time_base_den)
      .def_readonly("sample_rate", &SrcStreamInfo::sample_rate)
      .def_readonly("num_channels", &SrcStreamInfo::num_channels)
      .def_readonly("width", &SrcStreamInfo::width)
      .def_readonly("height", &SrcStreamInfo::height)
      .def_readonly("frame_rate", &SrcStreamInfo::frame_rate);
  py::class_<StreamingMediaDecoder>(
      m, "StreamingMediaDecoder", py::module_local())
      .def(py::init<
           const std::string&,
           const std::optional<std::string>&,
           const std::optional<OptionDict>&>())
      .def("num_src_streams", &StreamingMediaDecoder::num_src_streams)
      .def("num_out_streams", &StreamingMediaDecoder::num_out_streams)
      .def(
          "find_best_audio_stream",
          &StreamingMediaDecoder::find_best_audio_stream)
      .def(
          "find_best_video_stream",
          &StreamingMediaDecoder::find_best_video_stream)
      .def("get_metadata", &StreamingMediaDecoder::get_metadata)
      .def("get_src_stream_info", &StreamingMediaDecoder::get_src_stream_info)
      .def("get_out_stream_info", &StreamingMediaDecoder::get_out_stream_info)
      .def("seek", &StreamingMediaDecoder::seek)
      .def("seek_to_byte_offset", &StreamingMediaDecoder::seek_to_byte_offset)
      .def("add_audio_stream", &StreamingMediaDecoder::add_audio_stream)
      .def("add_video_stream", &StreamingMediaDecoder::add_video_stream)
      .def("remove_stream", &StreamingMediaDecoder::remove_stream)
      .def(
          "process_packet",
          py::overload_cast<const std::optional<double>&, const double>(
              &StreamingMediaDecoder::process_packet))
      .def("process_all_packets", &StreamingMediaDecoder::process_all_packets)
      .def("fill_buffer", &StreamingMediaDecoder::fill_buffer)
      .def("is_buffer_ready", &StreamingMediaDecoder::is_buffer_ready)
      .def("pop_chunks_dlpack", &pop_chunks_dlpack_impl<StreamingMediaDecoder>)
      .def("build_packet_index", &StreamingMediaDecoder::build_packet_index,
           py::arg("stream_index"), py::arg("resolution_bytes") = 524288);
  py::class_<StreamingMediaDecoderFileObj>(
      m, "StreamingMediaDecoderFileObj", py::module_local())
      .def(py::init<
           py::object,
           const std::optional<std::string>&,
           const std::optional<OptionDict>&,
           int64_t>())
      .def("num_src_streams", &StreamingMediaDecoderFileObj::num_src_streams)
      .def("num_out_streams", &StreamingMediaDecoderFileObj::num_out_streams)
      .def(
          "find_best_audio_stream",
          &StreamingMediaDecoderFileObj::find_best_audio_stream)
      .def(
          "find_best_video_stream",
          &StreamingMediaDecoderFileObj::find_best_video_stream)
      .def("get_metadata", &StreamingMediaDecoderFileObj::get_metadata)
      .def(
          "get_src_stream_info",
          &StreamingMediaDecoderFileObj::get_src_stream_info)
      .def(
          "get_out_stream_info",
          &StreamingMediaDecoderFileObj::get_out_stream_info)
      .def("seek", &StreamingMediaDecoderFileObj::seek)
      .def("seek_to_byte_offset", &StreamingMediaDecoderFileObj::seek_to_byte_offset)
      .def("add_audio_stream", &StreamingMediaDecoderFileObj::add_audio_stream)
      .def("add_video_stream", &StreamingMediaDecoderFileObj::add_video_stream)
      .def("remove_stream", &StreamingMediaDecoderFileObj::remove_stream)
      .def(
          "process_packet",
          py::overload_cast<const std::optional<double>&, const double>(
              &StreamingMediaDecoder::process_packet))
      .def(
          "process_all_packets",
          &StreamingMediaDecoderFileObj::process_all_packets)
      .def("fill_buffer", &StreamingMediaDecoderFileObj::fill_buffer)
      .def("is_buffer_ready", &StreamingMediaDecoderFileObj::is_buffer_ready)
      .def("pop_chunks_dlpack", &pop_chunks_dlpack_impl<StreamingMediaDecoderFileObj>)
      .def("build_packet_index", &StreamingMediaDecoderFileObj::build_packet_index,
           py::arg("stream_index"), py::arg("resolution_bytes") = 524288);
  py::class_<StreamingMediaDecoderBytes>(
      m, "StreamingMediaDecoderBytes", py::module_local())
      .def(py::init<
           std::string_view,
           const std::optional<std::string>&,
           const std::optional<OptionDict>&,
           int64_t>())
      .def("num_src_streams", &StreamingMediaDecoderBytes::num_src_streams)
      .def("num_out_streams", &StreamingMediaDecoderBytes::num_out_streams)
      .def(
          "find_best_audio_stream",
          &StreamingMediaDecoderBytes::find_best_audio_stream)
      .def(
          "find_best_video_stream",
          &StreamingMediaDecoderBytes::find_best_video_stream)
      .def("get_metadata", &StreamingMediaDecoderBytes::get_metadata)
      .def(
          "get_src_stream_info",
          &StreamingMediaDecoderBytes::get_src_stream_info)
      .def(
          "get_out_stream_info",
          &StreamingMediaDecoderBytes::get_out_stream_info)
      .def("seek", &StreamingMediaDecoderBytes::seek)
      .def("seek_to_byte_offset", &StreamingMediaDecoderBytes::seek_to_byte_offset)
      .def("add_audio_stream", &StreamingMediaDecoderBytes::add_audio_stream)
      .def("add_video_stream", &StreamingMediaDecoderBytes::add_video_stream)
      .def("remove_stream", &StreamingMediaDecoderBytes::remove_stream)
      .def(
          "process_packet",
          py::overload_cast<const std::optional<double>&, const double>(
              &StreamingMediaDecoder::process_packet))
      .def(
          "process_all_packets",
          &StreamingMediaDecoderBytes::process_all_packets)
      .def("fill_buffer", &StreamingMediaDecoderBytes::fill_buffer)
      .def("is_buffer_ready", &StreamingMediaDecoderBytes::is_buffer_ready)
      .def("pop_chunks_dlpack", &pop_chunks_dlpack_impl<StreamingMediaDecoderBytes>)
      .def("build_packet_index", &StreamingMediaDecoderBytes::build_packet_index,
           py::arg("stream_index"), py::arg("resolution_bytes") = 524288);
}

} // namespace
} // namespace humecodec
