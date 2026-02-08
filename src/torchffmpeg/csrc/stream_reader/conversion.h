#pragma once
#include "torchffmpeg/csrc/ffmpeg.h"
#include "torchffmpeg/csrc/managed_buffer.h"
#include "torchffmpeg/csrc/tensor_view.h"

namespace torchffmpeg {

////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////
template <DLDataTypeCode type_code, uint8_t bits, bool is_planar>
class AudioConverter {
  const int num_channels;

 public:
  explicit AudioConverter(int num_channels);
  ManagedBuffer convert(const AVFrame* src);
};

////////////////////////////////////////////////////////////////////////////////
// Image
////////////////////////////////////////////////////////////////////////////////
struct ImageConverterBase {
  const int height;
  const int width;
  const int num_channels;

  ImageConverterBase(int h, int w, int c);
};

////////////////////////////////////////////////////////////////////////////////
// Interlaced Images - output as NCHW
////////////////////////////////////////////////////////////////////////////////
struct InterlacedImageConverter : public ImageConverterBase {
  using ImageConverterBase::ImageConverterBase;
  ManagedBuffer convert(const AVFrame* src);
};

struct Interlaced16BitImageConverter : public ImageConverterBase {
  using ImageConverterBase::ImageConverterBase;
  ManagedBuffer convert(const AVFrame* src);
};

////////////////////////////////////////////////////////////////////////////////
// Planar Images - NCHW
////////////////////////////////////////////////////////////////////////////////
struct PlanarImageConverter : public ImageConverterBase {
  using ImageConverterBase::ImageConverterBase;
  ManagedBuffer convert(const AVFrame* src);
};

////////////////////////////////////////////////////////////////////////////////
// Family of YUVs - NCHW
////////////////////////////////////////////////////////////////////////////////
class YUV420PConverter : public ImageConverterBase {
 public:
  YUV420PConverter(int height, int width);
  ManagedBuffer convert(const AVFrame* src);
};

class YUV420P10LEConverter : public ImageConverterBase {
 public:
  YUV420P10LEConverter(int height, int width);
  ManagedBuffer convert(const AVFrame* src);
};

class NV12Converter : public ImageConverterBase {
 public:
  NV12Converter(int height, int width);
  ManagedBuffer convert(const AVFrame* src);
};

#ifdef USE_CUDA

struct CudaImageConverterBase {
  const DLDevice cuda_device;
  bool init = false;
  int height = -1;
  int width = -1;
  explicit CudaImageConverterBase(const DLDevice& dev);
};

class NV12CudaConverter : CudaImageConverterBase {
 public:
  explicit NV12CudaConverter(const DLDevice& dev);
  ManagedBuffer convert(const AVFrame* src);
};

class P010CudaConverter : CudaImageConverterBase {
 public:
  explicit P010CudaConverter(const DLDevice& dev);
  ManagedBuffer convert(const AVFrame* src);
};

class YUV444PCudaConverter : CudaImageConverterBase {
 public:
  explicit YUV444PCudaConverter(const DLDevice& dev);
  ManagedBuffer convert(const AVFrame* src);
};

#endif

} // namespace torchffmpeg
