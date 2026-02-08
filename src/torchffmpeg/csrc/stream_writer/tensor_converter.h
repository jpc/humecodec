#pragma once

#include "torchffmpeg/csrc/ffmpeg.h"
#include "torchffmpeg/csrc/managed_buffer.h"

namespace torchffmpeg {

class TensorConverter {
 public:
  // Initialization is one-time process applied to frames before the iteration
  // starts. i.e. either convert to NHWC.
  using InitFunc = std::function<ManagedBuffer(const ManagedBuffer&, AVFrame*)>;
  // Convert function writes input frame ManagedBuffer to destination AVFrame
  // both buffer input and AVFrame are expected to be valid and properly
  // allocated. (i.e. glorified copy). It is used in Iterator.
  using ConvertFunc = std::function<void(const ManagedBuffer&, AVFrame*)>;

  //////////////////////////////////////////////////////////////////////////////
  // Generator
  //////////////////////////////////////////////////////////////////////////////
  class Generator {
   public:
    ////////////////////////////////////////////////////////////////////////////
    // Iterator
    ////////////////////////////////////////////////////////////////////////////
    class Iterator {
      const ManagedBuffer& frames;
      AVFrame* buffer;
      ConvertFunc& convert_func;

      int64_t step;
      int64_t i = 0;

     public:
      Iterator(
          const ManagedBuffer& buf,
          AVFrame* buffer,
          ConvertFunc& convert_func,
          int64_t step);

      Iterator& operator++();
      AVFrame* operator*() const;
      bool operator!=(const int64_t other) const;
    };

   private:
    ManagedBuffer frames;
    AVFrame* buffer;
    ConvertFunc& convert_func;
    int64_t step;

   public:
    Generator(
        ManagedBuffer frames,
        AVFrame* buffer,
        ConvertFunc& convert_func,
        int64_t step = 1);

    [[nodiscard]] Iterator begin() const;
    [[nodiscard]] int64_t end() const;
  };

 private:
  AVFrame* buffer;
  const int buffer_size = 1;

  InitFunc init_func{};
  ConvertFunc convert_func{};

 public:
  TensorConverter(AVMediaType type, AVFrame* buffer, int buffer_size = 1);
  Generator convert(const ManagedBuffer& t);
};

} // namespace torchffmpeg
