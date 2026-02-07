#pragma once

#include "torchffmpeg/csrc/ffmpeg.h"

namespace torchffmpeg {

AVBufferRef* get_cuda_context(int index);

void clear_cuda_context_cache();

} // namespace torchffmpeg
