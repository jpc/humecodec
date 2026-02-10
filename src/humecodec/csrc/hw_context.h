#pragma once

#include "humecodec/csrc/ffmpeg.h"

namespace humecodec {

AVBufferRef* get_cuda_context(int index);

void clear_cuda_context_cache();

} // namespace humecodec
