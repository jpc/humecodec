#pragma once

#include "torchffmpeg/csrc/error_utils.h"
#include <cstddef>

namespace torchffmpeg {

/**
 * @brief Optional CUDA utilities for memory management.
 *
 * These functions provide a thin wrapper over CUDA runtime functions,
 * allowing the library to manage CUDA memory without depending on PyTorch.
 */

#ifdef USE_CUDA

/**
 * @brief Allocate device memory.
 * @param size Number of bytes to allocate.
 * @return Pointer to allocated memory.
 */
void* cuda_malloc(size_t size);

/**
 * @brief Free device memory.
 * @param ptr Pointer to free.
 */
void cuda_free(void* ptr);

/**
 * @brief Copy memory from host to device.
 * @param dst Device pointer.
 * @param src Host pointer.
 * @param size Number of bytes.
 */
void cuda_memcpy_h2d(void* dst, const void* src, size_t size);

/**
 * @brief Copy memory from device to host.
 * @param dst Host pointer.
 * @param src Device pointer.
 * @param size Number of bytes.
 */
void cuda_memcpy_d2h(void* dst, const void* src, size_t size);

/**
 * @brief Copy memory from device to device.
 * @param dst Device pointer.
 * @param src Device pointer.
 * @param size Number of bytes.
 */
void cuda_memcpy_d2d(void* dst, const void* src, size_t size);

/**
 * @brief 2D memory copy from device to device.
 * @param dst Device pointer.
 * @param dpitch Pitch of dst memory.
 * @param src Device pointer.
 * @param spitch Pitch of src memory.
 * @param width Width in bytes.
 * @param height Height in rows.
 */
void cuda_memcpy2d_d2d(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height);

/**
 * @brief Get the current CUDA device.
 * @return Device index.
 */
int cuda_get_device();

/**
 * @brief Set the current CUDA device.
 * @param device Device index.
 */
void cuda_set_device(int device);

/**
 * @brief Synchronize the current CUDA device.
 */
void cuda_device_synchronize();

#else  // !USE_CUDA

// Stub implementations that throw errors when CUDA is not available

inline void* cuda_malloc(size_t /*size*/) {
  TFMPEG_CHECK(false, "CUDA support not compiled in");
  return nullptr;
}

inline void cuda_free(void* /*ptr*/) {
  TFMPEG_CHECK(false, "CUDA support not compiled in");
}

inline void cuda_memcpy_h2d(void* /*dst*/, const void* /*src*/, size_t /*size*/) {
  TFMPEG_CHECK(false, "CUDA support not compiled in");
}

inline void cuda_memcpy_d2h(void* /*dst*/, const void* /*src*/, size_t /*size*/) {
  TFMPEG_CHECK(false, "CUDA support not compiled in");
}

inline void cuda_memcpy_d2d(void* /*dst*/, const void* /*src*/, size_t /*size*/) {
  TFMPEG_CHECK(false, "CUDA support not compiled in");
}

inline void cuda_memcpy2d_d2d(
    void* /*dst*/,
    size_t /*dpitch*/,
    const void* /*src*/,
    size_t /*spitch*/,
    size_t /*width*/,
    size_t /*height*/) {
  TFMPEG_CHECK(false, "CUDA support not compiled in");
}

inline int cuda_get_device() {
  TFMPEG_CHECK(false, "CUDA support not compiled in");
  return -1;
}

inline void cuda_set_device(int /*device*/) {
  TFMPEG_CHECK(false, "CUDA support not compiled in");
}

inline void cuda_device_synchronize() {
  TFMPEG_CHECK(false, "CUDA support not compiled in");
}

#endif  // USE_CUDA

}  // namespace torchffmpeg
