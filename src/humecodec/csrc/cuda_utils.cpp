#include "humecodec/csrc/cuda_utils.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace humecodec {

#ifdef USE_CUDA

void* cuda_malloc(size_t size) {
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  HCODEC_CHECK(
      err == cudaSuccess,
      "cudaMalloc failed for ", size, " bytes: ", cudaGetErrorString(err));
  return ptr;
}

void cuda_free(void* ptr) {
  if (ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
      // Don't throw in destructor-like context, just log
      std::cerr << "Warning: cudaFree failed: " << cudaGetErrorString(err) << std::endl;
    }
  }
}

void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  HCODEC_CHECK(
      err == cudaSuccess,
      "cudaMemcpy H2D failed for ", size, " bytes: ", cudaGetErrorString(err));
}

void cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  HCODEC_CHECK(
      err == cudaSuccess,
      "cudaMemcpy D2H failed for ", size, " bytes: ", cudaGetErrorString(err));
}

void cuda_memcpy_d2d(void* dst, const void* src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  HCODEC_CHECK(
      err == cudaSuccess,
      "cudaMemcpy D2D failed for ", size, " bytes: ", cudaGetErrorString(err));
}

void cuda_memcpy2d_d2d(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height) {
  cudaError_t err = cudaMemcpy2D(
      dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
  HCODEC_CHECK(
      err == cudaSuccess,
      "cudaMemcpy2D D2D failed: ", cudaGetErrorString(err));
}

int cuda_get_device() {
  int device;
  cudaError_t err = cudaGetDevice(&device);
  HCODEC_CHECK(
      err == cudaSuccess,
      "cudaGetDevice failed: ", cudaGetErrorString(err));
  return device;
}

void cuda_set_device(int device) {
  cudaError_t err = cudaSetDevice(device);
  HCODEC_CHECK(
      err == cudaSuccess,
      "cudaSetDevice(", device, ") failed: ", cudaGetErrorString(err));
}

void cuda_device_synchronize() {
  cudaError_t err = cudaDeviceSynchronize();
  HCODEC_CHECK(
      err == cudaSuccess,
      "cudaDeviceSynchronize failed: ", cudaGetErrorString(err));
}

#endif  // USE_CUDA

}  // namespace humecodec
