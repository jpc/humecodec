#pragma once

#include "torchffmpeg/csrc/dlpack.h"
#include "torchffmpeg/csrc/error_utils.h"
#include <vector>
#include <cstdint>
#include <numeric>

namespace torchffmpeg {

/**
 * @brief Lightweight, non-owning view into tensor data.
 *
 * TensorView is a minimal structure that holds the essential information
 * needed to work with tensor data without owning the memory. It is designed
 * to be created from DLPack tensors and provides access to shape, strides,
 * dtype, and the underlying data pointer.
 */
struct TensorView {
  void* data = nullptr;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLDataType dtype{};
  DLDevice device{};

  TensorView() = default;

  TensorView(
      void* data_,
      std::vector<int64_t> shape_,
      std::vector<int64_t> strides_,
      DLDataType dtype_,
      DLDevice device_)
      : data(data_),
        shape(std::move(shape_)),
        strides(std::move(strides_)),
        dtype(dtype_),
        device(device_) {}

  // Number of dimensions
  int ndim() const {
    return static_cast<int>(shape.size());
  }

  // Total number of elements
  int64_t numel() const {
    if (shape.empty()) {
      return 0;
    }
    return std::accumulate(
        shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
  }

  // Number of bytes per element
  size_t element_size() const {
    return (dtype.bits * dtype.lanes + 7) / 8;
  }

  // Total size in bytes (for contiguous tensors)
  size_t nbytes() const {
    return static_cast<size_t>(numel()) * element_size();
  }

  // Get size of a specific dimension
  int64_t size(int dim) const {
    if (dim < 0) {
      dim = ndim() + dim;
    }
    TFMPEG_CHECK(
        dim >= 0 && dim < ndim(),
        "Dimension ", dim, " out of range for tensor with ", ndim(), " dimensions");
    return shape[dim];
  }

  // Get stride of a specific dimension
  int64_t stride(int dim) const {
    if (dim < 0) {
      dim = ndim() + dim;
    }
    TFMPEG_CHECK(
        dim >= 0 && dim < ndim(),
        "Dimension ", dim, " out of range for tensor with ", ndim(), " dimensions");
    return strides[dim];
  }

  // Check if tensor is contiguous in row-major (C) order
  bool is_contiguous() const {
    if (shape.empty()) {
      return true;
    }
    int64_t expected_stride = 1;
    for (int i = ndim() - 1; i >= 0; --i) {
      if (shape[i] != 1 && strides[i] != expected_stride) {
        return false;
      }
      expected_stride *= shape[i];
    }
    return true;
  }

  // Check if tensor is on CPU
  bool is_cpu() const {
    return device.device_type == kDLCPU;
  }

  // Check if tensor is on CUDA
  bool is_cuda() const {
    return device.device_type == kDLCUDA;
  }

  // Get typed data pointer
  template <typename T>
  T* data_ptr() const {
    return static_cast<T*>(data);
  }

  // Create TensorView from DLManagedTensor (non-owning)
  static TensorView from_dlpack(DLManagedTensor* managed) {
    TFMPEG_CHECK(managed != nullptr, "DLManagedTensor is null");
    const DLTensor& dl = managed->dl_tensor;

    std::vector<int64_t> shape_vec(dl.shape, dl.shape + dl.ndim);
    std::vector<int64_t> strides_vec;

    if (dl.strides) {
      strides_vec.assign(dl.strides, dl.strides + dl.ndim);
    } else {
      // Compute default contiguous strides
      strides_vec.resize(dl.ndim);
      if (dl.ndim > 0) {
        strides_vec[dl.ndim - 1] = 1;
        for (int i = dl.ndim - 2; i >= 0; --i) {
          strides_vec[i] = strides_vec[i + 1] * dl.shape[i + 1];
        }
      }
    }

    // Apply byte offset
    void* adjusted_data = static_cast<uint8_t*>(dl.data) + dl.byte_offset;

    return TensorView(
        adjusted_data,
        std::move(shape_vec),
        std::move(strides_vec),
        dl.dtype,
        dl.device);
  }

  // Create TensorView from DLTensor (non-owning)
  static TensorView from_dltensor(const DLTensor& dl) {
    std::vector<int64_t> shape_vec(dl.shape, dl.shape + dl.ndim);
    std::vector<int64_t> strides_vec;

    if (dl.strides) {
      strides_vec.assign(dl.strides, dl.strides + dl.ndim);
    } else {
      // Compute default contiguous strides
      strides_vec.resize(dl.ndim);
      if (dl.ndim > 0) {
        strides_vec[dl.ndim - 1] = 1;
        for (int i = dl.ndim - 2; i >= 0; --i) {
          strides_vec[i] = strides_vec[i + 1] * dl.shape[i + 1];
        }
      }
    }

    // Apply byte offset
    void* adjusted_data = static_cast<uint8_t*>(dl.data) + dl.byte_offset;

    return TensorView(
        adjusted_data,
        std::move(shape_vec),
        std::move(strides_vec),
        dl.dtype,
        dl.device);
  }
};

// Helper functions to create DLDataType from common types
namespace dtype {

inline DLDataType uint8() {
  return DLDataType{kDLUInt, 8, 1};
}

inline DLDataType int8() {
  return DLDataType{kDLInt, 8, 1};
}

inline DLDataType int16() {
  return DLDataType{kDLInt, 16, 1};
}

inline DLDataType int32() {
  return DLDataType{kDLInt, 32, 1};
}

inline DLDataType int64() {
  return DLDataType{kDLInt, 64, 1};
}

inline DLDataType float32() {
  return DLDataType{kDLFloat, 32, 1};
}

inline DLDataType float64() {
  return DLDataType{kDLFloat, 64, 1};
}

}  // namespace dtype

// Helper functions to create DLDevice
namespace device {

inline DLDevice cpu() {
  return DLDevice{kDLCPU, 0};
}

inline DLDevice cuda(int device_id = 0) {
  return DLDevice{kDLCUDA, device_id};
}

}  // namespace device

}  // namespace torchffmpeg
