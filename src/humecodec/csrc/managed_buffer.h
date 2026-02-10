#pragma once

#include "humecodec/csrc/dlpack.h"
#include "humecodec/csrc/error_utils.h"
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>

namespace humecodec {

/**
 * @brief RAII buffer that owns tensor memory and can export to DLPack.
 *
 * ManagedBuffer allocates and owns tensor memory (either CPU or CUDA),
 * and provides methods to export the data as a DLManagedTensor for
 * zero-copy transfer to other frameworks via DLPack protocol.
 */
class ManagedBuffer {
 public:
  ManagedBuffer() = default;

  /**
   * @brief Allocate a new buffer with the given shape and dtype.
   *
   * @param shape The dimensions of the tensor.
   * @param dtype The data type of the tensor elements.
   * @param device The device to allocate on (CPU or CUDA).
   */
  ManagedBuffer(
      const std::vector<int64_t>& shape,
      DLDataType dtype,
      DLDevice device);

  ~ManagedBuffer();

  // Move-only semantics
  ManagedBuffer(ManagedBuffer&& other) noexcept;
  ManagedBuffer& operator=(ManagedBuffer&& other) noexcept;

  // Non-copyable
  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  // Data access
  void* data() {
    return data_;
  }
  const void* data() const {
    return data_;
  }

  template <typename T>
  T* data_ptr() {
    return static_cast<T*>(data_);
  }
  template <typename T>
  const T* data_ptr() const {
    return static_cast<const T*>(data_);
  }

  // Metadata access
  const std::vector<int64_t>& shape() const {
    return shape_;
  }
  const std::vector<int64_t>& strides() const {
    return strides_;
  }
  DLDataType dtype() const {
    return dtype_;
  }
  DLDevice device() const {
    return device_;
  }
  size_t nbytes() const {
    return nbytes_;
  }

  int ndim() const {
    return static_cast<int>(shape_.size());
  }

  int64_t size(int dim) const {
    if (dim < 0) {
      dim = ndim() + dim;
    }
    HCODEC_CHECK(
        dim >= 0 && dim < ndim(),
        "Dimension ", dim, " out of range for buffer with ", ndim(), " dimensions");
    return shape_[dim];
  }

  int64_t numel() const {
    if (shape_.empty()) {
      return 0;
    }
    return std::accumulate(
        shape_.begin(), shape_.end(), int64_t{1}, std::multiplies<int64_t>());
  }

  size_t element_size() const {
    return (dtype_.bits * dtype_.lanes + 7) / 8;
  }

  bool is_cpu() const {
    return device_.device_type == kDLCPU;
  }

  bool is_cuda() const {
    return device_.device_type == kDLCUDA;
  }

  bool is_valid() const {
    return data_ != nullptr || nbytes_ == 0;
  }

  /**
   * @brief Copy data into the buffer from a source pointer.
   *
   * @param src Source pointer.
   * @param bytes Number of bytes to copy.
   */
  void copy_from(const void* src, size_t bytes);

  /**
   * @brief Copy data from the buffer to a destination pointer.
   *
   * @param dst Destination pointer.
   * @param bytes Number of bytes to copy.
   */
  void copy_to(void* dst, size_t bytes) const;

  /**
   * @brief Export buffer as DLManagedTensor, transferring ownership.
   *
   * After calling this method, the ManagedBuffer is left in an empty state.
   * The caller is responsible for eventually calling the deleter on the
   * returned DLManagedTensor.
   *
   * @return Pointer to a newly allocated DLManagedTensor.
   */
  DLManagedTensor* to_dlpack();

  /**
   * @brief Create a ManagedBuffer by taking ownership of a DLManagedTensor.
   *
   * The ManagedBuffer will call the original deleter when destroyed.
   *
   * @param managed The DLManagedTensor to take ownership of.
   * @return A new ManagedBuffer owning the data.
   */
  static ManagedBuffer from_dlpack(DLManagedTensor* managed);

 private:
  void* data_ = nullptr;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  DLDataType dtype_{};
  DLDevice device_{};
  size_t nbytes_ = 0;

  // For buffers created from DLPack, we need to call the original deleter
  DLManagedTensor* original_managed_ = nullptr;

  void free_data();
  void compute_strides();
};

}  // namespace humecodec
