#include "humecodec/csrc/managed_buffer.h"

#ifdef _WIN32
#include <malloc.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace {
inline void* aligned_alloc_portable(size_t alignment, size_t size) {
  // std::aligned_alloc requires size to be a multiple of alignment (strictly
  // enforced on macOS). Round up to satisfy this constraint.
  size = (size + alignment - 1) & ~(alignment - 1);
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  return std::aligned_alloc(alignment, size);
#endif
}
inline void aligned_free_portable(void* ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}
}  // namespace

namespace humecodec {

ManagedBuffer::ManagedBuffer(
    const std::vector<int64_t>& shape,
    DLDataType dtype,
    DLDevice device)
    : shape_(shape), dtype_(dtype), device_(device) {
  // Compute strides (row-major contiguous layout)
  compute_strides();

  // Compute total size
  int64_t num_elements = numel();
  nbytes_ = static_cast<size_t>(num_elements) * element_size();

  if (nbytes_ == 0) {
    data_ = nullptr;
    return;
  }

  // Allocate memory
  if (device_.device_type == kDLCPU) {
    // Use aligned allocation for better performance
    data_ = aligned_alloc_portable(64, nbytes_);
    HCODEC_CHECK(data_ != nullptr, "Failed to allocate ", nbytes_, " bytes on CPU");
  } else if (device_.device_type == kDLCUDA) {
#ifdef USE_CUDA
    cudaError_t err = cudaMalloc(&data_, nbytes_);
    HCODEC_CHECK(
        err == cudaSuccess,
        "Failed to allocate ", nbytes_, " bytes on CUDA device ", device_.device_id,
        ": ", cudaGetErrorString(err));
#else
    HCODEC_CHECK(false, "CUDA support not compiled in");
#endif
  } else {
    HCODEC_CHECK(
        false, "Unsupported device type: ", static_cast<int>(device_.device_type));
  }
}

ManagedBuffer::~ManagedBuffer() {
  free_data();
}

ManagedBuffer::ManagedBuffer(ManagedBuffer&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      dtype_(other.dtype_),
      device_(other.device_),
      nbytes_(other.nbytes_),
      original_managed_(other.original_managed_) {
  other.data_ = nullptr;
  other.nbytes_ = 0;
  other.original_managed_ = nullptr;
}

ManagedBuffer& ManagedBuffer::operator=(ManagedBuffer&& other) noexcept {
  if (this != &other) {
    free_data();

    data_ = other.data_;
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    dtype_ = other.dtype_;
    device_ = other.device_;
    nbytes_ = other.nbytes_;
    original_managed_ = other.original_managed_;

    other.data_ = nullptr;
    other.nbytes_ = 0;
    other.original_managed_ = nullptr;
  }
  return *this;
}

void ManagedBuffer::free_data() {
  if (original_managed_) {
    // This buffer was created from DLPack, call original deleter
    if (original_managed_->deleter) {
      original_managed_->deleter(original_managed_);
    }
    original_managed_ = nullptr;
    data_ = nullptr;
    nbytes_ = 0;
    return;
  }

  if (data_ == nullptr) {
    return;
  }

  if (device_.device_type == kDLCPU) {
    aligned_free_portable(data_);
  } else if (device_.device_type == kDLCUDA) {
#ifdef USE_CUDA
    cudaFree(data_);
#endif
  }
  data_ = nullptr;
  nbytes_ = 0;
}

void ManagedBuffer::compute_strides() {
  strides_.resize(shape_.size());
  if (shape_.empty()) {
    return;
  }
  strides_.back() = 1;
  for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * shape_[i + 1];
  }
}

void ManagedBuffer::copy_from(const void* src, size_t bytes) {
  HCODEC_CHECK(bytes <= nbytes_, "Copy size ", bytes, " exceeds buffer size ", nbytes_);

  if (bytes == 0) {
    return;
  }

  if (device_.device_type == kDLCPU) {
    std::memcpy(data_, src, bytes);
  } else if (device_.device_type == kDLCUDA) {
#ifdef USE_CUDA
    cudaError_t err = cudaMemcpy(data_, src, bytes, cudaMemcpyHostToDevice);
    HCODEC_CHECK(
        err == cudaSuccess,
        "CUDA memcpy H2D failed: ", cudaGetErrorString(err));
#else
    HCODEC_CHECK(false, "CUDA support not compiled in");
#endif
  }
}

void ManagedBuffer::copy_to(void* dst, size_t bytes) const {
  HCODEC_CHECK(bytes <= nbytes_, "Copy size ", bytes, " exceeds buffer size ", nbytes_);

  if (bytes == 0) {
    return;
  }

  if (device_.device_type == kDLCPU) {
    std::memcpy(dst, data_, bytes);
  } else if (device_.device_type == kDLCUDA) {
#ifdef USE_CUDA
    cudaError_t err = cudaMemcpy(dst, data_, bytes, cudaMemcpyDeviceToHost);
    HCODEC_CHECK(
        err == cudaSuccess,
        "CUDA memcpy D2H failed: ", cudaGetErrorString(err));
#else
    HCODEC_CHECK(false, "CUDA support not compiled in");
#endif
  }
}

// Context structure for the DLPack deleter
struct ManagedTensorContext {
  void* data;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLDevice device;

  ~ManagedTensorContext() {
    if (data) {
      if (device.device_type == kDLCPU) {
        aligned_free_portable(data);
      } else if (device.device_type == kDLCUDA) {
#ifdef USE_CUDA
        cudaFree(data);
#endif
      }
    }
  }
};

static void managed_tensor_deleter(DLManagedTensor* self) {
  if (self) {
    delete static_cast<ManagedTensorContext*>(self->manager_ctx);
    delete self;
  }
}

DLManagedTensor* ManagedBuffer::to_dlpack() {
  HCODEC_CHECK(
      original_managed_ == nullptr,
      "Cannot export a buffer that was imported from DLPack");

  auto* managed = new DLManagedTensor();
  auto* ctx = new ManagedTensorContext();

  // Transfer ownership of data to context
  ctx->data = data_;
  ctx->shape = std::move(shape_);
  ctx->strides = std::move(strides_);
  ctx->device = device_;

  // Set up DLTensor
  managed->dl_tensor.data = ctx->data;
  managed->dl_tensor.device = device_;
  managed->dl_tensor.ndim = static_cast<int32_t>(ctx->shape.size());
  managed->dl_tensor.dtype = dtype_;
  managed->dl_tensor.shape = ctx->shape.data();
  managed->dl_tensor.strides = ctx->strides.data();
  managed->dl_tensor.byte_offset = 0;

  managed->manager_ctx = ctx;
  managed->deleter = managed_tensor_deleter;

  // Clear our state (ownership transferred)
  data_ = nullptr;
  shape_.clear();
  strides_.clear();
  nbytes_ = 0;

  return managed;
}

ManagedBuffer ManagedBuffer::from_dlpack(DLManagedTensor* managed) {
  HCODEC_CHECK(managed != nullptr, "DLManagedTensor is null");

  ManagedBuffer buf;
  const DLTensor& dl = managed->dl_tensor;

  buf.data_ = static_cast<uint8_t*>(dl.data) + dl.byte_offset;
  buf.shape_.assign(dl.shape, dl.shape + dl.ndim);
  if (dl.strides) {
    buf.strides_.assign(dl.strides, dl.strides + dl.ndim);
  } else {
    buf.compute_strides();
  }
  buf.dtype_ = dl.dtype;
  buf.device_ = dl.device;
  buf.nbytes_ = static_cast<size_t>(buf.numel()) * buf.element_size();
  buf.original_managed_ = managed;

  return buf;
}

}  // namespace humecodec
