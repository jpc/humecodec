/*!
 *  Copyright (c) 2017 -  by Contributors
 * \file dlpack.h
 * \brief The common header of DLPack.
 *
 * Vendored from https://github.com/dmlc/dlpack
 * MIT License
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"
#else
#define DLPACK_EXTERN_C
#endif

/*! \brief The current major version of dlpack */
#define DLPACK_MAJOR_VERSION 1

/*! \brief The current minor version of dlpack */
#define DLPACK_MINOR_VERSION 0

#ifdef _WIN32
#ifdef DLPACK_EXPORTS
#define DLPACK_DLL __declspec(dllexport)
#else
#define DLPACK_DLL __declspec(dllimport)
#endif
#else
#define DLPACK_DLL
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief The device type in DLDevice.
 */
#ifdef __cplusplus
typedef enum : int32_t {
#else
typedef enum {
#endif
  /*! \brief CPU device */
  kDLCPU = 1,
  /*! \brief CUDA GPU device */
  kDLCUDA = 2,
  /*! \brief Pinned CUDA CPU memory by cudaMallocHost */
  kDLCUDAHost = 3,
  /*! \brief OpenCL devices. */
  kDLOpenCL = 4,
  /*! \brief Vulkan buffer for next generation graphics. */
  kDLVulkan = 7,
  /*! \brief Metal for Apple GPU. */
  kDLMetal = 8,
  /*! \brief Verilog simulator buffer */
  kDLVPI = 9,
  /*! \brief ROCm GPUs for AMD GPUs */
  kDLROCM = 10,
  /*! \brief Pinned ROCm CPU memory allocated by hipMallocHost */
  kDLROCMHost = 11,
  /*! \brief Reserved extension device type */
  kDLExtDev = 12,
  /*! \brief CUDA managed/unified memory allocated by cudaMallocManaged */
  kDLCUDAManaged = 13,
  /*! \brief Unified shared memory allocated on a oneAPI non-partitioned device */
  kDLOneAPI = 14,
  /*! \brief GPU support for next generation WebGPU standard. */
  kDLWebGPU = 15,
  /*! \brief Qualcomm Hexagon DSP */
  kDLHexagon = 16,
} DLDeviceType;

/*!
 * \brief A Device for Tensor and operator.
 */
typedef struct {
  /*! \brief The device type used in the device. */
  DLDeviceType device_type;
  /*! \brief The device index. For vanilla CPU memory, pinned memory, or managed memory, this is set to 0. */
  int32_t device_id;
} DLDevice;

/*!
 * \brief The type code options DLDataType.
 */
typedef enum {
  /*! \brief signed integer */
  kDLInt = 0U,
  /*! \brief unsigned integer */
  kDLUInt = 1U,
  /*! \brief IEEE floating point */
  kDLFloat = 2U,
  /*! \brief Opaque handle type, reserved for testing purposes. */
  kDLOpaqueHandle = 3U,
  /*! \brief bfloat16 */
  kDLBfloat = 4U,
  /*! \brief complex number (C/C++/Python layout: compact struct per complex number) */
  kDLComplex = 5U,
  /*! \brief boolean */
  kDLBool = 6U,
} DLDataTypeCode;

/*!
 * \brief The data type the tensor can hold.
 */
typedef struct {
  /*! \brief Type code of base types. */
  uint8_t code;
  /*! \brief Number of bits, common choices are 8, 16, 32. */
  uint8_t bits;
  /*! \brief Number of lanes in the type, used for vector types. */
  uint16_t lanes;
} DLDataType;

/*!
 * \brief Plain C Tensor object, does not manage memory.
 */
typedef struct {
  /*! \brief The data pointer points to the allocated data. */
  void* data;
  /*! \brief The device of the tensor */
  DLDevice device;
  /*! \brief Number of dimensions */
  int32_t ndim;
  /*! \brief The data type of the pointer*/
  DLDataType dtype;
  /*! \brief The shape of the tensor */
  int64_t* shape;
  /*! \brief strides of the tensor (in number of elements, not bytes) */
  int64_t* strides;
  /*! \brief The offset in bytes to the beginning pointer to data */
  uint64_t byte_offset;
} DLTensor;

/*!
 * \brief C Tensor object, manage memory of DLTensor.
 *
 * This data structure is intended to facilitate the borrowing of DLTensor by
 * another framework. When the borrowing framework doesn't need the tensor,
 * it should call the deleter to notify the host that the resource is no longer needed.
 */
typedef struct DLManagedTensor {
  /*! \brief DLTensor which is being memory managed */
  DLTensor dl_tensor;
  /*! \brief the context of the original host framework of DLManagedTensor */
  void* manager_ctx;
  /*! \brief Destructor - this should be called to destruct the manager_ctx */
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DLPACK_DLPACK_H_
