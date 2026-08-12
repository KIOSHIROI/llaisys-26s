#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <stdexcept>

#include "dispatch.cuh"   // DISPATCH_DTYPE (F32/F16/BF16 -> float/__half/__nv_bfloat16)

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t _err_ = (call);                                                   \
        if (_err_ != cudaSuccess) {                                                   \
            std::fprintf(stderr, "[ERROR] CUDA %s: %s at %s:%d\n",                    \
                         cudaGetErrorName(_err_), cudaGetErrorString(_err_),          \
                         __FILE__, __LINE__);                                         \
            throw std::runtime_error(cudaGetErrorString(_err_));                      \
        }                                                                             \
    } while (0)

// Call after a kernel launch: catches launch-time errors (invalid block counts etc).
// Async execution-time errors surface on the next cudaMemcpy / deviceSynchronize.
#define CHECK_LAST_CUDA() CHECK_CUDA(cudaGetLastError())

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

// float / __half / __nv_bfloat16 can all be constructed from float directly (RNE),
// matching the CPU round-trip through llaisys::utils::cast (within 1 ulp).
template <typename T>
__device__ __forceinline__ T from_float(float x) { return T(x); }
