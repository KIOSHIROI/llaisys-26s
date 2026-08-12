#pragma once
#include <musa_runtime.h>
#include <musa_fp16.h>

#include <cstdio>
#include <stdexcept>

// bfloat16 lives in musa_bf16.h on newer MUSA SDKs; the type name has varied
// between __musa_bf16 and the CUDA-compatible __nv_bfloat16. Pick whatever
// exists and expose it as `musa_bfloat16` for the kernels below.
#if __has_include(<musa_bf16.h>)
#include <musa_bf16.h>
#endif

#if defined(__musa_bf16)
using musa_bfloat16 = __musa_bf16;
#define MUSA_BF16_TO_FLOAT(x) __musa_bf162float(x)
#elif defined(__nv_bfloat16)
using musa_bfloat16 = __nv_bfloat16;
#define MUSA_BF16_TO_FLOAT(x) __bfloat162float(x)
#else
#error "MUSA SDK: no bfloat16 type found (expected __musa_bf16 or __nv_bfloat16 in musa_bf16.h)"
#endif

#define CHECK_MUSA(call)                                                              \
    do {                                                                              \
        musaError_t _err_ = (call);                                                   \
        if (_err_ != musaSuccess) {                                                   \
            std::fprintf(stderr, "[ERROR] MUSA %s: %s at %s:%d\n",                    \
                         musaGetErrorName(_err_), musaGetErrorString(_err_),          \
                         __FILE__, __LINE__);                                         \
            throw std::runtime_error(musaGetErrorString(_err_));                      \
        }                                                                             \
    } while (0)

// Call after a kernel launch: catches launch-time errors (invalid block counts etc).
// Async execution-time errors surface on the next musaMemcpy / deviceSynchronize.
#define CHECK_LAST_MUSA() CHECK_MUSA(musaGetLastError())

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(musa_bfloat16 x) { return MUSA_BF16_TO_FLOAT(x); }

// float / __half / bf16 can all be constructed from float directly (RNE),
// matching the CPU round-trip through llaisys::utils::cast (within 1 ulp).
template <typename T>
__device__ __forceinline__ T from_float(float x) { return T(x); }
