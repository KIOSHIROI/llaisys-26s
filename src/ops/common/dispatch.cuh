#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../core/llaisys_core.hpp"

#define DISPATCH_DTYPE(TYPE, T, ...) \
    switch (TYPE) { \
    case LLAISYS_DTYPE_F32: { \
        using T = float; \
        __VA_ARGS__; \
        break; \
    } \
    case LLAISYS_DTYPE_F16: { \
        using T = __half; \
        __VA_ARGS__; \
        break; \
    } \
    case LLAISYS_DTYPE_BF16: { \
        using T = __nv_bfloat16; \
        __VA_ARGS__; \
        break; \
    } \
    default: \
        EXCEPTION_UNSUPPORTED_DATATYPE(TYPE); \
    }
