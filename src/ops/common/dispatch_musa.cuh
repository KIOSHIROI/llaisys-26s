#pragma once

#include "musa_common.cuh"

#include "../../core/llaisys_core.hpp"

// Same dtype dispatch as dispatch.cuh, but with MUSA scalar types.
#define DISPATCH_DTYPE_MUSA(TYPE, T, ...) \
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
        using T = musa_bfloat16; \
        __VA_ARGS__; \
        break; \
    } \
    default: \
        EXCEPTION_UNSUPPORTED_DATATYPE(TYPE); \
    }
