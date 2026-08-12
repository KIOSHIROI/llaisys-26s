#include "argmax_musa.hpp"

#include "../../common/dispatch_musa.cuh"

// Single thread — mirrors the CPU scan exactly: strictly-greater comparison,
// first (lowest index) wins, max_val is stored in the element type.
template <typename T>
__global__ void argmax_kernel_musa(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    size_t idx = 0;
    float value = to_float(vals[0]);
    for (size_t i = 1; i < numel; ++i) {
        float cur = to_float(vals[i]);
        if (cur > value) {
            idx = i;
            value = cur;
        }
    }
    *max_idx = (int64_t)idx;
    *max_val = from_float<T>(value);
}

namespace llaisys::ops::musa {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
    if (numel == 0) throw std::runtime_error("vals is empty");  // mirror the CPU op
    DISPATCH_DTYPE_MUSA(type, T, {
        argmax_kernel_musa<T><<<1, 1>>>(reinterpret_cast<int64_t *>(max_idx),
                                        reinterpret_cast<T *>(max_val),
                                        reinterpret_cast<const T *>(vals), numel);
    });
    CHECK_LAST_MUSA();
}
} // namespace llaisys::ops::musa
