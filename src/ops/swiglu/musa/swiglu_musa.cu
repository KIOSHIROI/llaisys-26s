#include "swiglu_musa.hpp"

#include "../../common/dispatch_musa.cuh"

// out = up * gate / (1 + exp(-gate)) — multiply first, then divide, matching
// the CPU expression exactly.
template <typename T>
__global__ void swiglu_kernel_musa(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float g = to_float(gate[idx]);
    out[idx] = from_float<T>(to_float(up[idx]) * g / (1.0f + expf(-g)));
}

constexpr int kThreads = 256;

namespace llaisys::ops::musa {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t u, size_t v) {
    size_t numel = u * v;
    if (numel == 0) return;
    DISPATCH_DTYPE_MUSA(type, T, {
        swiglu_kernel_musa<T><<<(int)((numel + kThreads - 1) / kThreads), kThreads>>>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate),
            reinterpret_cast<const T *>(up), numel);
    });
    CHECK_LAST_MUSA();
}
} // namespace llaisys::ops::musa
