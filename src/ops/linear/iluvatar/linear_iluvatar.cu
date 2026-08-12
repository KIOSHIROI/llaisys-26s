#include "linear_iluvatar.hpp"

#include "../../common/nvidia_common.cuh"

// Iluvatar CoreX provides a CUDA-compatible toolchain and headers, so the
// shared CUDA common header applies as-is.
//
// One thread per output element (i,k): acc starts at bias[k], then j=0..h-1 in
// ascending order — the exact accumulation order of the CPU implementation, so
// the result is bitwise identical for identical inputs.
// (No cuBLAS path here: the CoreX cuBLAS shim is not required for the tests.)
template <typename T>
__global__ void linear_kernel_iluvatar(T *out, const T *in, const T *weight, const T *bias,
                                       size_t u, size_t h, size_t v) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= u * v) return;
    size_t i = idx / v, k = idx % v;
    float acc = to_float(bias[k]);
    const T *in_row = in + i * h;
    const T *w_row = weight + k * h;
    for (size_t j = 0; j < h; ++j)
        acc += to_float(in_row[j]) * to_float(w_row[j]);
    out[idx] = from_float<T>(acc);
}

constexpr int kThreads = 256;

namespace llaisys::ops::iluvatar {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t u, size_t h, size_t v) {
    if (u == 0 || v == 0) return;
    DISPATCH_DTYPE(type, T, {
        size_t total = u * v;
        linear_kernel_iluvatar<T><<<(int)((total + kThreads - 1) / kThreads), kThreads>>>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
            reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(bias),
            u, h, v);
        CHECK_LAST_CUDA();
    });
}
} // namespace llaisys::ops::iluvatar
