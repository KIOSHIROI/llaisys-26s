#include "rms_norm_musa.hpp"

#include "../../common/dispatch_musa.cuh"

// Pass 1: per-row sum of squares, ascending j — the exact order of the CPU op.
template <typename T>
__global__ void rms_norm_sumsq_kernel_musa(const T *in, float *sumsq, size_t u, size_t v) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= u) return;
    float acc = 0.0f;
    const T *row = in + i * v;
    for (size_t j = 0; j < v; ++j) {
        float x = to_float(row[j]);
        acc += x * x;
    }
    sumsq[i] = acc;
}

// Pass 2: apply. rms = sqrt(sumsq/v + eps) — divide first, then add, then sqrt,
// matching the CPU order; division (not rsqrt) matches too.
template <typename T>
__global__ void rms_norm_apply_kernel_musa(T *out, const T *in, const T *weight,
                                           const float *sumsq, float eps, size_t u, size_t v) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= u * v) return;
    size_t i = idx / v, j = idx % v;
    float rms = sqrtf(sumsq[i] / (float)v + eps);
    out[idx] = from_float<T>(to_float(in[idx]) * to_float(weight[j]) / rms);
}

constexpr int kThreads = 256;

namespace llaisys::ops::musa {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t type, size_t u, size_t v) {
    if (u == 0 || v == 0) return;
    float *d_sumsq = nullptr;
    CHECK_MUSA(musaMalloc(&d_sumsq, u * sizeof(float)));
    DISPATCH_DTYPE_MUSA(type, T, {
        rms_norm_sumsq_kernel_musa<T><<<(int)((u + kThreads - 1) / kThreads), kThreads>>>(
            reinterpret_cast<const T *>(in), d_sumsq, u, v);
        rms_norm_apply_kernel_musa<T><<<(int)((u * v + kThreads - 1) / kThreads), kThreads>>>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
            reinterpret_cast<const T *>(weight), d_sumsq, eps, u, v);
    });
    CHECK_LAST_MUSA();
    CHECK_MUSA(musaFree(d_sumsq));
}
} // namespace llaisys::ops::musa
