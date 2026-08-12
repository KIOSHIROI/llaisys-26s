#include "linear_musa.hpp"

#include "../../common/dispatch_musa.cuh"

#include <cstdlib>

// One thread per output element (i,k): acc starts at bias[k], then j=0..h-1 in
// ascending order — the exact accumulation order of the CPU implementation, so
// the result is bitwise identical for identical inputs.
template <typename T>
__global__ void linear_kernel_musa(T *out, const T *in, const T *weight, const T *bias,
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

template <typename T>
__global__ void add_bias_kernel_musa(T *out, const T *bias, size_t u, size_t v) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= u * v) return;
    size_t k = idx % v;
    out[idx] = from_float<T>(to_float(out[idx]) + to_float(bias[k]));
}

// Optional mublas path (perf/fallback), enabled at runtime with
// LLAISYS_LINEAR_MUBLAS=1. Default is the naive kernel above, which mirrors
// the CPU accumulation order bit-for-bit. Compiles out entirely when the
// mublas headers are not installed.
#if __has_include(<mublas_v2.h>)
#include <mublas_v2.h>

template <typename T> struct mublas_dtype;
template <> struct mublas_dtype<float>         { static constexpr mublasDataType_t value = MUSA_R_32F; };
template <> struct mublas_dtype<__half>        { static constexpr mublasDataType_t value = MUSA_R_16F; };
template <> struct mublas_dtype<musa_bfloat16> { static constexpr mublasDataType_t value = MUSA_R_16BF; };

constexpr int kThreads = 256;

bool linearUseMublas() {
    static const bool use = [] {
        const char *e = std::getenv("LLAISYS_LINEAR_MUBLAS");
        return e != nullptr && e[0] == '1';
    }();
    return use;
}

mublasHandle_t mublasHandle() {
    static mublasHandle_t handle = [] {
        mublasHandle_t h = nullptr;
        if (mublasCreate(&h) != MUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("mublasCreate failed");
        return h;
    }();
    return handle;
}

template <typename T>
void linear_mublas(T *out, const T *in, const T *weight, const T *bias,
                   size_t u, size_t h, size_t v) {
    // out[u,v] = in[u,h] @ weight[v,h]^T
    // mublas is column-major: opA=T treats in (stored row-major) as [h,u],
    // opB=N treats weight as [h,v].
    const float alpha = 1.0f, beta = 0.0f;
    mublasStatus_t st = mublasGemmEx(
        mublasHandle(),
        MUBLAS_OP_T, MUBLAS_OP_N,
        (int)u, (int)v, (int)h,
        &alpha,
        in,     mublas_dtype<T>::value, (int)h,
        weight, mublas_dtype<T>::value, (int)h,
        &beta,
        out,    mublas_dtype<T>::value, (int)u,
        MUBLAS_COMPUTE_32F,               // all input types accumulate in FP32, like the CPU op
        MUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != MUBLAS_STATUS_SUCCESS) throw std::runtime_error("mublasGemmEx failed");
    size_t total = u * v;
    add_bias_kernel_musa<T><<<(int)((total + kThreads - 1) / kThreads), kThreads>>>(out, bias, u, v);
    CHECK_LAST_MUSA();
}
#endif // __has_include(<mublas_v2.h>)

template <typename T>
void linear_impl(T *out, const T *in, const T *weight, const T *bias,
                 size_t u, size_t h, size_t v) {
#if __has_include(<mublas_v2.h>)
    if (linearUseMublas()) {
        linear_mublas(out, in, weight, bias, u, h, v);
        return;
    }
#endif
    size_t total = u * v;
    linear_kernel_musa<T><<<(int)((total + kThreads - 1) / kThreads), kThreads>>>(
        out, in, weight, bias, u, h, v);
    CHECK_LAST_MUSA();
}

namespace llaisys::ops::musa {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t u, size_t h, size_t v) {
    if (u == 0 || v == 0) return;
    DISPATCH_DTYPE_MUSA(type, T, {
        linear_impl<T>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
                       reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(bias),
                       u, h, v);
    });
}
} // namespace llaisys::ops::musa
