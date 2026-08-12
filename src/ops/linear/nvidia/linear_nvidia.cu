#include "linear_nvidia.hpp"

#include "../../common/nvidia_common.cuh"

#include <cublas_v2.h>

#include <cstdlib>

// One thread per output element (i,k): acc starts at bias[k], then j=0..h-1 in
// ascending order — the exact accumulation order of the CPU implementation, so
// the result is bitwise identical for identical inputs.
template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
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
__global__ void add_bias_kernel(T *out, const T *bias, size_t u, size_t v) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= u * v) return;
    size_t k = idx % v;
    out[idx] = from_float<T>(to_float(out[idx]) + to_float(bias[k]));
}

template <typename T> struct cublas_dtype;
template <> struct cublas_dtype<float>         { static constexpr cudaDataType_t value = CUDA_R_32F; };
template <> struct cublas_dtype<__half>        { static constexpr cudaDataType_t value = CUDA_R_16F; };
template <> struct cublas_dtype<__nv_bfloat16> { static constexpr cudaDataType_t value = CUDA_R_16BF; };

constexpr int kThreads = 256;

// Optional cuBLAS path (perf/fallback), enabled at runtime with
// LLAISYS_LINEAR_CUBLAS=1. Default is the naive kernel above, which mirrors
// the CPU accumulation order bit-for-bit.
// (File scope: with -rdc=true the device-link step needs externally-visible
// symbols, including for callers of __global__ kernels.)
bool linearUseCublas() {
    static const bool use = [] {
        const char *e = std::getenv("LLAISYS_LINEAR_CUBLAS");
        return e != nullptr && e[0] == '1';
    }();
    return use;
}

cublasHandle_t cublasHandle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasCreate failed");
        return h;
    }();
    return handle;
}

template <typename T>
void linear_cublas(T *out, const T *in, const T *weight, const T *bias,
                   size_t u, size_t h, size_t v) {
    // out[u,v] = in[u,h] @ weight[v,h]^T
    // cuBLAS is column-major: opA=T treats in (stored row-major) as [h,u],
    // opB=N treats weight as [h,v].
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        cublasHandle(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)u, (int)v, (int)h,
        &alpha,
        in,     cublas_dtype<T>::value, (int)h,
        weight, cublas_dtype<T>::value, (int)h,
        &beta,
        out,    cublas_dtype<T>::value, (int)u,
        CUBLAS_COMPUTE_32F,               // all input types accumulate in FP32, like the CPU op
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx failed");
    size_t total = u * v;
    add_bias_kernel<T><<<(int)((total + kThreads - 1) / kThreads), kThreads>>>(out, bias, u, v);
    CHECK_LAST_CUDA();
}

template <typename T>
void linear_impl(T *out, const T *in, const T *weight, const T *bias,
                 size_t u, size_t h, size_t v) {
    if (linearUseCublas()) {
        linear_cublas(out, in, weight, bias, u, h, v);
    } else {
        size_t total = u * v;
        linear_kernel<T><<<(int)((total + kThreads - 1) / kThreads), kThreads>>>(
            out, in, weight, bias, u, h, v);
        CHECK_LAST_CUDA();
    }
}

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t u, size_t h, size_t v) {
    if (u == 0 || v == 0) return;
    DISPATCH_DTYPE(type, T, {
        linear_impl<T>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
                       reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(bias),
                       u, h, v);
    });
}
} // namespace llaisys::ops::nvidia
