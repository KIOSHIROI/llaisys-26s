#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace llaisys::device::iluvatar {

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

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    // No driver / no GPU → report 0 so the Context falls back to CPU.
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
}

void setDevice(int device_id) {
    CHECK_CUDA(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    CHECK_CUDA(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    CHECK_CUDA(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    CHECK_CUDA(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    CHECK_CUDA(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    CHECK_CUDA(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
    return ptr;
}

void freeHost(void *ptr) {
    CHECK_CUDA(cudaFreeHost(ptr));
}

static cudaMemcpyKind toCudaKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return cudaMemcpyDeviceToDevice;
    default: throw std::runtime_error("Invalid memcpy kind");
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CHECK_CUDA(cudaMemcpy(dst, src, size, toCudaKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CHECK_CUDA(cudaMemcpyAsync(dst, src, size, toCudaKind(kind),
                               reinterpret_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::iluvatar
