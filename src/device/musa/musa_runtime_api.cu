#include "../runtime_api.hpp"

#include <musa_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace llaisys::device::musa {

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

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    // No driver / no GPU → report 0 so the Context falls back to CPU.
    if (musaGetDeviceCount(&count) != musaSuccess) {
        return 0;
    }
    return count;
}

void setDevice(int device_id) {
    CHECK_MUSA(musaSetDevice(device_id));
}

void deviceSynchronize() {
    CHECK_MUSA(musaDeviceSynchronize());
}

llaisysStream_t createStream() {
    musaStream_t stream = nullptr;
    CHECK_MUSA(musaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    CHECK_MUSA(musaStreamDestroy(reinterpret_cast<musaStream_t>(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    CHECK_MUSA(musaStreamSynchronize(reinterpret_cast<musaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CHECK_MUSA(musaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    CHECK_MUSA(musaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    CHECK_MUSA(musaHostAlloc(&ptr, size, musaHostAllocDefault));
    return ptr;
}

void freeHost(void *ptr) {
    CHECK_MUSA(musaFreeHost(ptr));
}

static musaMemcpyKind toMusaKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return musaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return musaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return musaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return musaMemcpyDeviceToDevice;
    default: throw std::runtime_error("Invalid memcpy kind");
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CHECK_MUSA(musaMemcpy(dst, src, size, toMusaKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CHECK_MUSA(musaMemcpyAsync(dst, src, size, toMusaKind(kind),
                               reinterpret_cast<musaStream_t>(stream)));
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
} // namespace llaisys::device::musa
