#include "add_iluvatar.hpp"

#include "../../common/nvidia_common.cuh"

template <typename T>
__global__ void add_kernel_iluvatar(T* c, const T* a, const T* b, size_t numel) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < numel) {
        c[i] = a[i] + b[i];
    }
}

namespace llaisys::ops::iluvatar {
void add(std::byte* c, const std::byte* a, const std::byte* b, llaisysDataType_t type, size_t numel) {
    if (numel == 0) return;
    constexpr int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    DISPATCH_DTYPE(type, T, {
        add_kernel_iluvatar<T><<<blocks, threads>>>(
            reinterpret_cast<T*>(c),
            reinterpret_cast<const T*>(a),
            reinterpret_cast<const T*>(b),
            numel);
    });
    CHECK_LAST_CUDA();
}
} // namespace llaisys::ops::iluvatar
