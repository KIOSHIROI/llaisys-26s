#include "embedding_iluvatar.hpp"

#include "../../common/nvidia_common.cuh"

// Embeds a single token: index[0] lives on device (int64_t).
// Kernels live at file scope: with -rdc=true the device-link step needs
// externally-visible kernel symbols.
template <typename T>
__global__ void embedding_kernel_iluvatar(T *out, const int64_t *index, const T *weight, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    size_t idx = (size_t)index[0];
    out[i] = weight[idx * numel + i];
}

constexpr int kThreads = 256;

namespace llaisys::ops::iluvatar {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t numel) {
    if (numel == 0) return;
    DISPATCH_DTYPE(type, T, {
        embedding_kernel_iluvatar<T><<<(int)((numel + kThreads - 1) / kThreads), kThreads>>>(
            reinterpret_cast<T *>(out), reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const T *>(weight), numel);
    });
    CHECK_LAST_CUDA();
}
} // namespace llaisys::ops::iluvatar
