#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t numel) {
    size_t idx = static_cast<size_t> (*index);
    for (size_t i = 0; i < numel; ++ i) {
        out[i] = weight[idx * numel + i];
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), numel);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), numel);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
