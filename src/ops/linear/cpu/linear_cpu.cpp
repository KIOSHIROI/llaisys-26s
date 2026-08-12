#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, const size_t u, const size_t h, const size_t v) {
    #pragma omp parallel for
    for(size_t i = 0; i < u; ++ i) {
        for (size_t k = 0; k < v; ++ k) {
            float value = llaisys::utils::cast<float>(bias[k]);
            for(size_t j = 0; j < h; ++ j) {
                value += llaisys::utils::cast<float>(in[i * h + j]) * llaisys::utils::cast<float>(weight[k * h + j]);
            }
            out[i * v + k] = llaisys::utils::cast<T>(value);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, const size_t u, const size_t h, const size_t v) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), 
                    reinterpret_cast<const float *> (bias), u, h, v);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), u, h ,v);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), u, h, v);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
