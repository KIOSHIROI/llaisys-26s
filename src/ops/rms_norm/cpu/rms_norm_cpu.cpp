#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t u, size_t v) {
    for(size_t i = 0; i < u; i++) {
        float rms = 0.0f;
        for(size_t j = 0; j < v; j ++) {
            float x = llaisys::utils::cast<float>(in[i * v + j]);
            rms += x * x;
        }
        rms = std::sqrt(rms / v + eps);

        for(size_t j = 0; j < v; j ++) {
            float x = llaisys::utils::cast<float>(in[i * v + j]);
            float w = llaisys::utils::cast<float>(weight[j]);

            out[i * v + j] = llaisys::utils::cast<T>(x * w / rms);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t u, size_t v) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), 
                    eps, u, v);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), eps, u, v);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), eps, u, v);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
