#include "rope_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
#include <vector>
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seqlen, size_t nkvhead, size_t dim) {
    for (size_t i = 0; i < seqlen; ++ i) {
        int64_t pos = pos_ids[i];
        for (size_t j = 0; j < nkvhead; ++ j) {
            size_t base = (i * nkvhead + j) * dim;

            for(size_t k = 0; k < dim / 2; ++ k) {
                float freq = static_cast<float>(pos) / std::pow(theta, static_cast<float>(2 * k) / dim);
                float x1 = llaisys::utils::cast<float>(in[base + k]);
                float x2 = llaisys::utils::cast<float>(in[base + dim/2 + k]);
                float cos = std::cos(freq);
                float sin = std::sin(freq);
                out[base + k] = llaisys::utils::cast<T>(cos * x1 - sin * x2);
                out[base + dim/2 + k] = llaisys::utils::cast<T>(sin * x1 + cos * x2);
            }
        }
    }

}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seq_len, size_t nkvhead, size_t dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), 
                    theta, seq_len, nkvhead, dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, nkvhead, dim);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, nkvhead, dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
