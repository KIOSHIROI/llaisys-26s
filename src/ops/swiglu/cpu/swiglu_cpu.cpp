#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t u, size_t v) {
    for(size_t i = 0; i < u; ++ i) {
        for(size_t j = 0; j < v; ++ j) {
            size_t idx = i * v + j;
            out[idx] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(up[idx]) * llaisys::utils::cast<float>(gate[idx])
                        / (1 + std::exp(-llaisys::utils::cast<float>(gate[idx]))));
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,llaisysDataType_t type, size_t u, size_t v) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), 
                    u, v);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
                    reinterpret_cast<const llaisys::bf16_t *>(up), u, v);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
                    reinterpret_cast<const llaisys::fp16_t *>(up), u, v);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
