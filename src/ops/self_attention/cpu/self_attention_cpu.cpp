#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
#include <vector>
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,  float scale, 
    size_t seqlen, size_t total_len,  size_t nhead, size_t nkvhead, size_t d, size_t dv) {

    const size_t past_len = total_len - seqlen;
    const size_t group_size = nhead / nkvhead;

    
    std::vector<float> scores(total_len);

    for(size_t i = 0; i < seqlen; ++ i) {
        for(size_t h = 0; h < nhead; ++ h) {
            const size_t kv_h = h / group_size;
            for(size_t j = 0; j < total_len; ++ j) {
                float score = 0.0f;
                for (size_t x = 0; x < d; ++ x)
                    score += llaisys::utils::cast<float>(q[(i * nhead + h) * d + x]) * llaisys::utils::cast<float>(k[(j * nkvhead + kv_h) * d + x]);
                score *= scale;
                if (j > past_len + i) 
                    score = -INFINITY;
                scores[j] = score;
            }
            float max_score = -INFINITY;

            for(size_t j = 0; j < total_len; ++ j) {
                max_score = std::max(max_score, scores[j]);
            }

            float sum = 0.0f;
            for(size_t j = 0; j < total_len; j++) {
                scores[j] = std::exp(scores[j] - max_score);
                sum += scores[j];
            }

            for(size_t j = 0; j < total_len; j++) {
                scores[j] /= sum;
            }

            for(size_t r = 0; r < dv; ++ r) {
                float attn_output = 0.0f;
                for (size_t j = 0; j < total_len; ++ j) {
                    attn_output += scores[j] * llaisys::utils::cast<float>(v[(j * nkvhead + kv_h) * dv + r]);
                }
                attn_val[(i * nhead + h) * dv + r] = llaisys::utils::cast<T>(attn_output);
            }
        }
            
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t type, 
    size_t seqlen, size_t total_len,  size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), 
                    scale, seqlen, total_len, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k),
                    reinterpret_cast<const llaisys::bf16_t *>(v), scale, seqlen, total_len, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k),
                    reinterpret_cast<const llaisys::fp16_t *>(v), scale, seqlen, total_len, nhead, nkvhead, d, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
