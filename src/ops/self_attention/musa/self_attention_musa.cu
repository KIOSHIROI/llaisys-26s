#include "self_attention_musa.hpp"

#include "../../common/dispatch_musa.cuh"

// score(j) for query row i: dot(q, k[j]) in ascending x, then * scale, then the
// causal mask (-inf for j > past_len + i) — same order as the CPU op.
template <typename T>
__device__ __forceinline__ float attention_score_musa(const T *q_row, const T *k, size_t nkvhead,
                                                      size_t kv_h, size_t j, size_t d,
                                                      float scale, size_t past_limit) {
    float sc = 0.0f;
    const T *k_row = k + (j * nkvhead + kv_h) * d;
    for (size_t x = 0; x < d; ++x)
        sc += to_float(q_row[x]) * to_float(k_row[x]);
    sc *= scale;
    if (j > past_limit) sc = -INFINITY;
    return sc;
}

// One thread per (i, h) runs the full three-pass softmax serially — the exact
// CPU semantics. score(j) is recomputed per pass (no shared scratch: the CPU's
// `scores` array would race between threads). Identical arithmetic order.
template <typename T>
__global__ void self_attention_kernel_musa(T *attn_val, const T *q, const T *k, const T *v,
                                           float scale, size_t seqlen, size_t total_len,
                                           size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seqlen * nhead) return;
    size_t h = idx % nhead;
    size_t i = idx / nhead;
    size_t kv_h = h / (nhead / nkvhead);
    size_t past_len = total_len - seqlen;
    size_t past_limit = past_len + i;
    const T *q_row = q + (i * nhead + h) * d;

    // pass 1: max over j (ascending, fmaxf == std::max)
    float max_score = -INFINITY;
    for (size_t j = 0; j < total_len; ++j)
        max_score = fmaxf(max_score, attention_score_musa(q_row, k, nkvhead, kv_h, j, d, scale, past_limit));

    // pass 2: sum of exp (ascending)
    float sum = 0.0f;
    for (size_t j = 0; j < total_len; ++j)
        sum += expf(attention_score_musa(q_row, k, nkvhead, kv_h, j, d, scale, past_limit) - max_score);

    // pass 3: output, per r a full sweep over j (ascending)
    T *out_row = attn_val + (i * nhead + h) * dv;
    for (size_t r = 0; r < dv; ++r) {
        float acc = 0.0f;
        for (size_t j = 0; j < total_len; ++j)
            acc += expf(attention_score_musa(q_row, k, nkvhead, kv_h, j, d, scale, past_limit) - max_score)
                   / sum * to_float(v[(j * nkvhead + kv_h) * dv + r]);
        out_row[r] = from_float<T>(acc);
    }
}

constexpr int kThreads = 256;

namespace llaisys::ops::musa {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t type, size_t seqlen, size_t total_len,
                    size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    if (seqlen == 0 || nhead == 0) return;
    size_t total = seqlen * nhead;
    DISPATCH_DTYPE_MUSA(type, T, {
        self_attention_kernel_musa<T><<<(int)((total + kThreads - 1) / kThreads), kThreads>>>(
            reinterpret_cast<T *>(attn_val), reinterpret_cast<const T *>(q),
            reinterpret_cast<const T *>(k), reinterpret_cast<const T *>(v),
            scale, seqlen, total_len, nhead, nkvhead, d, dv);
    });
    CHECK_LAST_MUSA();
}
} // namespace llaisys::ops::musa
