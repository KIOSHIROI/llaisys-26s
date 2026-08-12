#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <vector>

// The concrete definition of the opaque `struct LlaisysQwen2Model` declared in
// include/llaisys/models/qwen2.h.
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int device_id;

    // Weights, one tensor per layer for the vector fields.
    llaisys::tensor_t in_embed, out_embed, out_norm_w;
    std::vector<llaisys::tensor_t> attn_norm_w, attn_q_w, attn_q_b;
    std::vector<llaisys::tensor_t> attn_k_w, attn_k_b, attn_v_w, attn_v_b, attn_o_w;
    std::vector<llaisys::tensor_t> mlp_norm_w, mlp_gate_w, mlp_up_w, mlp_down_w;

    // KV cache: [maxseq, nkvh, dh] per layer.
    std::vector<llaisys::tensor_t> k_cache, v_cache;

    // Workspaces (pre-allocated at [maxseq, ...], sliced per infer call).
    llaisys::tensor_t hidden, normed, q, k, v, attn_val, attn_proj, gate, up, act;
    llaisys::tensor_t pos_ids, tok_idx;
    llaisys::tensor_t zero_bias_hs, zero_bias_di, zero_bias_voc;

    size_t total_len;

    LlaisysQwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);
    ~LlaisysQwen2Model() = default;

    // Forward pass over `ntoken` new tokens and return the argmax next token.
    int64_t infer(const int64_t *token_ids, size_t ntoken);

    // Build the C-API weights view (owned by this model, freed on destroy).
    LlaisysQwen2Weights *weights() const;
};
