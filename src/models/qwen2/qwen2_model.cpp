#include "qwen2_model.hpp"

#include "../../llaisys/llaisys_tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include "../../core/context/context.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

// Zero-initialize a tensor (naive allocator does not zero memory).
// Device tensors cannot be memset'ed from the host; build a zero host
// buffer and upload it instead.
void zeroTensor(llaisys::tensor_t t) {
    if (t->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memset(t->data(), 0, t->numel() * t->elementSize());
    } else {
        std::vector<std::byte> zeros(t->numel() * t->elementSize(), std::byte{0});
        t->load(zeros.data());
    }
}

} // namespace

LlaisysQwen2Model::LlaisysQwen2Model(const LlaisysQwen2Meta &meta_, llaisysDeviceType_t device_, int device_id_)
    : meta(meta_), device(device_), device_id(device_id_), total_len(0) {
    const size_t nlayer = meta.nlayer, hs = meta.hs, nh = meta.nh, nkvh = meta.nkvh, dh = meta.dh;
    const size_t di = meta.di, voc = meta.voc, maxseq = meta.maxseq;
    const llaisysDataType_t dtype = meta.dtype;

    auto make = [&](const std::vector<size_t> &shape) {
        return llaisys::Tensor::create(shape, dtype, device, device_id);
    };

    // Weights
    in_embed = make({voc, hs});
    out_embed = make({voc, hs});
    out_norm_w = make({hs});
    for (size_t i = 0; i < nlayer; ++i) {
        attn_norm_w.push_back(make({hs}));
        attn_q_w.push_back(make({nh * dh, hs}));
        attn_q_b.push_back(make({nh * dh}));
        attn_k_w.push_back(make({nkvh * dh, hs}));
        attn_k_b.push_back(make({nkvh * dh}));
        attn_v_w.push_back(make({nkvh * dh, hs}));
        attn_v_b.push_back(make({nkvh * dh}));
        attn_o_w.push_back(make({hs, nh * dh}));
        mlp_norm_w.push_back(make({hs}));
        mlp_gate_w.push_back(make({di, hs}));
        mlp_up_w.push_back(make({di, hs}));
        mlp_down_w.push_back(make({hs, di}));
    }

    // KV cache
    for (size_t i = 0; i < nlayer; ++i) {
        k_cache.push_back(make({maxseq, nkvh, dh}));
        zeroTensor(k_cache.back());
        v_cache.push_back(make({maxseq, nkvh, dh}));
        zeroTensor(v_cache.back());
    }

    // Workspaces
    hidden = make({maxseq, hs});
    normed = make({maxseq, hs});
    q = make({maxseq, nh * dh});
    k = make({maxseq, nkvh * dh});
    v = make({maxseq, nkvh * dh});
    attn_val = make({maxseq, nh * dh});
    attn_proj = make({maxseq, hs});
    gate = make({maxseq, di});
    up = make({maxseq, di});
    act = make({maxseq, di});
    pos_ids = llaisys::Tensor::create({maxseq}, LLAISYS_DTYPE_I64, device, device_id);
    tok_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);

    // Zero biases for the linear layers without bias (o_proj, mlp, lm_head).
    zero_bias_hs = make({hs});
    zero_bias_di = make({di});
    zero_bias_voc = make({voc});
    zeroTensor(zero_bias_hs);
    zeroTensor(zero_bias_di);
    zeroTensor(zero_bias_voc);
}

int64_t LlaisysQwen2Model::infer(const int64_t *token_ids, size_t seqlen) {
    const size_t past = total_len;
    const size_t total = past + seqlen;
    ASSERT(total <= meta.maxseq, "KV cache overflow");

    const size_t nlayer = meta.nlayer, hs = meta.hs, nh = meta.nh, nkvh = meta.nkvh, dh = meta.dh;
    const size_t voc = meta.voc;

    // Position ids of the new tokens: [past, past+seqlen).
    // Upload from host: the device copy cannot be written directly.
    std::vector<int64_t> pos_host(seqlen);
    for (size_t i = 0; i < seqlen; ++i) {
        pos_host[i] = static_cast<int64_t>(past + i);
    }
    pos_ids->load(pos_host.data());

    // Embedding (one token at a time) into hidden rows [0, seqlen).
    for (size_t i = 0; i < seqlen; ++i) {
        int64_t tok = token_ids[i];
        tok_idx->load(&tok);
        llaisys::tensor_t row = hidden->slice(0, i, i + 1)->view({hs});
        llaisys::ops::embedding(row, tok_idx, in_embed);
    }

    llaisys::tensor_t hid = hidden->slice(0, 0, seqlen);        // [seqlen, hs]
    llaisys::tensor_t nor = normed->slice(0, 0, seqlen);        // [seqlen, hs]
    llaisys::tensor_t pos = pos_ids->slice(0, 0, seqlen);       // [seqlen]
    llaisys::tensor_t gg = gate->slice(0, 0, seqlen);           // [seqlen, di]
    llaisys::tensor_t uu = up->slice(0, 0, seqlen);             // [seqlen, di]
    llaisys::tensor_t aa = act->slice(0, 0, seqlen);            // [seqlen, di]
    llaisys::tensor_t qq = q->slice(0, 0, seqlen)->view({seqlen, nh, dh});
    llaisys::tensor_t kk = k->slice(0, 0, seqlen)->view({seqlen, nkvh, dh});
    llaisys::tensor_t vv = v->slice(0, 0, seqlen)->view({seqlen, nkvh, dh});
    llaisys::tensor_t attn = attn_val->slice(0, 0, seqlen)->view({seqlen, nh, dh});
    llaisys::tensor_t attn_o = attn_proj->slice(0, 0, seqlen); // [seqlen, hs]

    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    for (size_t L = 0; L < nlayer; ++L) {
        // Attention
        llaisys::ops::rms_norm(nor, hid, attn_norm_w[L], meta.epsilon);
        llaisys::ops::linear(q->slice(0, 0, seqlen), nor, attn_q_w[L], attn_q_b[L]);
        llaisys::ops::rope(qq, qq, pos, meta.theta); // in-place: safe

        llaisys::ops::linear(k->slice(0, 0, seqlen), nor, attn_k_w[L], attn_k_b[L]);
        llaisys::ops::rope(k_cache[L]->slice(0, past, total), kk, pos, meta.theta);

        llaisys::ops::linear(v_cache[L]->slice(0, past, total), nor, attn_v_w[L], attn_v_b[L]);

        llaisys::ops::self_attention(attn, qq, k_cache[L]->slice(0, 0, total), v_cache[L]->slice(0, 0, total), scale);
        llaisys::ops::linear(attn_o, attn->view({seqlen, nh * dh}), attn_o_w[L], zero_bias_hs);
        llaisys::ops::add(hid, hid, attn_o);

        // MLP
        llaisys::ops::rms_norm(nor, hid, mlp_norm_w[L], meta.epsilon);
        llaisys::ops::linear(gg, nor, mlp_gate_w[L], zero_bias_di);
        llaisys::ops::linear(uu, nor, mlp_up_w[L], zero_bias_di);
        llaisys::ops::swiglu(aa, gg, uu);
        llaisys::ops::linear(nor, aa, mlp_down_w[L], zero_bias_hs);
        llaisys::ops::add(hid, hid, nor);
    }

    // Final norm + lm_head
    llaisys::ops::rms_norm(nor, hid, out_norm_w, meta.epsilon);
    llaisys::tensor_t logits = llaisys::Tensor::create({seqlen, voc}, meta.dtype, device, device_id);
    llaisys::ops::linear(logits, nor, out_embed, zero_bias_voc);

    // Argmax over the last position.
    llaisys::tensor_t max_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
    llaisys::tensor_t max_val = llaisys::Tensor::create({1}, meta.dtype, device, device_id);
    llaisys::ops::argmax(max_idx, max_val, logits->slice(0, seqlen - 1, seqlen));

    // Copy the argmax result back to the host.
    int64_t result = 0;
    llaisys::core::context().setDevice(device, device_id);
    llaisys::core::context().runtime().api()->memcpy_sync(
        &result, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H
    );

    total_len = total;
    return result;
}

LlaisysQwen2Weights *LlaisysQwen2Model::weights() const {
    auto *w = new LlaisysQwen2Weights;
    w->in_embed = new LlaisysTensor{in_embed};
    w->out_embed = new LlaisysTensor{out_embed};
    w->out_norm_w = new LlaisysTensor{out_norm_w};

    auto fill = [&](llaisysTensor_t *&arr, const std::vector<llaisys::tensor_t> &vec) {
        arr = new llaisysTensor_t[vec.size()];
        for (size_t i = 0; i < vec.size(); ++i) {
            arr[i] = new LlaisysTensor{vec[i]};
        }
    };
    fill(w->attn_norm_w, attn_norm_w);
    fill(w->attn_q_w, attn_q_w);
    fill(w->attn_q_b, attn_q_b);
    fill(w->attn_k_w, attn_k_w);
    fill(w->attn_k_b, attn_k_b);
    fill(w->attn_v_w, attn_v_w);
    fill(w->attn_v_b, attn_v_b);
    fill(w->attn_o_w, attn_o_w);
    fill(w->mlp_norm_w, mlp_norm_w);
    fill(w->mlp_gate_w, mlp_gate_w);
    fill(w->mlp_up_w, mlp_up_w);
    fill(w->mlp_down_w, mlp_down_w);
    return w;
}
