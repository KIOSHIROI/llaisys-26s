#include "rope_musa.hpp"

#include "../../common/dispatch_musa.cuh"

// GPT-J half-split rotary. Each thread handles exactly its own (k, k+half)
// element pair, reading both before writing either — so in-place (out == in)
// is safe. Frequency and cos/sin mirror the CPU order:
//   freq = pos / pow(theta, 2k/dim); cos, sin; out = c*x1 - s*x2 / s*x1 + c*x2
template <typename T>
__global__ void rope_kernel_musa(T *out, const T *in, const int64_t *pos_ids, float theta,
                                 size_t seqlen, size_t nkvhead, size_t dim) {
    size_t half = dim / 2;
    size_t total = seqlen * nkvhead * half;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t k = idx % half;
    size_t head = (idx / half) % nkvhead;
    size_t i = idx / (half * nkvhead);
    int64_t pos = pos_ids[i];
    float freq = (float)pos / powf(theta, (float)(2 * k) / (float)dim);
    size_t base = (i * nkvhead + head) * dim;
    float x1 = to_float(in[base + k]);
    float x2 = to_float(in[base + half + k]);
    float c = cosf(freq), s = sinf(freq);
    out[base + k]        = from_float<T>(c * x1 - s * x2);
    out[base + half + k] = from_float<T>(s * x1 + c * x2);
}

constexpr int kThreads = 256;

namespace llaisys::ops::musa {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t type, size_t seqlen, size_t nkvhead, size_t dim) {
    if (seqlen == 0 || nkvhead == 0 || dim == 0) return;
    size_t total = seqlen * nkvhead * (dim / 2);
    DISPATCH_DTYPE_MUSA(type, T, {
        rope_kernel_musa<T><<<(int)((total + kThreads - 1) / kThreads), kThreads>>>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
            reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nkvhead, dim);
    });
    CHECK_LAST_MUSA();
}
} // namespace llaisys::ops::musa
