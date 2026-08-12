#include "op.hpp"
#include "cpu/self_attention_cpu.cpp"
#include "nvidia/self_attention_nvidia.hpp"
#include "iluvatar/self_attention_iluvatar.hpp"
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, q->dtype(), q->shape()[0], k->shape()[0], q->shape()[1], k->shape()[1], q->shape()[2], v->shape()[2]);
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, q->dtype(), q->shape()[0], k->shape()[0], q->shape()[1], k->shape()[1], q->shape()[2], v->shape()[2]);
    #endif
    #ifdef ENABLE_ILUVATAR_API
        case LLAISYS_DEVICE_ILUVATAR:
            return iluvatar::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, q->dtype(), q->shape()[0], k->shape()[0], q->shape()[1], k->shape()[1], q->shape()[2], v->shape()[2]);
    #endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
}
} // namespace llaisys::ops
