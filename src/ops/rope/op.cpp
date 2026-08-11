#include "op.hpp"
#include "cpu/rope_cpu.hpp"
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, in->dtype(), in->shape()[0], in->shape()[1], in->shape()[2]);
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, in->dtype(), in->shape()[0], in->shape()[1], in->shape()[2]);
    #endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
}
} // namespace llaisys::ops
