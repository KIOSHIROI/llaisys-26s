#include "op.hpp"

#include "cpu/rms_norm_cpu.hpp"
#include "nvidia/rms_norm_nvidia.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, in->dtype(), in->shape()[0], in->shape()[1]);
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, in->dtype(), in->shape()[0], in->shape()[1]);
    #endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
}
} // namespace llaisys::ops
