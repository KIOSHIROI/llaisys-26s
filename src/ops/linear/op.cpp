#include "op.hpp"
#include "cpu/linear_cpu.hpp"
#include "nvidia/linear_nvidia.hpp"
#include "musa/linear_musa.hpp"
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), weight->dtype(), in->shape()[0], weight->shape()[1], weight->shape()[0]);
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::linear(out->data(), in->data(), weight->data(), bias->data(), weight->dtype(), in->shape()[0], weight->shape()[1], weight->shape()[0]);
    #endif
    #ifdef ENABLE_MUSA_API
        case LLAISYS_DEVICE_MUSA:
            return musa::linear(out->data(), in->data(), weight->data(), bias->data(), weight->dtype(), in->shape()[0], weight->shape()[1], weight->shape()[0]);
    #endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
}
} // namespace llaisys::ops
