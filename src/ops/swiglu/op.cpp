#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"
#include "nvidia/swiglu_nvidia.hpp"
namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::swiglu(out->data(), gate->data(), up->data(), up->dtype(), up->shape()[0], up->shape()[1]);
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::swiglu(out->data(), gate->data(), up->data(), up->dtype(), up->shape()[0], up->shape()[1]);
    #endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
}
} // namespace llaisys::ops
