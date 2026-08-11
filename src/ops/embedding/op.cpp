#include "op.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), weight->shape()[1]);
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::embedding(out->data(), index->data(), weight->data(), weight->dtype(), weight->shape()[1]);
    #endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
}
} // namespace llaisys::ops
