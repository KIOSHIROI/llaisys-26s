#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::iluvatar {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t u, size_t h, size_t v);
}
