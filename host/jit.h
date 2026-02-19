#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "config.h"

#if JIT

std::string jit_compile(const std::vector<uint8_t>& rpn_ops,
                        const char* type_name);

#endif
