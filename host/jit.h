#pragma once

#include <cstdint>
#include <string>
#include <utility>  // For std::pair
#include <vector>

#include "config.h"

#if JIT

// Compiles a batch of unique RPN sequences into a single DPU binary
std::string jit_compile(
    const std::vector<std::pair<std::vector<uint8_t>, std::string>>& kernels);

// Cleanup JIT files at shutdown
void jit_cleanup();

#endif
