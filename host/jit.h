#pragma once

#include <cstdint>
#include <string>
#include <utility>  // For std::pair
#include <vector>

#include "config.h"

// Number of statically compiled kernels in the default DPU binary.
// JIT-compiled kernels are assigned IDs starting at this offset.
static constexpr uint32_t JIT_STATIC_KERNEL_COUNT = 17;

#if JIT

// Compiles a batch of unique RPN sequences into a single DPU binary
std::string jit_compile(
    const std::vector<std::pair<std::vector<uint8_t>, std::string>>& kernels);

// Cleanup JIT files at shutdown
void jit_cleanup();

#endif
