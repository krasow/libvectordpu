#pragma once

#include <common.h>

#include <cstdint>
#include <vector>

namespace detail {
struct VectorSegment {
  uint32_t ptr;
  uint32_t size_bytes;  // bytes
};

struct VectorDesc {
  size_t num_elements;    //
  uint32_t element_size;  // sizeof(T)
  uint32_t reserved_bytes;
  /// ...

  // Sharded per DPU.
  std::vector<VectorSegment> desc;
};

// Implemented in vectordpu.cc
void vec_xfer_to_dpu(char* cpu, VectorDesc& desc);
void vec_xfer_from_dpu(char* cpu, VectorDesc& desc);
void launch_binary(VectorDesc& out, const VectorDesc& lhs,
                   const VectorDesc& rhs, KernelID kernel_id);
void launch_unary(VectorDesc& out, const VectorDesc& lhs, KernelID kernel_id);
void launch_reduction(VectorDesc& buf, const VectorDesc& rhs,
                      KernelID kernel_id);
}  // namespace detail
