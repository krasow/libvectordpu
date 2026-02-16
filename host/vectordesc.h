#pragma once

#include <common.h>
#include <config.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace detail {
struct VectorSegment {
  uint32_t ptr;
  uint32_t size_bytes;  // bytes
};

struct VectorDesc {
  size_t num_elements;    // total number of elements
  uint32_t element_size;  // sizeof(T)
  uint32_t reserved_bytes;
  /// ...

  // Sharded per DPU.
  std::vector<VectorSegment> desc;

  bool is_reduction_result = false;
  KernelID reduction_rid;

  bool ptr_allocated =
      true;  // Default to true for backwards/eager compatibility
  size_t last_producer_id = 0;
};

using VectorDescRef = std::shared_ptr<VectorDesc>;

// Implemented in vectordpu.cc
void vec_xfer_to_dpu(char* cpu, VectorDescRef desc);
void vec_xfer_from_dpu(char* cpu, VectorDescRef desc);
void launch_binary(VectorDescRef out, VectorDescRef lhs, VectorDescRef rhs,
                   KernelID kernel_id);
void launch_unary(VectorDescRef out, VectorDescRef lhs, KernelID kernel_id);
void launch_reduction(VectorDescRef buf, VectorDescRef rhs, KernelID kernel_id);
}  // namespace detail
