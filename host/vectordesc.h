#pragma once

#include <common.h>
#include <config.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace detail {
struct VectorSegment {
  uint32_t ptr;
  uint32_t size_bytes;       // Logical bytes (used for count)
  uint32_t allocated_bytes;  // Physical bytes (aligned to 8)
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

  bool ptr_allocated = false;
  size_t last_producer_id = 0;

  // When this vector is absorbed by vertical fusion (i.e., it is an
  // intermediate result that is consumed inline and never written to MRAM),
  // these fields record the RPN prefix and scalar values that produce it,
  // along with the primary source vector.  Later events that need this vector
  // can inline the prefix rather than reading from (unwritten) MRAM.
  std::vector<uint8_t> absorbed_rpn;
  std::vector<uint32_t> absorbed_scalars;
  std::vector<std::shared_ptr<VectorDesc>>
      absorbed_inputs;  // full input list of absorbed event

  // True when this vector is a shared intermediate written by a standalone
  // kernel (e.g. error_shifted consumed by DIM gradient chains).  Prevents
  // try_vfuse from absorbing it on-stack, which would skip the MRAM write.
  bool is_shared_intermediate = false;

  bool is_local_vector = false;
  uint8_t local_reduce_opcode = OP_SUM;

  const char* type_name = nullptr;
  const char* debug_name = nullptr;
  const char* debug_file = nullptr;
  int debug_line = -1;
  virtual ~VectorDesc();
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
