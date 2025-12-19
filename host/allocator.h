#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <utility>
#include <vector>

using std::size_t;
using std::vector;

#include "desc.h"

struct FreeBlock {
  uint32_t addr;
  size_t size;
};

class allocator {
 public:
  allocator(uint32_t start_addr, std::size_t total_size, std::size_t num_dpus);

  detail::VectorDescRef allocate_upmem_vector(std::size_t n,
                                              std::size_t reserved_mem_per_dpu,
                                              std::size_t size_type);
  void deallocate_upmem_vector(detail::VectorDescRef data);

 private:
  uint32_t start_addr_;  // starting base address
  std::size_t dpu_mem_;  // memory size per DPU
  std::size_t num_dpus_;

  vector<uint32_t> ptrs_;                        // base addresses per DPU
  vector<uint32_t> sizes_;                       // total size per DPU
  vector<uint32_t> offsets_;                     // bump pointer per DPU
  std::vector<std::list<FreeBlock>> free_list_;  // free blocks per DPU

  // Allocate 'n' units on a specific DPU
  uint32_t allocate(std::size_t dpu_id, std::size_t n);

  // Deallocate a block and merge adjacent free blocks
  void deallocate(std::size_t dpu_id, uint32_t addr, size_t size);

  std::mutex lock;
};