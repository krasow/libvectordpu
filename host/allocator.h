#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <utility>
#include <vector>

using std::size_t;
using std::vector;
using vector_desc =
    std::pair<vector<uint32_t>, vector<uint32_t>>;  // ptrs and sizes

struct FreeBlock {
  uint32_t addr;
  size_t size;
};

class allocator {
 public:
  allocator(uint32_t start_addr, std::size_t total_size, std::size_t num_dpus);

  vector_desc allocate_upmem_vector(std::size_t n, std::size_t size_type);
  void deallocate_upmem_vector(vector_desc &data);

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

  // Get vector_desc (pointers and sizes per DPU)
  vector_desc get_vector_desc() const;

  std::mutex lock;
};