
#include "allocator.h"

#include <algorithm>
#include <stdexcept>

#include "logger.h"
#include "runtime.h"

allocator::allocator(uint32_t start_addr, std::size_t dpu_mem,
                     std::size_t num_dpus)
    : start_addr_(start_addr), dpu_mem_(dpu_mem), num_dpus_(num_dpus) {
  // Initialize internal state, but do NOT pre-allocate vectors
  ptrs_.resize(num_dpus_, start_addr_);  // all start at base address
  sizes_.resize(num_dpus_, dpu_mem_);
  offsets_.resize(num_dpus_, 0);
  free_list_.resize(num_dpus_);
}

vector_desc allocator::allocate_upmem_vector(std::size_t n,
                                             std::size_t size_type) {
  // grab lock
  std::lock_guard<std::mutex> lock(this->lock);
  std::size_t num_dpus = this->num_dpus_;
  vector<uint32_t> vec_ptrs(num_dpus);
  vector<uint32_t> vec_sizes(num_dpus);

  size_t size_per_dpu = n / num_dpus;
  size_t remainder = n % num_dpus;

  for (size_t i = 0; i < num_dpus; i++) {
    size_t alloc_size = (size_per_dpu + (i < remainder ? 1 : 0)) * size_type;
    uint32_t addr = allocate(i, alloc_size);  // use bump/free-list allocator

    vec_ptrs[i] = addr;
    vec_sizes[i] = alloc_size;
  }

  return std::make_pair(vec_ptrs, vec_sizes);
}

void allocator::deallocate_upmem_vector(vector_desc& data) {
  std::lock_guard<std::mutex> lock(this->lock);
  for (size_t i = 0; i < num_dpus_; ++i) {
    uint32_t addr = data.first[i];
    size_t size = data.second[i];
    deallocate(i, addr, size);
  }
}

uint32_t allocator::allocate(std::size_t dpu_id, std::size_t n) {
  if (dpu_id >= num_dpus_) throw std::out_of_range("Invalid DPU ID");

  auto& flist = free_list_[dpu_id];

  // best-fit free block
  auto best_it = flist.end();
  size_t best_size = SIZE_MAX;
  for (auto it = flist.begin(); it != flist.end(); ++it) {
    if (it->size >= n && it->size < best_size) {
      best_it = it;
      best_size = it->size;
    }
  }

  if (best_it != flist.end()) {
    uint32_t addr = best_it->addr;
    if (best_it->size > n) {
      best_it->addr += n;
      best_it->size -= n;
    } else {
      flist.erase(best_it);
    }
    return addr;
  }

  if (offsets_[dpu_id] + n > sizes_[dpu_id]) {
    Logger& logger = DpuRuntime::get().get_logger();
    logger.lock() << "[allocator] DPU " << dpu_id
                  << " out of memory: requested " << n << " bytes, available "
                  << (sizes_[dpu_id] - offsets_[dpu_id]) << " bytes"
                  << std::endl;
    throw std::runtime_error("DPU out of memory!");
  }

  uint32_t addr = ptrs_[dpu_id] + offsets_[dpu_id];
  offsets_[dpu_id] += n;
  return addr;
}

void allocator::deallocate(std::size_t dpu_id, uint32_t addr, size_t size) {
  if (dpu_id >= num_dpus_) throw std::out_of_range("Invalid DPU ID");

  FreeBlock new_block{addr, size};
  auto& flist = free_list_[dpu_id];

  // Find the first block whose address is greater than new_block
  auto it = std::find_if(flist.begin(), flist.end(),
                         [&](const FreeBlock& b) { return b.addr > addr; });

  // Insert the new block at the found position
  auto inserted = flist.insert(it, new_block);

  // Merge with previous block if adjacent
  if (inserted != flist.begin()) {
    auto prev = std::prev(inserted);
    if (prev->addr + prev->size == inserted->addr) {
      prev->size += inserted->size;
      inserted =
          flist.erase(inserted);  // erase returns iterator to next element
      inserted = prev;            // keep prev as the merged block
    }
  }
  // Merge with next block if adjacent
  auto next = std::next(inserted);
  if (next != flist.end() && inserted->addr + inserted->size == next->addr) {
    inserted->size += next->size;
    flist.erase(next);
  }
}
vector_desc allocator::get_vector_desc() const {
  return std::make_pair(ptrs_, sizes_);
}
