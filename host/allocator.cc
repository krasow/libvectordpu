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

detail::VectorDescRef allocator::allocate_upmem_vector(
    std::size_t n, std::size_t reserved_mem_per_dpu, std::size_t size_type) {
  // Check if we can use optimized broadcast allocation
  bool uniform_size = (n % num_dpus_ == 0);

  if (uniform_size) {
    size_t elems_per_dpu = n / num_dpus_;
    if (elems_per_dpu * size_type < 4) {
      if (elems_per_dpu * size_type < 8) uniform_size = false;
    }
  }

  {
    std::lock_guard<std::mutex> lock(this->lock);
    if (is_synchronized_ && uniform_size) {
      return allocate_upmem_vector_broadcast(n, reserved_mem_per_dpu,
                                             size_type);
    }
  }

  // Fallback to per-DPU allocation (original logic)
  std::lock_guard<std::mutex> lock(this->lock);

  is_synchronized_ = false;

  size_t effective_elems = n / num_dpus_;
  if (effective_elems * size_type == 4) effective_elems = 2;
  size_t rem = (effective_elems / size_type) % num_dpus_;

  auto vec_desc = std::make_shared<detail::VectorDesc>();

  for (size_t i = 0; i < num_dpus_; i++) {
    size_t alloc_size = (effective_elems + (i < rem ? 1 : 0)) * size_type;
    uint32_t addr = allocate(i, alloc_size + reserved_mem_per_dpu);

    detail::VectorSegment seg{
        addr, static_cast<uint32_t>(alloc_size + reserved_mem_per_dpu)};
    vec_desc->desc.push_back(seg);
  }

  vec_desc->reserved_bytes = static_cast<uint32_t>(reserved_mem_per_dpu);
  vec_desc->element_size = static_cast<uint32_t>(size_type);
  vec_desc->num_elements = n;

  return vec_desc;
}

detail::VectorDescRef allocator::allocate_upmem_vector_broadcast(
    std::size_t n, std::size_t reserved_mem_per_dpu, std::size_t size_type) {
  std::size_t num_dpus = this->num_dpus_;
  size_t elems_per_dpu = n / num_dpus;

  // Ensure 8-byte alignment/minimum size logic
  if (elems_per_dpu * size_type < 8) {
    // If less than 8 bytes, round up to 8 bytes.
    // Assuming size_type <= 8. if size_type is 4, we need 2 elems.
    // If size_type is 8, 1 elem is fine.
    // The original logic was: if (elems * size == 4) elems = 2;
    // This covers the specific case of 4-byte integers.
    // What if size is 1 or 2? Not supported heavily but should be safe.
    // Let's just enforce alloc_size >= 8.
    // Allocation logic uses alloc_size (bytes).
    // Elems calculation is just for derivation.
  }

  size_t alloc_size = elems_per_dpu * size_type;
  if (alloc_size < 8) alloc_size = 8;

  size_t total_size = alloc_size + reserved_mem_per_dpu;

  // Perform O(1) allocation
  uint32_t addr = allocate_broadcast(total_size);

  auto vec_desc = std::make_shared<detail::VectorDesc>();

  detail::VectorSegment seg{addr, static_cast<uint32_t>(total_size)};
  vec_desc->desc.resize(num_dpus, seg);

  vec_desc->reserved_bytes = static_cast<uint32_t>(reserved_mem_per_dpu);
  vec_desc->element_size = static_cast<uint32_t>(size_type);
  vec_desc->num_elements = n;

  return vec_desc;
}

uint32_t allocator::allocate_broadcast(std::size_t n) {
  auto& flist = broadcast_free_list_;

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

  uint32_t addr = ptrs_[0] + broadcast_offset_;

  if (broadcast_offset_ + n > sizes_[0]) {
    throw std::runtime_error("DPU out of memory (broadcast)!");
  }

  broadcast_offset_ += n;

  // Sync offsets for fallback compatibility
  std::fill(offsets_.begin(), offsets_.end(), broadcast_offset_);

  return addr;
}

void allocator::deallocate_upmem_vector_broadcast(detail::VectorDescRef data) {
  std::size_t alloc_size = data->desc[0].size_bytes;  // Uniform size
  uint32_t addr = data->desc[0].ptr;                  // Uniform address

  deallocate_broadcast(addr, alloc_size);
}

void allocator::deallocate_broadcast(uint32_t addr, std::size_t size) {
  // Logic identical to deallocate() but using broadcast_free_list_
  FreeBlock new_block{addr, size};
  auto& flist = broadcast_free_list_;

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
      inserted = flist.erase(inserted);
      inserted = prev;
    }
  }
  // Merge with next block if adjacent
  auto next = std::next(inserted);
  if (next != flist.end() && inserted->addr + inserted->size == next->addr) {
    inserted->size += next->size;
    flist.erase(next);
  }
}

void allocator::deallocate_upmem_vector(detail::VectorDescRef data) {
  std::lock_guard<std::mutex> lock(this->lock);

  // Check if we can use optimized broadcast deallocation
  if (!data->desc.empty()) {
    // We could check all, but if we assume they were created uniformly...
    // Let's rely on is_synchronized_ and just check if allocation was likely
    // broadcast If we differ, is_synchronized_ should have been false already.
    // But let's be safe: unique vectors might have different ptrs?
    // Actually, allocate_upmem_vector creates uniform ptrs if bump pointer is
    // sync. But if we ever supported non-uniform, we must check. Checking 1024
    // items is O(N), but fast memory access. Let's just check if
    // is_synchronized_.
    if (is_synchronized_) {
      // If synced, we expect them to be uniform.
      // Call broadcast dealloc.
      deallocate_upmem_vector_broadcast(data);
      return;
    }
  }

  // Fallback
  // If we are here, either not synced or empty.
  is_synchronized_ = false;  // We are entering per-DPU mode (or already were)

  for (size_t i = 0; i < num_dpus_; ++i) {
    uint32_t addr = data->desc[i].ptr;
    size_t size = data->desc[i].size_bytes;
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
