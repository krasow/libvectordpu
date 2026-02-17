#include "allocator.h"

#include <algorithm>
#include <stdexcept>

#include "logger.h"
#include "perfetto/trace.h"
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
    std::size_t n, std::size_t reserved_mem_per_dpu, std::size_t size_type,
    bool lazy) {
  // Check if we can use optimized broadcast allocation
  bool uniform_size = (n % num_dpus_ == 0);

  if (uniform_size) {
    size_t elems_per_dpu = n / num_dpus_;
    if (elems_per_dpu * size_type < 8) {
      uniform_size = false;
    }
  }

  {
    std::lock_guard<std::mutex> lock(this->lock);
    if (is_synchronized_ && uniform_size) {
      return allocate_upmem_vector_broadcast(n, reserved_mem_per_dpu, size_type,
                                             lazy);
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
    uint32_t addr = 0;
    if (!lazy) {
      addr = allocate(i, alloc_size + reserved_mem_per_dpu);
    }

    detail::VectorSegment seg{
        addr, static_cast<uint32_t>(alloc_size + reserved_mem_per_dpu)};
    vec_desc->desc.push_back(seg);
  }

  vec_desc->ptr_allocated = !lazy;

  vec_desc->reserved_bytes = static_cast<uint32_t>(reserved_mem_per_dpu);
  vec_desc->element_size = static_cast<uint32_t>(size_type);
  vec_desc->num_elements = n;

  TRACE_EVENT("runtime", "allocate_upmem_vector", "num_elements", (uint64_t)n,
              "num_dpus", (uint32_t)num_dpus_, "element_size",
              (uint32_t)size_type, "total_bytes", (uint64_t)(n * size_type));

  return vec_desc;
}

detail::VectorDescRef allocator::allocate_upmem_vector_broadcast(
    std::size_t n, std::size_t reserved_mem_per_dpu, std::size_t size_type,
    bool lazy) {
  std::size_t num_dpus = this->num_dpus_;
  size_t elems_per_dpu = n / num_dpus;

  size_t alloc_size = elems_per_dpu * size_type;
  if (alloc_size < 8) alloc_size = 8;

  size_t total_size = alloc_size + reserved_mem_per_dpu;

  // Perform O(1) allocation
  uint32_t addr = 0;
  if (!lazy) {
    addr = allocate_broadcast(total_size);
  }

  auto vec_desc = std::make_shared<detail::VectorDesc>();

  detail::VectorSegment seg{addr, static_cast<uint32_t>(total_size)};
  vec_desc->desc.resize(num_dpus, seg);
  vec_desc->ptr_allocated = !lazy;

  vec_desc->reserved_bytes = static_cast<uint32_t>(reserved_mem_per_dpu);
  vec_desc->element_size = static_cast<uint32_t>(size_type);
  vec_desc->num_elements = n;

  return vec_desc;
}

void allocator::realize_allocation(detail::VectorDescRef data) {
  if (data->ptr_allocated) return;

  std::lock_guard<std::mutex> lock(this->lock);
  if (is_synchronized_) {
    size_t total_size = data->desc[0].size_bytes;
    uint32_t addr = allocate_broadcast(total_size);
    for (size_t i = 0; i < num_dpus_; ++i) {
      data->desc[i].ptr = addr;
    }
  } else {
    for (size_t i = 0; i < num_dpus_; ++i) {
      size_t total_size = data->desc[i].size_bytes;
      data->desc[i].ptr = allocate(i, total_size);
    }
  }
  data->ptr_allocated = true;
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
    total_allocated_bytes_ += n * num_dpus_;
    TRACE_COUNTER("runtime", "allocated_bytes", total_allocated_bytes_);
    return addr;
  }

  // Ensure we use the maximum offset across all DPUs
  uint32_t max_offset = *std::max_element(offsets_.begin(), offsets_.end());
  uint32_t addr = ptrs_[0] + max_offset;

  if (max_offset + n > sizes_[0]) {
    Logger& logger = DpuRuntime::get().get_logger();
    logger.lock() << "[allocator] broadcast out of memory: requested " << n
                  << " bytes, available " << (sizes_[0] - max_offset)
                  << " bytes" << std::endl;
    throw std::runtime_error("DPU out of memory! (broadcast)");
  }

  broadcast_offset_ = max_offset + n;
  std::fill(offsets_.begin(), offsets_.end(), broadcast_offset_);

  total_allocated_bytes_ += n * num_dpus_;
  TRACE_COUNTER("runtime", "allocated_bytes", total_allocated_bytes_);
  TRACE_COUNTER("runtime", "broadcast_offset", (uint64_t)broadcast_offset_);
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

  // Retract bump pointer if possible
  if (!flist.empty()) {
    auto last = flist.back();
    if (last.addr + last.size == start_addr_ + broadcast_offset_) {
      broadcast_offset_ -= last.size;
      std::fill(offsets_.begin(), offsets_.end(), broadcast_offset_);
      flist.pop_back();
      TRACE_COUNTER("runtime", "broadcast_offset", (uint64_t)broadcast_offset_);
    }
  }

  total_allocated_bytes_ -= size * num_dpus_;
  TRACE_COUNTER("runtime", "allocated_bytes", total_allocated_bytes_);
}

void allocator::deallocate_upmem_vector(detail::VectorDescRef data) {
  if (!data->ptr_allocated) return;  // Nothing to do if never realized

  std::lock_guard<std::mutex> lock(this->lock);

  // Check if we can use optimized broadcast deallocation
  if (!data->desc.empty()) {
    if (is_synchronized_) {
      deallocate_broadcast(data->desc[0].ptr, data->desc[0].size_bytes);
      TRACE_EVENT("runtime", "deallocate_upmem_vector_broadcast");
      return;
    }
  }

  // Fallback
  is_synchronized_ = false;

  for (size_t i = 0; i < num_dpus_; ++i) {
    uint32_t addr = data->desc[i].ptr;
    size_t size = data->desc[i].size_bytes;
    deallocate(i, addr, size);
  }

  TRACE_EVENT("runtime", "deallocate_upmem_vector");
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
    total_allocated_bytes_ += n;
    TRACE_COUNTER("runtime", "allocated_bytes", total_allocated_bytes_);
    return addr;
  }

  if (offsets_[dpu_id] + n > sizes_[dpu_id]) {
    Logger& logger = DpuRuntime::get().get_logger();
    logger.lock() << "[allocator] DPU " << dpu_id
                  << " out of memory: requested " << n << " bytes, available "
                  << (sizes_[dpu_id] - offsets_[dpu_id]) << " bytes"
                  << std::endl;
    throw std::runtime_error("DPU out of memory! (non-broadcast)");
  }

  uint32_t addr = ptrs_[dpu_id] + offsets_[dpu_id];
  offsets_[dpu_id] += n;
  total_allocated_bytes_ += n;
  TRACE_COUNTER("runtime", "allocated_bytes", total_allocated_bytes_);
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

  // // Retract bump pointer if possible
  // if (!flist.empty()) {
  //   auto last = flist.back();
  //   if (last.addr + last.size == ptrs_[dpu_id] + offsets_[dpu_id]) {
  //     offsets_[dpu_id] -= last.size;
  //     flist.pop_back();
  //     // If we are still synchronized, we can update the global
  //     broadcast_offset if (is_synchronized_) {
  //       broadcast_offset_ = *std::max_element(offsets_.begin(),
  //       offsets_.end()); TRACE_COUNTER("runtime", "broadcast_offset",
  //                     (uint64_t)broadcast_offset_);
  //     }
  //   }
  // }

  total_allocated_bytes_ -= size;
  TRACE_COUNTER("runtime", "allocated_bytes", total_allocated_bytes_);
}
