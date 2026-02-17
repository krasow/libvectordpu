#include "allocator.h"

#include <algorithm>
#include <stdexcept>

#include "logger.h"
#include "perfetto/trace.h"
#include "runtime.h"

allocator::allocator(uint32_t start_addr, std::size_t dpu_mem,
                     std::size_t num_dpus)
    : start_addr_(start_addr), dpu_mem_(dpu_mem), num_dpus_(num_dpus) {
  ptrs_.resize(num_dpus_, start_addr_);
  sizes_.resize(num_dpus_, dpu_mem_);
  offsets_.resize(num_dpus_, 0);
  free_list_.resize(num_dpus_);
}

detail::VectorDescRef allocator::allocate_upmem_vector(std::size_t n,
                                                       std::size_t reserved,
                                                       std::size_t size_type,
                                                       bool lazy) {
  bool uniform = (n % num_dpus_ == 0) && (n / num_dpus_ * size_type >= 8);
  {
    std::lock_guard<std::mutex> lock(this->lock);
    if (is_synchronized_ && uniform)
      return allocate_upmem_vector_broadcast(n, reserved, size_type, lazy);
    is_synchronized_ = false;
  }
  std::lock_guard<std::mutex> lock(this->lock);
  size_t eff = n / num_dpus_, rem = (eff / size_type) % num_dpus_;
  if (eff * size_type == 4) eff = 2;

  auto vec = std::make_shared<detail::VectorDesc>();
  for (size_t i = 0; i < num_dpus_; i++) {
    size_t sz = (eff + (i < rem ? 1 : 0)) * size_type + reserved;
    size_t aligned_sz = (sz + 7) & ~7;
    vec->desc.push_back({!lazy ? raw_allocate(i, aligned_sz) : 0, (uint32_t)sz,
                         (uint32_t)aligned_sz});
  }
  vec->ptr_allocated = !lazy;
  vec->reserved_bytes = reserved;
  vec->element_size = size_type;
  vec->num_elements = n;
  return vec;
}

detail::VectorDescRef allocator::allocate_upmem_vector_broadcast(
    std::size_t n, std::size_t reserved, std::size_t size_type, bool lazy) {
  size_t sz = std::max((size_t)8, (n / num_dpus_) * size_type) + reserved;
  size_t aligned_sz = (sz + 7) & ~7;
  auto vec = std::make_shared<detail::VectorDesc>();
  uint32_t addr = !lazy ? raw_allocate(DPU_BROADCAST, aligned_sz) : 0;
  vec->desc.assign(num_dpus_, {addr, (uint32_t)sz, (uint32_t)aligned_sz});
  vec->ptr_allocated = !lazy;
  vec->reserved_bytes = reserved;
  vec->element_size = size_type;
  vec->num_elements = n;
  return vec;
}

void allocator::realize_allocation(detail::VectorDescRef data) {
  if (data->ptr_allocated) return;
  std::lock_guard<std::mutex> lock(this->lock);
  if (is_synchronized_) {
    uint32_t addr = raw_allocate(DPU_BROADCAST, data->desc[0].allocated_bytes);
    for (auto& s : data->desc) s.ptr = addr;
  } else {
    for (size_t i = 0; i < num_dpus_; i++)
      data->desc[i].ptr = raw_allocate(i, data->desc[i].allocated_bytes);
  }
  data->ptr_allocated = true;
}

uint32_t allocator::raw_allocate(int id, std::size_t n) {
  auto& fl = (id == DPU_BROADCAST) ? broadcast_free_list_ : free_list_[id];
  auto best = fl.end();
  size_t bsz = SIZE_MAX;
  for (auto it = fl.begin(); it != fl.end(); ++it)
    if (it->size >= n && it->size < bsz) {
      best = it;
      bsz = it->size;
    }

  if (best != fl.end()) {
    uint32_t addr = best->addr;
    if (best->size > n) {
      best->addr += n;
      best->size -= n;
    } else
      fl.erase(best);
    total_allocated_bytes_ += n * (id == DPU_BROADCAST ? num_dpus_ : 1);
    TRACE_COUNTER("runtime", "total_bytes", total_allocated_bytes_);
    return addr;
  }

  uint32_t& off = (id == DPU_BROADCAST) ? broadcast_offset_ : offsets_[id];
  if (id == DPU_BROADCAST)
    off = *std::max_element(offsets_.begin(), offsets_.end());
  if (off + n > (id == DPU_BROADCAST ? sizes_[0] : sizes_[id]))
    throw std::runtime_error("DPU OOM");

  uint32_t addr = ptrs_[0] + off;
  off += n;
  if (id == DPU_BROADCAST) {
    std::fill(offsets_.begin(), offsets_.end(), off);
  }
  total_allocated_bytes_ += n * (id == DPU_BROADCAST ? num_dpus_ : 1);
  TRACE_COUNTER("runtime", "total_bytes", total_allocated_bytes_);
  return addr;
}

void allocator::deallocate_upmem_vector(detail::VectorDesc* data) {
  if (!data->ptr_allocated) return;
  data->ptr_allocated = false;
  std::lock_guard<std::mutex> lock(this->lock);
  if (is_synchronized_ && !data->desc.empty()) {
    raw_deallocate(DPU_BROADCAST, data->desc[0].ptr,
                   data->desc[0].allocated_bytes);
    return;
  }
  is_synchronized_ = false;
  for (size_t i = 0; i < num_dpus_; i++)
    raw_deallocate(i, data->desc[i].ptr, data->desc[i].allocated_bytes);
}

void allocator::deallocate_upmem_vector_broadcast(detail::VectorDesc* data) {
  deallocate_upmem_vector(data);
}

void allocator::raw_deallocate(int id, uint32_t addr, size_t sz) {
  auto& fl = (id == DPU_BROADCAST) ? broadcast_free_list_ : free_list_[id];
  auto it = std::find_if(fl.begin(), fl.end(),
                         [&](const FreeBlock& b) { return b.addr > addr; });
  auto ins = fl.insert(it, {addr, sz});
  if (ins != fl.begin()) {
    auto p = std::prev(ins);
    if (p->addr + p->size == ins->addr) {
      p->size += ins->size;
      fl.erase(ins);
      ins = p;
    }
  }
  auto nxt = std::next(ins);
  if (nxt != fl.end() && ins->addr + ins->size == nxt->addr) {
    ins->size += nxt->size;
    fl.erase(nxt);
  }

  uint32_t& off = (id == DPU_BROADCAST) ? broadcast_offset_ : offsets_[id];
  if (id == DPU_BROADCAST && !fl.empty() &&
      fl.back().addr + fl.back().size == start_addr_ + off) {
    off -= fl.back().size;
    fl.pop_back();
    std::fill(offsets_.begin(), offsets_.end(), off);
  }
  total_allocated_bytes_ -= sz * (id == DPU_BROADCAST ? num_dpus_ : 1);
  TRACE_COUNTER("runtime", "total_bytes", total_allocated_bytes_);
}