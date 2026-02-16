#pragma once

#include <common.h>
#include <config.h>

#include <string_view>
#include <vector>

#include "kernelids.h"
#include "opinfo.h"
#include "runtime.h"
#include "timer.h"
#include "vectordesc.h"

#if __cplusplus < 202002L
// Fake source_location for pre-C++20
// debian upmem machine has outdated compiler that doesn't support C++20 yet
namespace std {
struct source_location {
  static source_location current() { return {}; }
  constexpr const char* file_name() const { return "unknown"; }
  constexpr int line() const { return 0; }
  constexpr int column() const { return 0; }
  constexpr const char* function_name() const { return "unknown"; }
};
};  // namespace std
#else
#include <source_location>
#endif

using std::vector;

#define LOGGER_ARGS_WITH_DEFAULTS \
  std::string_view name = "",     \
                   std::source_location loc = std::source_location::current()

// ============================
// DPU Vector
// ============================
#if PIPELINE
template <typename T>
struct pipeline_result;
#endif

template <typename T>
class dpu_vector {
 public:
  dpu_vector(uint32_t n, uint32_t reserved = 0, bool lazy = false,
             LOGGER_ARGS_WITH_DEFAULTS);

  ~dpu_vector();

  dpu_vector(const dpu_vector& other);                 // copy constructor
  dpu_vector(dpu_vector&& other) noexcept;             // move constructor
  dpu_vector& operator=(const dpu_vector& other);      // copy assignment
  dpu_vector& operator=(dpu_vector&& other) noexcept;  // move assignment

  vector<T> to_cpu();

  static dpu_vector<T> from_cpu(std::vector<T>& cpu_vec,
                                LOGGER_ARGS_WITH_DEFAULTS);
  void add_fence();
  dpu_vector<T>& operator+=(const dpu_vector<T>& other);
  dpu_vector<T>& operator-=(const dpu_vector<T>& other);
  dpu_vector<T>& operator*=(const dpu_vector<T>& other);
  dpu_vector<T>& operator/=(const dpu_vector<T>& other);

  dpu_vector<T>& operator+=(T scalar);
  dpu_vector<T>& operator-=(T scalar);
  dpu_vector<T>& operator*=(T scalar);
  dpu_vector<T>& operator/=(T scalar);
  dpu_vector<T>& operator>>=(T scalar);

  dpu_vector<T> operator-() const;

#if PIPELINE
  dpu_vector<T>& operator=(const pipeline_result<T>& other);
#endif

  const detail::VectorDesc& data_desc() const { return *data_; }
  detail::VectorDescRef data_desc_ref() const { return data_; }

  uint32_t size() const { return size_; }
  uint32_t reserved() const { return reserved_; }

 private:
  detail::VectorDescRef data_;
  uint32_t size_;
  uint32_t reserved_ = 0;
  const char* debug_name = nullptr;

  const char* debug_file = nullptr;
  int debug_line = -1;
  bool copied = false;

 public:
#if PIPELINE
  pipeline_result<T> pipeline(const std::vector<uint8_t>& ops);
  pipeline_result<T> pipeline(const std::vector<uint8_t>& ops,
                              const std::vector<dpu_vector<T>>& operands);
  T pipeline_reduce(const std::vector<uint8_t>& ops,
                    const std::vector<dpu_vector<T>>& operands = {});
#endif
};

#if PIPELINE
template <typename T>
struct pipeline_result {
  dpu_vector<T> vec;
  pipeline_result(dpu_vector<T> v) : vec(v) {}
  operator T();
  operator dpu_vector<T>() { return vec; }
  dpu_vector<T>* operator->() { return &vec; }
};
#endif

namespace detail {
void launch_binary(VectorDescRef res, VectorDescRef lhs, VectorDescRef rhs,
                   KernelID kernel_id, uint8_t opcode, KernelID pipeline_kid);
void launch_binary_scalar(VectorDescRef res, VectorDescRef lhs, uint32_t scalar,
                          KernelID kernel_id, uint8_t opcode,
                          KernelID pipeline_kid);
void launch_unary(VectorDescRef res, VectorDescRef rhs, KernelID kernel_id,
                  uint8_t opcode, KernelID pipeline_kid);
void launch_reduction(VectorDescRef buf, VectorDescRef rhs, KernelID kernel_id,
                      uint8_t opcode, KernelID pipeline_kid);

void internal_launch_binary(VectorDescRef res, VectorDescRef lhs,
                            VectorDescRef rhs, KernelID kernel_id);
void internal_launch_unary(VectorDescRef res, VectorDescRef rhs,
                           KernelID kernel_id);
void internal_launch_reduction(VectorDescRef res, VectorDescRef rhs,
                               KernelID kernel_id);

#if PIPELINE
void launch_universal_pipeline(VectorDescRef res, VectorDescRef init,
                               const std::vector<uint8_t>& ops,
                               const std::vector<VectorDescRef>& operands,
                               KernelID kernel_id);

void internal_launch_universal_pipeline(
    VectorDescRef res, VectorDescRef init, const std::vector<uint8_t>& ops,
    const std::vector<VectorDescRef>& operands, KernelID kernel_id);
#endif
}  // namespace detail

#include "vectordpu.inl"