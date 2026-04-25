#pragma once

#include <common.h>
#include <config.h>

#include <cstring>
#include <string_view>
#include <vector>

#include "jit.h"
#include "kernelids.h"
#include "opinfo.h"
#include "runtime.h"
#include "timer.h"
#include "vectordesc.h"

#if __cplusplus < 202002L
// Fake source_location for pre-C++20
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

// Forward declarations
template <typename T>
class dpu_vector;

template <typename T>
struct lazy_reduction_result;

template <typename T>
class dpu_local_vector;

namespace detail {
struct jit_recorder {
  std::vector<uint8_t> rpn;
  std::vector<detail::VectorDescRef> operands;
  std::vector<detail::VectorDescRef> locals;

  uint8_t add_operand(detail::VectorDescRef d) {
    for (size_t i = 0; i < operands.size(); i++)
      if (operands[i] == d) return (uint8_t)i;
    operands.push_back(d);
    return (uint8_t)(operands.size() - 1);
  }
  uint8_t add_local(detail::VectorDescRef d) {
    for (size_t i = 0; i < locals.size(); i++)
      if (locals[i] == d) return (uint8_t)i;
    locals.push_back(d);
    return (uint8_t)(locals.size() - 1);
  }
};

struct jit_index_expr {
  jit_recorder& rec;
};
template <typename T>
struct jit_indirect_load_expr {
  const dpu_vector<T>& vec;
  jit_recorder& rec;
};
template <typename T>
struct jit_indirect_ref_expr {
  dpu_local_vector<T>& vec;
  jit_recorder& rec;

  void operator++(int);
  void apply(T value);
};
}  // namespace detail

inline detail::jit_index_expr dpu_index(detail::jit_recorder& rec) {
  return {rec};
}

#if PIPELINE
template <typename T>
struct pipeline_result;
#endif

// ============================
// DPU Vector
// ============================

template <typename T>
struct reduction_result {
  using type = T;
};

#if ENABLE_PROMOTION_REDUCTIONS
template <>
struct reduction_result<int32_t> {
  using type = int64_t;
};
#endif

template <typename T>
class dpu_vector {
 public:
  dpu_vector() noexcept;
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
  dpu_vector<T> operator==(T scalar) const;

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
  mutable bool copied = false;

  static std::vector<uint8_t> prepare_rpn(const std::vector<uint8_t>& ops);

 public:
  using reduction_result_t = typename reduction_result<T>::type;

#if PIPELINE
  pipeline_result<T> pipeline(const std::vector<uint8_t>& ops);
  pipeline_result<T> pipeline(const std::vector<uint8_t>& ops,
                              const std::vector<dpu_vector<T>>& operands);
  lazy_reduction_result<T> pipeline_reduce(
      const std::vector<uint8_t>& ops,
      const std::vector<dpu_vector<T>>& operands = {});
#endif
#if JIT
  pipeline_result<T> jit(const std::vector<uint8_t>& ops);
  pipeline_result<T> jit(const std::vector<uint8_t>& ops,
                         const std::vector<dpu_vector<T>>& operands);

  detail::jit_indirect_load_expr<T> operator[](detail::jit_index_expr idx) const;
#endif
};

template <typename T>
struct lazy_reduction_result {
  dpu_vector<T> vec;
  KernelID rid = 0;
  lazy_reduction_result() noexcept = default;
  lazy_reduction_result(dpu_vector<T> v, KernelID r)
      : vec(std::move(v)), rid(r) {}
  typename dpu_vector<T>::reduction_result_t get();
  operator typename dpu_vector<T>::reduction_result_t() { return get(); }
#if ENABLE_PROMOTION_REDUCTIONS
  operator T() { return (T)get(); }
#endif
};

template <typename T>
using dpu_future = lazy_reduction_result<T>;

enum class dpu_local_reduce_op : uint8_t {
  sum,
  product,
  min,
  max,
};

#if PIPELINE
template <typename T>
struct pipeline_result {
  dpu_vector<T> vec;
  pipeline_result(dpu_vector<T> v) : vec(std::move(v)) {}
  operator T();
  operator int64_t();
  operator dpu_vector<T>() { return std::move(vec); }
  dpu_vector<T>* operator->() { return &vec; }
  operator lazy_reduction_result<T>() {
    return lazy_reduction_result<T>(std::move(vec),
                                    vec.data_desc().reduction_rid);
  }
};
#endif

template <typename T>
lazy_reduction_result<T> sum(const dpu_vector<T>& a);
template <typename T>
lazy_reduction_result<T> product(const dpu_vector<T>& a);
template <typename T>
lazy_reduction_result<T> min(const dpu_vector<T>& a);
template <typename T>
lazy_reduction_result<T> max(const dpu_vector<T>& a);

template <typename T>
class dpu_local_vector {
 public:
  dpu_local_vector(uint32_t n, LOGGER_ARGS_WITH_DEFAULTS);
  dpu_local_vector(uint32_t n,
                   dpu_local_reduce_op reduce_op,
                   LOGGER_ARGS_WITH_DEFAULTS);
  ~dpu_local_vector() = default;

  detail::jit_indirect_ref_expr<T> operator[](
      const detail::jit_indirect_load_expr<T>& idx);

  vector<T> to_cpu();
  dpu_local_reduce_op reduce_op() const { return reduce_op_; }

  const detail::VectorDesc& data_desc() const { return *data_; }
  detail::VectorDescRef data_desc_ref() const { return data_; }
  uint32_t size() const { return size_; }

 private:
  detail::VectorDescRef data_;
  uint32_t size_;
  dpu_local_reduce_op reduce_op_ = dpu_local_reduce_op::sum;
};

template <typename T, typename F>
void dpu_jit_foreach(uint32_t n, F f);

void dpu_fence();

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
void internal_launch_unary(VectorDescRef res, VectorDescRef lhs,
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
    const std::vector<VectorDescRef>& operands, KernelID kernel_id,
    const std::vector<uint32_t>& scalars,
    const std::vector<uint32_t>& extra_scalars = {},
    const std::vector<VectorDescRef>& extra_outputs = {});

void internal_launch_jit(const std::string& binary_path, VectorDescRef output,
                         const std::vector<VectorDescRef>& inputs,
                         const std::vector<uint8_t>& rpn_ops,
                         const std::vector<uint32_t>& extra_scalars = {},
                         const std::vector<VectorDescRef>& extra_outputs = {});
#endif
}  // namespace detail

#include "vectordpu.inl"
