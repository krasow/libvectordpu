#pragma once

#include <common.h>

#include <iostream>
#include <string_view>
#include <type_traits>
#include <vector>

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
using vector_desc =
    std::pair<vector<uint32_t>, vector<uint32_t>>;  // ptrs and sizes

#define LOGGER_ARGS_WITH_DEFAULTS \
  std::string_view name = "",     \
                   std::source_location loc = std::source_location::current()

// ============================q
// DPU Vector
// ============================
template <typename T>
class dpu_vector {
 public:
  dpu_vector(uint32_t n, LOGGER_ARGS_WITH_DEFAULTS);

  ~dpu_vector();

  dpu_vector(const dpu_vector& other);             // copy constructor
  dpu_vector& operator=(const dpu_vector& other);  // copy assignment
  vector<uint32_t> data() const;
  uint32_t size() const;

  vector<T> to_cpu();

  static dpu_vector<T> from_cpu(std::vector<T>& cpu_vec,
                                LOGGER_ARGS_WITH_DEFAULTS);
  void add_fence();

  vector_desc data_desc() const { return data_; }

 private:
  vector_desc data_;
  uint32_t size_;
  const char* debug_name = nullptr;
  const char* debug_file = nullptr;
  int debug_line = -1;
  bool copied = false;
};

// ============================
// Kernel selectors
// ============================
template <typename T>
struct BinaryKernelSelector;

// float specialization
template <>
struct BinaryKernelSelector<float> {
  static KernelID add() { return KernelID::K_BINARY_FLOAT_ADD; }
  static KernelID sub() { return KernelID::K_BINARY_FLOAT_SUB; }
};

// int specialization
template <>
struct BinaryKernelSelector<int> {
  static KernelID add() { return KernelID::K_BINARY_INT_ADD; }
  static KernelID sub() { return KernelID::K_BINARY_INT_SUB; }
};

template <typename T>
struct UnaryKernelSelector;

// float specialization
template <>
struct UnaryKernelSelector<float> {
  static KernelID negate() { return KernelID::K_UNARY_FLOAT_NEGATE; }
  static KernelID abs() { return KernelID::K_UNARY_FLOAT_ABS; }
};

// int specialization
template <>
struct UnaryKernelSelector<int> {
  static KernelID negate() { return KernelID::K_UNARY_INT_NEGATE; }
  static KernelID abs() { return KernelID::K_UNARY_INT_ABS; }
};

template <typename T>
struct ReductionKernelSelector;

// float specialization
template <>
struct ReductionKernelSelector<float> {
  static KernelID sum() { return KernelID::K_REDUCTION_FLOAT_SUM; }
  static KernelID product() { return KernelID::K_REDUCTION_FLOAT_PRODUCT; }
  static KernelID max() { return KernelID::K_REDUCTION_FLOAT_MAX; }
  static KernelID min() { return KernelID::K_REDUCTION_FLOAT_MIN; }
};

// int specialization
template <>
struct ReductionKernelSelector<int> {
  static KernelID sum() { return KernelID::K_REDUCTION_INT_SUM; }
  static KernelID product() { return KernelID::K_REDUCTION_INT_PRODUCT; }
  static KernelID max() { return KernelID::K_REDUCTION_INT_MAX; }
  static KernelID min() { return KernelID::K_REDUCTION_INT_MIN; }
};

// ============================
// DPU Launch helpers
// ============================
template <typename T>
dpu_vector<T> launch_binop(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs,
                           KernelID kernel_id);

template <typename T>
dpu_vector<T> launch_unary(const dpu_vector<T>& a, KernelID kernel_id);

// ============================
// Operators
// ============================
template <typename T>
dpu_vector<T> operator+(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs);

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs);

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& a);

template <typename T>
dpu_vector<T> abs(const dpu_vector<T>& a);

template <typename T>
dpu_vector<T> sum(const dpu_vector<T>& a);

template <typename T>
dpu_vector<T> product(const dpu_vector<T>& a);

template <typename T>
dpu_vector<T> max(const dpu_vector<T>& a);

template <typename T>
dpu_vector<T> min(const dpu_vector<T>& a); 