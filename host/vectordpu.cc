#include "logger.inl"
#include "vectordpu.inl"

// Binary operators
template <typename T>
dpu_vector<T> operator+(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  return launch_binop(lhs, rhs, BinaryKernelSelector<T>::add());
}

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs) {
  return launch_binop(lhs, rhs, BinaryKernelSelector<T>::sub());
}

// Unary operators
template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& a) {
  return launch_unary(a, UnaryKernelSelector<T>::negate());
}

template <typename T>
dpu_vector<T> abs(const dpu_vector<T>& a) {
  return launch_unary(a, UnaryKernelSelector<T>::abs());
}

template <typename T>
T sum(const dpu_vector<T>& a) {
  return launch_reduction(a, ReductionKernelSelector<T>::sum());
}

template <typename T>
T product(const dpu_vector<T>& a) {
  return launch_reduction(a, ReductionKernelSelector<T>::product());
}

template <typename T>
T max(const dpu_vector<T>& a) {
  return launch_reduction(a, ReductionKernelSelector<T>::max());
}

template <typename T>
T min(const dpu_vector<T>& a) {
  return launch_reduction(a, ReductionKernelSelector<T>::min());
}

// OperationType::COMPUTE
#define INSTANTIATE_BINARY_OP(T, OP)                     \
  template dpu_vector<T> OP<T>(const dpu_vector<T>& lhs, \
                               const dpu_vector<T>& rhs);

#define INSTANTIATE_UNARY_OP(T, OP) \
  template dpu_vector<T> OP<T>(const dpu_vector<T>& vec);

#define INSTANTIATE_REDUCE_OP(T, OP) template T OP<T>(const dpu_vector<T>& vec);

// OperationType::FENCE
#define INSTANTIATE_FENCE(T) template void dpu_vector<T>::add_fence();
// OperationType::DPU_TRANSFER and HOST_TRANSFER
#define INSTANTIATE_TO_CPU(T) template std::vector<T> dpu_vector<T>::to_cpu();
#define INSTANTIATE_FROM_CPU(T)                                           \
  template dpu_vector<T> dpu_vector<T>::from_cpu(std::vector<T>& cpu_vec, \
                                                 std::string_view name,   \
                                                 std::source_location loc);

#define INSTANTIATE_ALL(T)            \
  INSTANTIATE_BINARY_OP(T, operator+) \
  INSTANTIATE_BINARY_OP(T, operator-) \
  INSTANTIATE_UNARY_OP(T, operator-)  \
  INSTANTIATE_UNARY_OP(T, abs)        \
  INSTANTIATE_REDUCE_OP(T, sum)       \
  INSTANTIATE_REDUCE_OP(T, product)   \
  INSTANTIATE_REDUCE_OP(T, max)       \
  INSTANTIATE_REDUCE_OP(T, min)       \
  INSTANTIATE_FROM_CPU(T)             \
  INSTANTIATE_TO_CPU(T)               \
  INSTANTIATE_FENCE(T)

INSTANTIATE_ALL(int)
INSTANTIATE_ALL(float)
INSTANTIATE_ALL(double)

#undef INSTANTIATE_BINARY_OP
#undef INSTANTIATE_UNARY_OP
#undef INSTANTIATE_ABS
#undef INSTANTIATE_NEGATE
#undef INSTANTIATE_REDUCE_OP
#undef INSTANTIATE_FROM_CPU
#undef INSTANTIATE_TO_CPU
#undef INSTANTIATE_FENCE
#undef INSTANTIATE_ALL