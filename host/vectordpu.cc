#include "vectordpu.inl"

#include "logger.inl"

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

// Template instantiations for shared library
#define INSTANTIATE_BINARY_OP(T, OP)                     \
  template dpu_vector<T> OP<T>(const dpu_vector<T>& lhs, \
                               const dpu_vector<T>& rhs);
#define INSTANTIATE_UNARY_OP(T, OP) \
  template dpu_vector<T> OP<T>(const dpu_vector<T>& vec);
#define INSTANTIATE_FENCE(T) template void dpu_vector<T>::add_fence();
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
  INSTANTIATE_FROM_CPU(T)             \
  INSTANTIATE_TO_CPU(T)               \
  INSTANTIATE_FENCE(T)

INSTANTIATE_ALL(int)
INSTANTIATE_ALL(float)

#undef INSTANTIATE_BINARY_OP
#undef INSTANTIATE_UNARY_OP
#undef INSTANTIATE_ABS
#undef INSTANTIATE_NEGATE
#undef INSTANTIATE_FROM_CPU
#undef INSTANTIATE_TO_CPU
#undef INSTANTIATE_FENCE
#undef INSTANTIATE_ALL