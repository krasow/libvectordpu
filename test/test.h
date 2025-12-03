#include <limits.h>
#include <runtime.h>
#include <vectordpu.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

using test_error = uint32_t;

#define TEST_UNIMPLIMENTED 2
#define TEST_ERROR 1
#define TEST_SUCCESS 0

#define RUN_TEST(fn)                                              \
  do {                                                            \
    auto result = fn();                                           \
    if (result == TEST_SUCCESS) {                                 \
      std::cout << "[PASS] " << #fn << std::endl;                 \
    } else {                                                      \
      std::cerr << "[FAIL] " << #fn << " (code " << result << ")" \
                << std::endl;                                     \
      all_passed = false;                                         \
    }                                                             \
  } while (0)

#define DEFINE_BINARY_TEST(TYPE, NAME, OP)                         \
  test_error test_##TYPE##_##NAME() {                              \
    return test_binary_op<TYPE>(                                   \
        [](const dpu_vector<TYPE>& a, const dpu_vector<TYPE>& b) { \
          return a OP b;                                           \
        },                                                         \
        cpu_equiv<TYPE>([](TYPE x, TYPE y) { return x OP y; }));   \
  }

#define DEFINE_UNARY_TEST(TYPE, NAME, DPU_EXPR, CPU_EXPR)   \
  test_error test_##TYPE##_##NAME() {                       \
    return test_unary_op<TYPE>(                             \
        [](const dpu_vector<TYPE>& a) { return DPU_EXPR; }, \
        [](TYPE x) { return CPU_EXPR; });                   \
  }

#define DEFINE_REDUCTION_TEST(TYPE, NAME, N, LO, HI, INIT, CPU_EXPR, DPU_EXPR) \
  test_error test_##TYPE##_##NAME() {                                          \
    return test_reduction<TYPE>(                                               \
        N, LO, HI, INIT, [](TYPE acc, TYPE x) { return (CPU_EXPR); },          \
        [](const dpu_vector<TYPE>& a) { return (DPU_EXPR); });                 \
  }

#include "test.inl"