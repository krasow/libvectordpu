/* This test will use the library to perform element wise tests on DPU data.
   David Krasowska, 2025
*/

#include "test.h"

DEFINE_BINARY_TEST(int, add, +)
DEFINE_BINARY_TEST(int, sub, -)
DEFINE_UNARY_TEST(int, negate, -a, -x)
DEFINE_UNARY_TEST(int, abs, abs(a), std::abs(x))

// DEFINE_BINARY_TEST(float, add, +)
// DEFINE_BINARY_TEST(float, sub, -)
// DEFINE_UNARY_TEST(float, negate, -a, -x)
// DEFINE_UNARY_TEST(float, abs, abs(a), std::fabs(x))

DEFINE_REDUCTION_TEST(int, sum_reduction, elements, 0, 9, 0, acc + x, sum(a))

DEFINE_REDUCTION_TEST(int, product_reduction, elements, 1, 5, 1, acc* x,
                      product(a))

DEFINE_REDUCTION_TEST(int, max_reduction, elements, 0, 999,
                      std::numeric_limits<int>::min(), (x > acc ? x : acc),
                      max(a))

DEFINE_REDUCTION_TEST(int, min_reduction, elements, 0, 999,
                      std::numeric_limits<int>::max(), (x < acc ? x : acc),
                      min(a))

// DEFINE_REDUCTION_TEST(float, sum_reduction, elements, 0.0f, 1.0f, 0.0f, acc +
// x,
//                       sum(a))

// DEFINE_REDUCTION_TEST(float, product_reduction, elements, 0.5f, 2.0f, 1.0f,
//                       acc* x, product(a))

// DEFINE_REDUCTION_TEST(float, max_reduction, elements, 0.0f, 100.0f,
//                       -std::numeric_limits<float>::infinity(),
//                       (x > acc ? x : acc), max(a))

// DEFINE_REDUCTION_TEST(float, min_reduction, elements, 0.0f, 100.0f,
//                       std::numeric_limits<float>::infinity(),
//                       (x < acc ? x : acc), min(a))

// DEFINE_REDUCTION_TEST(double, sum_reduction, elements, 0.0, 1.0, 0.0, acc +
// x,
//                       sum(a))

test_error test_chained_operations() {
  const uint32_t N = elements;

  vector<int> a(N), b(N);
  for (uint32_t i = 0; i < N; i++) {
    a[i] = random_value<int>();
    b[i] = random_value<int>();
  }
  dpu_vector<int> da = dpu_vector<int>::from_cpu(a);
  dpu_vector<int> db = dpu_vector<int>::from_cpu(b);

  // Chain operations on DPU: ((a + b) - a) -> negate -> abs
  dpu_vector<int> res = abs(-((da + db) - da));

  // Compute same operations on CPU
  vector<int> cpu_res(N);
  for (uint32_t i = 0; i < N; i++) {
    cpu_res[i] = std::abs(-((a[i] + b[i]) - a[i]));
  }

  // Transfer back and compare
  vector<int> final_res = res.to_cpu();

  // if ENABLE_AUTO_FENCING is disabled, we need to manually add a fence here
  // res.add_fence();

  for (uint32_t i = 0; i < N; i++) {
    if (final_res[i] != cpu_res[i]) {
      return TEST_ERROR;
    }
  }

  return TEST_SUCCESS;
}

#if JIT
#include <cmath>
#include <iostream>
#include <vector>

test_error test_jit_chain() {
  const uint32_t N = elements;
  std::cout << "Testing JIT chain..." << std::endl;

  vector<int> a(N), b(N);
  for (uint32_t i = 0; i < N; i++) {
    a[i] = random_value<int>();
    b[i] = random_value<int>();
  }
  dpu_vector<int> da = dpu_vector<int>::from_cpu(a);
  dpu_vector<int> db = dpu_vector<int>::from_cpu(b);

  // Manual RPN construction for: abs(-((a + b) - a))
  // a, b, ADD, a, SUB, NEGATE, ABS
  // Stack:
  // PUSH_INPUT (da)
  // PUSH_OPERAND_0 (db)
  // ADD
  // PUSH_INPUT (da)
  // SUB
  // NEGATE
  // ABS

  std::vector<uint8_t> ops = {
      OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_ADD, OP_PUSH_INPUT,
      OP_SUB,        OP_NEGATE,         OP_ABS};

  dpu_vector<int> res_jit = da.jit(ops, {db});

  // Compute same operations on CPU
  vector<int> cpu_res(N);
  for (uint32_t i = 0; i < N; i++) {
    cpu_res[i] = std::abs(-((a[i] + b[i]) - a[i]));
  }

  // Transfer back and compare
  vector<int> final_res = res_jit.to_cpu();

  for (uint32_t i = 0; i < N; i++) {
    if (final_res[i] != cpu_res[i]) {
      std::cerr << "Mismatch at " << i << ": valid=" << cpu_res[i]
                << " jit=" << final_res[i] << std::endl;
      return TEST_ERROR;
    }
  }

  return TEST_SUCCESS;
}
#endif

test_error test_fusion_lookahead() {
  const uint32_t N = elements;
  std::cout << "Testing fusion look-ahead..." << std::endl;

  vector<int> a(N), b(N), c(N);
  for (uint32_t i = 0; i < N; i++) {
    a[i] = random_value<int>();
    b[i] = random_value<int>();
    c[i] = random_value<int>();
  }
  dpu_vector<int> da = dpu_vector<int>::from_cpu(a);
  dpu_vector<int> db = dpu_vector<int>::from_cpu(b);
  dpu_vector<int> dc = dpu_vector<int>::from_cpu(c);

  // Submit multiple operations that should be fused via look-ahead
  // res = ((da + db) + dc) + da
  dpu_vector<int> res1 = da + db;
  dpu_vector<int> res2 = res1 + dc;
  dpu_vector<int> res3 = res2 + da;

  // Compute same operations on CPU
  vector<int> cpu_res(N);
  for (uint32_t i = 0; i < N; i++) {
    cpu_res[i] = ((a[i] + b[i]) + c[i]) + a[i];
  }

  // Transfer back and compare
  vector<int> final_res = res3.to_cpu();

  for (uint32_t i = 0; i < N; i++) {
    if (final_res[i] != cpu_res[i]) {
      std::cerr << "Mismatch at " << i << ": valid=" << cpu_res[i]
                << " dpu=" << final_res[i] << std::endl;
      return TEST_ERROR;
    }
  }

  return TEST_SUCCESS;
}

#if JIT
int test_jit_caching() {
  printf("Testing JIT caching...\n");
  const size_t N = 1024 * 1024;
  std::vector<int> h_a(N);

  // Fill a
  for (size_t i = 0; i < N; ++i) h_a[i] = i % 100;

  // Create DPU vector from CPU vector
  dpu_vector<int> a = dpu_vector<int>::from_cpu(h_a);

  // Ops for a + a: PUSH_INPUT, PUSH_INPUT, ADD
  std::vector<uint8_t> ops = {OP_PUSH_INPUT, OP_PUSH_INPUT, OP_ADD};

  // First call - should compile
  auto res1 = a.jit(ops);

  // Second call - should hit cache
  auto res2 = a.jit(ops);

  auto h_res1 = res1.vec.to_cpu();
  auto h_res2 = res2.vec.to_cpu();

  if (h_res1[0] != h_res2[0]) return TEST_ERROR;

  printf("[PASS] test_jit_caching (Execution successful)\n");
  return TEST_SUCCESS;
}
#endif

int main(void) {
  bool all_passed = true;
  RUN_TEST(test_int_add);
  RUN_TEST(test_int_sub);
  RUN_TEST(test_int_negate);
  RUN_TEST(test_int_abs);
  RUN_TEST(test_chained_operations);
  RUN_TEST(test_int_sum_reduction);
  RUN_TEST(test_fusion_lookahead);
#if JIT
  RUN_TEST(test_jit_chain);
  RUN_TEST(test_jit_caching);
#endif
  RUN_TEST(memory_test);

  // RUN_TEST(test_int_product_reduction);
  // RUN_TEST(test_int_max_reduction);
  // RUN_TEST(test_int_min_reduction);

  // RUN_TEST(test_float_sum_reduction);
  // RUN_TEST(test_float_product_reduction);
  // RUN_TEST(test_float_max_reduction);
  // RUN_TEST(test_float_min_reduction);

  // RUN_TEST(test_double_sum_reduction);

  if (!all_passed) {
    std::cerr << "Some tests failed.\n";
    return 1;
  }

  DpuRuntime::get().shutdown();
  std::cout << "All DPU vector tests passed successfully." << std::endl;
  return 0;
}