/* This test will use the library to perform element wise tests on DPU data.
   David Krasowska, 2025
*/

#include "test.h"

DEFINE_BINARY_TEST(int, add, +)
DEFINE_BINARY_TEST(int, sub, -)
DEFINE_UNARY_TEST(int, negate, -a, -x)
DEFINE_UNARY_TEST(int, abs, abs(a), std::abs(x))

DEFINE_BINARY_TEST(float, add, +)
DEFINE_BINARY_TEST(float, sub, -)
DEFINE_UNARY_TEST(float, negate, -a, -x)
DEFINE_UNARY_TEST(float, abs, abs(a), std::fabs(x))

DEFINE_REDUCTION_TEST(int, sum_reduction, elements, 0, 9, 0, acc + x, sum(a))

DEFINE_REDUCTION_TEST(int, product_reduction, elements, 1, 5, 1, acc* x,
                      product(a))

DEFINE_REDUCTION_TEST(int, max_reduction, elements, 0, 999,
                      std::numeric_limits<int>::min(), (x > acc ? x : acc),
                      max(a))

DEFINE_REDUCTION_TEST(int, min_reduction, elements, 0, 999,
                      std::numeric_limits<int>::max(), (x < acc ? x : acc),
                      min(a))

DEFINE_REDUCTION_TEST(float, sum_reduction, elements, 0.0f, 1.0f, 0.0f, acc + x,
                      sum(a))

DEFINE_REDUCTION_TEST(float, product_reduction, elements, 0.5f, 2.0f, 1.0f,
                      acc* x, product(a))

DEFINE_REDUCTION_TEST(float, max_reduction, elements, 0.0f, 100.0f,
                      -std::numeric_limits<float>::infinity(),
                      (x > acc ? x : acc), max(a))

DEFINE_REDUCTION_TEST(float, min_reduction, elements, 0.0f, 100.0f,
                      std::numeric_limits<float>::infinity(),
                      (x < acc ? x : acc), min(a))

DEFINE_REDUCTION_TEST(double, sum_reduction, elements, 0.0, 1.0, 0.0, acc + x,
                      sum(a))

test_error test_chained_operations() {
  const uint32_t N = elements * elements;

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

int main(void) {
  bool all_passed = true;
  RUN_TEST(test_int_add);
  RUN_TEST(test_int_sub);
  RUN_TEST(test_float_add);
  RUN_TEST(test_float_sub);
  RUN_TEST(test_int_negate);
  RUN_TEST(test_int_abs);
  RUN_TEST(test_float_negate);
  RUN_TEST(test_float_abs);
  RUN_TEST(test_chained_operations);

  RUN_TEST(test_int_sum_reduction);
  // RUN_TEST(test_int_product_reduction);
  // RUN_TEST(test_int_max_reduction);
  // RUN_TEST(test_int_min_reduction);

  RUN_TEST(test_float_sum_reduction);
  // RUN_TEST(test_float_product_reduction);
  // RUN_TEST(test_float_max_reduction);
  // RUN_TEST(test_float_min_reduction);

  RUN_TEST(test_double_sum_reduction);

  if (!all_passed) {
    std::cerr << "Some tests failed.\n";
    return 1;
  }

  DpuRuntime::get().shutdown();
  std::cout << "All DPU vector tests passed successfully." << std::endl;
  return 0;
}