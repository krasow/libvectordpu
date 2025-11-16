/* This test will use the library to perform element wise tests on DPU data.

   The main comparison is between our driver implementation and the default
   UPMEM driver. UPMEM driver does a transpose of data to distribute it across
   DPUs, while our driver returns an mmaped region. With our driver, it doesn't
   transpose data and places data in a round robin fashion across DPUs. It is
   expected that high level element wise operations will work correctly and
   faster on our implementation due to the lack of transpose.


   David Krasowska, October 2025
*/

#include <limits.h>
#include <runtime.h>
#include <vectordpu.h>

#include <cassert>
#include <cmath>
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

// Generic reduction test helper for any type T
template <typename T, typename Op, typename DpuOp>
test_error test_reduction(uint32_t N, T min_val, T max_val, T init_val,
                          Op cpu_op, DpuOp dpu_op) {
  vector<T> a(N);
  for (uint32_t i = 0; i < N; i++) {
    // Handle both integer and float uniformly
    if constexpr (std::is_integral<T>::value)
      a[i] = rand() % (static_cast<int>(max_val - min_val + 1)) + min_val;
    else
      a[i] = min_val + static_cast<T>(rand()) / RAND_MAX * (max_val - min_val);
  }

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  T dpu_result = dpu_op(da);

  T cpu_result = init_val;
  for (uint32_t i = 0; i < N; i++) {
    cpu_result = cpu_op(cpu_result, a[i]);
  }

  printf("[TEST] CPU result: %.6f, DPU result: %.6f\n",
         static_cast<double>(cpu_result), static_cast<double>(dpu_result));

  return (fabs(dpu_result - cpu_result) < 1e-4) ? TEST_SUCCESS : TEST_ERROR;
}

template <typename T, typename F>
test_error compare_cpu_unary(vector<T>& a, dpu_vector<T>& res, F func) {
  vector<T> cpu_res = res.to_cpu();
  for (uint32_t i = 0; i < a.size(); i++) {
    if (cpu_res[i] == func(a[i]))
      continue;
    else
      return TEST_ERROR;
  }
  return TEST_SUCCESS;
}

template <typename T, typename F>
test_error compare_cpu_binary(vector<T>& a, vector<T>& b, dpu_vector<T>& res,
                              F func) {
  vector<T> cpu_res = res.to_cpu();
  for (uint32_t i = 0; i < a.size(); i++) {
    if (cpu_res[i] == func(a[i], b[i]))
      continue;
    else {
      return TEST_ERROR;
    }
  }
  return TEST_SUCCESS;
}

test_error test_int_add() {
  const uint32_t N = 1024 * 1024;
  vector<int> a(N), b(N);
  for (uint32_t i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  dpu_vector<int> da = dpu_vector<int>::from_cpu(a);
  dpu_vector<int> db = dpu_vector<int>::from_cpu(b);
  dpu_vector<int> res = da + db;

  return compare_cpu_binary(a, b, res, [](int x, int y) { return x + y; });
}

test_error test_int_sub() {
  const uint32_t N = 1024 * 1024;
  vector<int> a(N), b(N);
  for (uint32_t i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  dpu_vector<int> da = dpu_vector<int>::from_cpu(a);
  dpu_vector<int> db = dpu_vector<int>::from_cpu(b);
  dpu_vector<int> res = da - db;

  return compare_cpu_binary(a, b, res, [](int x, int y) { return x - y; });
}

test_error test_float_add() {
  const uint32_t N = 1024 * 1024;
  vector<float> a(N), b(N);
  for (uint32_t i = 0; i < N; i++) {
    a[i] = (float)rand() / RAND_MAX;
    b[i] = (float)rand() / RAND_MAX;
  }

  dpu_vector<float> da = dpu_vector<float>::from_cpu(a);
  dpu_vector<float> db = dpu_vector<float>::from_cpu(b);
  dpu_vector<float> res = da + db;

  return compare_cpu_binary(a, b, res, [](float x, float y) { return x + y; });
}

test_error test_float_sub() {
  const uint32_t N = 1024 * 1024;
  vector<float> a(N), b(N);
  for (uint32_t i = 0; i < N; i++) {
    a[i] = (float)rand() / RAND_MAX;
    b[i] = (float)rand() / RAND_MAX;
  }

  dpu_vector<float> da = dpu_vector<float>::from_cpu(a);
  dpu_vector<float> db = dpu_vector<float>::from_cpu(b);
  dpu_vector<float> res = da - db;

  return compare_cpu_binary(a, b, res, [](float x, float y) { return x - y; });
}

test_error test_int_negate() {
  const uint32_t N = 1024 * 1024;
  vector<int> a(N);
  for (uint32_t i = 0; i < N; i++) a[i] = rand() % 200 - 100;

  dpu_vector<int> da = dpu_vector<int>::from_cpu(a);
  dpu_vector<int> res = -da;

  return compare_cpu_unary(a, res, [](int x) { return -x; });
}

test_error test_int_abs() {
  const uint32_t N = 1024 * 1024;
  vector<int> a(N);
  for (uint32_t i = 0; i < N; i++) a[i] = rand() % 200 - 100;

  dpu_vector<int> da = dpu_vector<int>::from_cpu(a);
  dpu_vector<int> res = abs(da);

  return compare_cpu_unary(a, res, [](int x) { return std::abs(x); });
}

test_error test_float_negate() {
  const uint32_t N = 1024 * 1024;
  vector<float> a(N);
  for (uint32_t i = 0; i < N; i++) a[i] = (float)rand() / RAND_MAX - 0.5f;

  dpu_vector<float> da = dpu_vector<float>::from_cpu(a);
  dpu_vector<float> res = -da;

  return compare_cpu_unary(a, res, [](float x) { return -x; });
}

test_error test_float_abs() {
  const uint32_t N = 1024 * 1024;
  vector<float> a(N);
  for (uint32_t i = 0; i < N; i++) a[i] = (float)rand() / RAND_MAX - 0.5f;

  dpu_vector<float> da = dpu_vector<float>::from_cpu(a);
  dpu_vector<float> res = abs(da);

  return compare_cpu_unary(a, res, [](float x) { return std::fabs(x); });
}

test_error test_chained_operations() {
  const uint32_t N = 1024 * 1024;

  vector<int> a(N), b(N);
  for (uint32_t i = 0; i < N; i++) {
    a[i] = rand() % 200 - 100;
    b[i] = rand() % 200 - 100;
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

test_error test_sum_reduction_int() {
  return test_reduction<int>(
      1024 * 1024, 0, 9, 0, [](int acc, int x) { return acc + x; },
      [](auto& da) { return sum(da); });
}

test_error test_product_reduction_int() {
  return test_reduction<int>(
      1024, 1, 5, 1, [](int acc, int x) { return acc * x; },
      [](auto& da) { return product(da); });
}

test_error test_max_reduction_int() {
  return test_reduction<int>(
      1024 * 1024, 0, 999, std::numeric_limits<int>::min(),
      [](int acc, int x) { return (x > acc) ? x : acc; },
      [](auto& da) { return max(da); });
}

test_error test_min_reduction_int() {
  return test_reduction<int>(
      1024 * 1024, 0, 999, std::numeric_limits<int>::max(),
      [](int acc, int x) { return (x < acc) ? x : acc; },
      [](auto& da) { return min(da); });
}
test_error test_sum_reduction_float() {
  return test_reduction<float>(
      1024, 0.0f, 1.0f, 0.0f, [](float acc, float x) { return acc + x; },
      [](auto& da) { return sum(da); });
}

test_error test_sum_reduction_double() {
  return test_reduction<double>(
      1024*1024, 0.0, 1.0, 0.0, [](double acc, double x) { return acc + x; },
      [](auto& da) { return sum(da); });
}

test_error test_product_reduction_float() {
  return test_reduction<float>(
      1024, 0.5f, 2.0f, 1.0f, [](float acc, float x) { return acc * x; },
      [](auto& da) { return product(da); });
}

test_error test_max_reduction_float() {
  return test_reduction<float>(
      1024 * 1024, 0.0f, 100.0f, -std::numeric_limits<float>::infinity(),
      [](float acc, float x) { return (x > acc) ? x : acc; },
      [](auto& da) { return max(da); });
}

test_error test_min_reduction_float() {
  return test_reduction<float>(
      1024 * 1024, 0.0f, 100.0f, std::numeric_limits<float>::infinity(),
      [](float acc, float x) { return (x < acc) ? x : acc; },
      [](auto& da) { return min(da); });
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

  // RUN_TEST(test_sum_reduction_int);
  // RUN_TEST(test_product_reduction_int);
  // RUN_TEST(test_max_reduction_int);
  // RUN_TEST(test_min_reduction_int);

  // RUN_TEST(test_sum_reduction_double);
  // RUN_TEST(test_sum_reduction_float);
  // RUN_TEST(test_product_reduction_float);
  // RUN_TEST(test_max_reduction_float);
  // RUN_TEST(test_min_reduction_float);

  if (!all_passed) {
    std::cerr << "Some tests failed.\n";
    return 1;
  }

  DpuRuntime::get().shutdown();
  std::cout << "All DPU vector tests passed successfully." << std::endl;
  return 0;
}