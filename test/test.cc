/* This test will use the library to perform element wise tests on DPU data.

   The main comparison is between our driver implementation and the default
   UPMEM driver. UPMEM driver does a transpose of data to distribute it across
   DPUs, while our driver returns an mmaped region. With our driver, it doesn't
   transpose data and places data in a round robin fashion across DPUs. It is
   expected that high level element wise operations will work correctly and
   faster on our implementation due to the lack of transpose.


   David Krasowska, October 2025
*/

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

  if (!all_passed) {
    std::cerr << "Some tests failed.\n";
    return 1;
  }

  DpuRuntime::get().shutdown();
  std::cout << "All DPU vector tests passed successfully." << std::endl;
  return 0;
}