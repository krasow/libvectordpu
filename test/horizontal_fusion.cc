#include <iostream>
#include <vector>

#include "test.h"

test_error test_horizontal_fusion_sums() {
#if PIPELINE && JIT
  const uint32_t N = 4096;
  std::cout << "Testing horizontal fusion of sums..." << std::endl;

  std::vector<int> a(N), b(N);
  long long cpu_sum_a = 0, cpu_sum_b = 0;
  for (uint32_t i = 0; i < N; i++) {
    a[i] = i % 100;
    b[i] = (i * 2) % 100;
    cpu_sum_a += a[i];
    cpu_sum_b += b[i];
  }

  dpu_vector<int> da = dpu_vector<int>::from_cpu(a);
  dpu_vector<int> db = dpu_vector<int>::from_cpu(b);

  // These should be fused horizontally
  auto lazy_sum_a = sum(da);
  auto lazy_sum_b = sum(db);

  // Trigger global sums and host fetches
  long long dpu_sum_a = (long long)lazy_sum_a.get();
  long long dpu_sum_b = (long long)lazy_sum_b.get();

  std::cout << "  CPU sum A: " << cpu_sum_a << ", DPU sum A: " << dpu_sum_a
            << std::endl;
  std::cout << "  CPU sum B: " << cpu_sum_b << ", DPU sum B: " << dpu_sum_b
            << std::endl;

  if (dpu_sum_a != cpu_sum_a || dpu_sum_b != cpu_sum_b) {
    return TEST_ERROR;
  }

  return TEST_SUCCESS;
#else
  std::cout << "Skipping horizontal fusion test (PIPELINE or JIT disabled)"
            << std::endl;
  return TEST_SUCCESS;
#endif
}

test_error test_linear_regression_like_fusion() {
#if PIPELINE && JIT
  const uint32_t N = 4096;
  std::cout << "Testing linear-regression-like horizontal fusion..."
            << std::endl;

  std::vector<int> x(N), y(N), error(N);
  long long cpu_grad_x = 0, cpu_grad_y = 0;
  for (uint32_t i = 0; i < N; i++) {
    x[i] = i % 10;
    y[i] = (i + 5) % 10;
    error[i] = 1;
    cpu_grad_x += (long long)x[i] * error[i];
    cpu_grad_y += (long long)y[i] * error[i];
  }

  dpu_vector<int> dx = dpu_vector<int>::from_cpu(x);
  dpu_vector<int> dy = dpu_vector<int>::from_cpu(y);
  dpu_vector<int> de = dpu_vector<int>::from_cpu(error);

  // Gradient computation: sum(x * error) and sum(y * error)
  // Each sum involves a MUL then a SUM.
  // We want these two chains to be fused into ONE DPU kernel pass.
  // Use explicit RPN: [PUSH_INPUT (x/y), PUSH_OPERAND_0 (error), MUL, SUM]
  auto grad_x = dx.pipeline_reduce(
      {OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_MUL, OP_SUM}, {de});
  auto grad_y = dy.pipeline_reduce(
      {OP_PUSH_INPUT, OP_PUSH_OPERAND_0, OP_MUL, OP_SUM}, {de});

  long long dpu_grad_x = (long long)grad_x.get();
  long long dpu_grad_y = (long long)grad_y.get();

  std::cout << "  CPU Grad X: " << cpu_grad_x << ", DPU Grad X: " << dpu_grad_x
            << std::endl;
  std::cout << "  CPU Grad Y: " << cpu_grad_y << ", DPU Grad Y: " << dpu_grad_y
            << std::endl;

  if (dpu_grad_x != cpu_grad_x || dpu_grad_y != cpu_grad_y) {
    return TEST_ERROR;
  }

  return TEST_SUCCESS;
#else
  return TEST_SUCCESS;
#endif
}

test_error test_hist_like_fusion() {
#if PIPELINE && JIT
  // Exercises vertical fusion (absorbed intermediate) + 3-way horizontal
  // fusion, requiring 9 scalars (3 per chain × 3 chains) — previously clipped
  // at 8.
  const uint32_t N = 4096;
  const int32_t BINS = 8;
  const int32_t DEPTH = 3;  // 2^3 = 8 = BINS
  std::cout << "Testing hist-like fusion (9 scalars, vertical+horizontal)..."
            << std::endl;

  std::vector<int32_t> a(N);
  for (uint32_t i = 0; i < N; i++) a[i] = i % 4096;

  std::vector<int32_t> cpu_counts(BINS, 0);
  for (uint32_t i = 0; i < N; i++) {
    int32_t bucket = ((int32_t)a[i] * BINS) >> DEPTH;
    if (bucket >= 0 && bucket < BINS) cpu_counts[bucket]++;
  }

  dpu_vector<int32_t> da = dpu_vector<int32_t>::from_cpu(a);
  da.add_fence();

  dpu_vector<int32_t> buckets = (da * (int32_t)BINS) >> (int32_t)DEPTH;

  std::vector<lazy_reduction_result<int32_t>> lazy_counts;
  for (int i = 0; i < BINS; i++)
    lazy_counts.push_back(sum(buckets == (int32_t)i));

  bool ok = true;
  for (int i = 0; i < BINS; i++) {
    int64_t dpu_count = (int64_t)lazy_counts[i].get();
    std::cout << "  bin " << i << ": CPU=" << cpu_counts[i]
              << " DPU=" << dpu_count << std::endl;
    if (dpu_count != cpu_counts[i]) ok = false;
  }
  return ok ? TEST_SUCCESS : TEST_ERROR;
#else
  return TEST_SUCCESS;
#endif
}

int main() {
  DpuRuntime::get().init(64);

  if (test_horizontal_fusion_sums() != TEST_SUCCESS) {
    std::cerr << "test_horizontal_fusion_sums FAILED" << std::endl;
    return 1;
  }

  if (test_linear_regression_like_fusion() != TEST_SUCCESS) {
    std::cerr << "test_linear_regression_like_fusion FAILED" << std::endl;
    return 1;
  }

  if (test_hist_like_fusion() != TEST_SUCCESS) {
    std::cerr << "test_hist_like_fusion FAILED" << std::endl;
    return 1;
  }

  std::cout << "All horizontal fusion tests PASSED" << std::endl;
  return 0;
}
