#include <cassert>
#include <iostream>
#include <vector>

#include "vectordpu.h"

test_error memory_test() {
  try {
    auto& runtime = DpuRuntime::get();
    runtime.init(4);  // Use 4 DPUs
    size_t num_dpus = runtime.num_dpus();
    std::cout << "Running on " << num_dpus << " DPUs." << std::endl;

    // N = (MB_PER_DPU * num_dpus) / 4 bytes
#if PIPELINE
    size_t mb_per_dpu = 15;
#else
    size_t mb_per_dpu =
        5;  // Need more space for intermediates when fusion is off
#endif
    size_t N = (mb_per_dpu * 1024 * 1024 * num_dpus) / 4;

    std::cout << "Allocating a, b, c (" << (N * 4 / 1024 / 1024)
              << " MB total, " << (N * 4 / num_dpus / 1024 / 1024)
              << " MB per DPU)..." << std::endl;
    std::vector<int32_t> data(N, 1);
    dpu_vector<int32_t> a = dpu_vector<int32_t>::from_cpu(data, "a");
    dpu_vector<int32_t> b = dpu_vector<int32_t>::from_cpu(data, "b");
    dpu_vector<int32_t> c = dpu_vector<int32_t>::from_cpu(data, "c");

    std::cout << "Running chained operation: res = a + b + c + a + b..."
              << std::endl;

    // Without lazy allocation, each '+' allocates an intermediate.
    // tmp1 = a + b (40MB) -> 160MB used
    // tmp2 = tmp1 + c (40MB) -> 200MB used
    // tmp3 = tmp2 + a (40MB) -> 240MB used
    // res = tmp3 + b (40MB) -> 280MB used -> OOM!

    auto res = a + b + c + a + b;

    std::cout << "Operation submitted successfully." << std::endl;

    auto host_res = res.to_cpu();
    std::cout << "Result retrieved. First element: " << host_res[0]
              << std::endl;
    assert(host_res[0] == 5);

    std::cout << "Test passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return TEST_ERROR;
  }
  return TEST_SUCCESS;
}
