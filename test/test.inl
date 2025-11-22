
template <typename T>
T random_value();

template <>
inline int random_value<int>() {
  return rand() % 200 - 100;
}

template <>
inline float random_value<float>() {
  return (float)rand() / RAND_MAX - 0.5f;
}

template <>
inline double random_value<double>() {
  return (double)rand() / RAND_MAX - 0.5;
}

template <typename T, typename Fn>
auto cpu_equiv(Fn fn) {
  return [fn](T x, T y) { return fn(x, y); };
}

template <typename T, typename Fn>
auto cpu_equiv_unary(Fn fn) {
  return [fn](T x) { return fn(x); };
}

template <typename T, typename FnDPU, typename FnCPU>
test_error test_binary_op(FnDPU dpu_op, FnCPU cpu_op) {
  const uint32_t N = 1024 * 1024;
  std::vector<T> a(N), b(N);

  for (uint32_t i = 0; i < N; i++) {
    a[i] = random_value<T>();
    b[i] = random_value<T>();
  }

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);
  dpu_vector<T> res = dpu_op(da, db);

  return compare_cpu_binary(a, b, res, cpu_op);
}

template <typename T, typename FnDPU, typename FnCPU>
test_error test_unary_op(FnDPU dpu_op, FnCPU cpu_op) {
  const uint32_t N = 1024 * 1024;
  std::vector<T> a(N);

  for (uint32_t i = 0; i < N; i++) a[i] = random_value<T>();

  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> res = dpu_op(da);

  return compare_cpu_unary(a, res, cpu_op);
}

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
