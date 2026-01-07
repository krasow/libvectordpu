#pragma once

#include <common.h>
#include <config.h>

#include <string_view>
#include <vector>

#include "kernelids.h"
#include "opinfo.h"
#include "runtime.h"
#include "vectordesc.h"
#include "timer.h"

#if __cplusplus < 202002L
// Fake source_location for pre-C++20
// debian upmem machine has outdated compiler that doesn't support C++20 yet
namespace std {
struct source_location {
  static source_location current() { return {}; }
  constexpr const char* file_name() const { return "unknown"; }
  constexpr int line() const { return 0; }
  constexpr int column() const { return 0; }
  constexpr const char* function_name() const { return "unknown"; }
};
};  // namespace std
#else
#include <source_location>
#endif

using std::vector;

#define LOGGER_ARGS_WITH_DEFAULTS \
  std::string_view name = "",     \
                   std::source_location loc = std::source_location::current()

// ============================
// DPU Vector
// ============================
template <typename T>
class dpu_vector {
 public:
  dpu_vector(uint32_t n, uint32_t reserved = 0, LOGGER_ARGS_WITH_DEFAULTS);

  ~dpu_vector();

  dpu_vector(const dpu_vector& other);             // copy constructor
  dpu_vector& operator=(const dpu_vector& other);  // copy assignment

  vector<T> to_cpu();

  static dpu_vector<T> from_cpu(std::vector<T>& cpu_vec,
                                LOGGER_ARGS_WITH_DEFAULTS);
  void add_fence();

  const detail::VectorDesc& data_desc() const { return *data_; }
  detail::VectorDescRef data_desc_ref() const { return data_; }

  uint32_t size() const { return size_; }
  uint32_t reserved() const { return reserved_; }

 private:
  detail::VectorDescRef data_;
  uint32_t size_;
  uint32_t reserved_ = 0;
  const char* debug_name = nullptr;
  const char* debug_file = nullptr;
  int debug_line = -1;
  bool copied = false;
};

#include "vectordpu.inl"