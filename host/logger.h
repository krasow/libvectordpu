#pragma once
#include <common.h>

#include <iomanip>
#include <iostream>
#include <mutex>
#include <string_view>
#include <vector>

using vector_desc =
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;  // ptrs and sizes

class Logger {
  std::mutex mtx_;
  std::ostream& stream_;

 public:
  Logger(std::ostream& stream = std::cout) : stream_(stream) {}

  // Proxy object that locks the mutex for the duration of the object
  struct Lock {
    std::ostream& stream;
    std::unique_lock<std::mutex> lock;

    Lock(Logger& logger) : stream(logger.stream_), lock(logger.mtx_) {}

    // For generic types
    template <typename T>
    Lock& operator<<(const T& value) {
      stream << value;
      return *this;
    }

    // For manipulators like std::endl
    using Manip = std::ostream& (*)(std::ostream&);
    Lock& operator<<(Manip manip) {
      manip(stream);
      return *this;
    }
  };

  Lock lock() { return Lock(*this); }
};

inline void print_vector_desc(vector_desc desc, uint32_t reserved);

inline void log_allocation(const std::type_info& type, uint32_t n,
                           std::string_view debug_name, const char* debug_file,
                           int debug_line, bool is_allocation = true);

#define log_deallocation(type, n, debug_name, debug_file, debug_line) \
  log_allocation(type, n, debug_name, debug_file, debug_line, false)

inline void log_dpu_launch_args(const DPU_LAUNCH_ARGS* args,
                                uint32_t nr_of_dpus);