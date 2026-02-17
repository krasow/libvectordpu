#pragma once
#include <string>
#include <cstdint>

#include "config.h"

// Descriptive helpers available even if TRACE=0
std::string opcode_to_string(uint8_t op);

namespace trace {

// Library-internal opaque tracing hooks to avoid pulling Perfetto into user headers.
// These are implemented in trace.cc using the full Perfetto SDK.
void internal_reduction_begin(uint64_t flow_id);
void internal_reduction_end();
void internal_to_cpu_begin(uint64_t flow_id);
void internal_to_cpu_end();
void internal_from_cpu_begin();
void internal_from_cpu_end();

// Opaque hooks for general tracing
void counter(const char* cat, const char* name, int64_t value);
void event_begin(const char* cat, const char* name);
void event_end(const char* cat);

struct scoped_event {
  const char* category;
  scoped_event(const char* cat, const char* name) : category(cat) {
    event_begin(cat, name);
  }
  ~scoped_event() { event_end(category); }
};

// RAII helpers for public template functions in vectordpu.inl.
// These call the opaque functions above so the user program doesn't need <perfetto.h>.
struct reduction_cpu {
  reduction_cpu(uint64_t flow_id) { internal_reduction_begin(flow_id); }
  ~reduction_cpu() { internal_reduction_end(); }
};

struct to_cpu {
  to_cpu(uint64_t flow_id) { internal_to_cpu_begin(flow_id); }
  ~to_cpu() { internal_to_cpu_end(); }
};

struct from_cpu {
  from_cpu() { internal_from_cpu_begin(); }
  ~from_cpu() { internal_from_cpu_end(); }
};

} // namespace trace

// Standard Trace Init/Shutdown
#if TRACE == 1
namespace trace {
void initialize();
void shutdown();
}
#define TRACE_INIT() trace::initialize()
#define TRACE_SHUTDOWN() trace::shutdown()
#else
#define TRACE_INIT()
#define TRACE_SHUTDOWN()
#endif