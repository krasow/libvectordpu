#pragma once
#include <string>

#include "queue.h"

#define DPU_TRACK_ID uint64_t(0x1000)
#define EVENT_TRACK_BASE uint64_t(0x2000)

// Descriptive helpers available even if TRACE=0 (useful for logging)
std::string operationtype_to_string(Event::OperationType op);
std::string opcode_to_string(uint8_t op);

#if TRACE == 1
#include <perfetto.h>

#include <deque>
#include <list>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("runtime").SetDescription(
        "Events related to runtime init and shutdown"),
    perfetto::Category("queue").SetDescription(
        "Events related to the event queue"),
    perfetto::Category("transfer")
        .SetDescription("Events related to MRAM transfers"),
    perfetto::Category("events").SetDescription(
        "Actual operation execution events"));

namespace trace {
void initialize();
void shutdown();
void event_enqueued(std::shared_ptr<Event> e,
                    const std::deque<std::shared_ptr<Event>>& ops,
                    const std::list<std::shared_ptr<Event>>& running);
void event_fused(std::shared_ptr<Event> e, std::shared_ptr<Event> into,
                 const std::string& fused_ops);
void inqueue_end(std::shared_ptr<Event> e);
void execution_begin(std::shared_ptr<Event> e);
void execution_end();
void active_ops_counter(size_t count);
void ensure_callback_thread_named();

// For handling the asynchronous start of next event in callback
void execution_begin_next(std::shared_ptr<Event> next);
}  // namespace trace

#define TRACE_INIT() trace::initialize()
#define TRACE_SHUTDOWN() trace::shutdown()
#define TRACE_EVENT_ENQUEUED(e, ops, running) \
  trace::event_enqueued(e, ops, running)
#define TRACE_EVENT_FUSED(e, into, ops) trace::event_fused(e, into, ops)
#define TRACE_INQUEUE_END(e) trace::inqueue_end(e)
#define TRACE_EXECUTION_BEGIN(e) trace::execution_begin(e)
#define TRACE_EXECUTION_END() trace::execution_end()
#define TRACE_ACTIVE_OPS(count) trace::active_ops_counter(count)
#define TRACE_CALLBACK_THREAD() trace::ensure_callback_thread_named()
#define TRACE_EXECUTION_BEGIN_NEXT(next) trace::execution_begin(next)

#else
#include <cstdint>
/* perfetto */
#define TRACE_EVENT(...)
#define TRACE_EVENT_INSTANT(...)
#define TRACE_EVENT_BEGIN(...)
#define TRACE_EVENT_END(...)
#define TRACE_COUNTER(...)
/* our custom stuff */
#define TRACE_INIT()
#define TRACE_SHUTDOWN()
#define TRACE_EVENT_ENQUEUED(e, ops, running)
#define TRACE_EVENT_FUSED(e, into, ops)
#define TRACE_INQUEUE_END(e)
#define TRACE_EXECUTION_BEGIN(e)
#define TRACE_EXECUTION_END()
#define TRACE_ACTIVE_OPS(count)
#define TRACE_CALLBACK_THREAD()
#define TRACE_EXECUTION_BEGIN_NEXT(next)

/* fake perfetto */
namespace perfetto {
struct Track {
  explicit Track(uint64_t) {}
};
struct StaticString {
  explicit StaticString(const char*) {}
};
struct DynamicString {
  explicit DynamicString(const char*) {}
};
}  // namespace perfetto
#endif