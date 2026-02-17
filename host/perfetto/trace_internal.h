#pragma once
#include <string>
#include <deque>
#include <list>
#include <memory>
#include <cstdint>

#include "queue.h"
#include "config.h"

std::string operationtype_to_string(Event::OperationType op);

#define DPU_TRACK_ID uint64_t(0x1000)
#define EVENT_TRACK_BASE uint64_t(0x2000)

#if TRACE == 1
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
}  // namespace trace

#define TRACE_INIT() trace::initialize()
#define TRACE_SHUTDOWN() trace::shutdown()
#else
#define TRACE_INIT()
#define TRACE_SHUTDOWN()
#endif
