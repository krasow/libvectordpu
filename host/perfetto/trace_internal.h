#pragma once
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <string>

#include "config.h"
#include "queue.h"

std::string operationtype_to_string(Event::OperationType op);

#define DPU_TRACK_ID uint64_t(0x1000)
#define EVENT_TRACK_BASE uint64_t(0x2000)

namespace trace {
#if TRACE == 1
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
#else
inline void initialize() {}
inline void shutdown() {}
inline void event_enqueued(std::shared_ptr<Event> e,
                           const std::deque<std::shared_ptr<Event>>& ops,
                           const std::list<std::shared_ptr<Event>>& running) {
  (void)e;
  (void)ops;
  (void)running;
}
inline void event_fused(std::shared_ptr<Event> e, std::shared_ptr<Event> into,
                        const std::string& fused_ops) {
  (void)e;
  (void)into;
  (void)fused_ops;
}
inline void inqueue_end(std::shared_ptr<Event> e) { (void)e; }
inline void execution_begin(std::shared_ptr<Event> e) { (void)e; }
inline void execution_end() {}
inline void active_ops_counter(size_t count) { (void)count; }
inline void ensure_callback_thread_named() {}
#endif
}  // namespace trace

#if TRACE == 1
#define TRACE_INIT() trace::initialize()
#define TRACE_SHUTDOWN() trace::shutdown()
#else
#define TRACE_INIT()
#define TRACE_SHUTDOWN()
#endif
