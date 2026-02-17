#include "perfetto/trace.h"
#include "perfetto/trace_internal.h"

#if TRACE == 1 && __has_include(<perfetto.h>)
#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("runtime").SetDescription(
        "Events related to runtime init and shutdown"),
    perfetto::Category("queue").SetDescription(
        "Events related to the event queue"),
    perfetto::Category("transfer")
        .SetDescription("Events related to MRAM transfers"),
    perfetto::Category("events").SetDescription(
        "Actual operation execution events"));

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
#endif

#include <cstdio>
#include <vector>

#include "runtime.h"

std::string operationtype_to_string(Event::OperationType op) {
  switch (op) {
    case Event::OperationType::COMPUTE:
      return "COMPUTE";
    case Event::OperationType::DPU_TRANSFER:
      return "DPU_TRANSFER";
    case Event::OperationType::HOST_TRANSFER:
      return "HOST_TRANSFER";
    case Event::OperationType::FENCE:
      return "FENCE";
    default:
      return "UNKNOWN";
  }
}

void trace::internal_reduction_begin(uint64_t flow_id) {
#if TRACE == 1 && __has_include(<perfetto.h>)
  TRACE_EVENT_BEGIN("events", "reduction_cpu", [flow_id](perfetto::EventContext& ctx) {
    if (flow_id) perfetto::Flow::ProcessScoped(flow_id)(ctx);
  });
#endif
}

void trace::internal_reduction_end() {
#if TRACE == 1 && __has_include(<perfetto.h>)
  TRACE_EVENT_END("events");
#endif
}

void trace::internal_to_cpu_begin(uint64_t flow_id) {
#if TRACE == 1 && __has_include(<perfetto.h>)
  TRACE_EVENT_BEGIN("transfer", "dpu_vector::to_cpu", [flow_id](perfetto::EventContext& ctx) {
    if (flow_id) perfetto::Flow::ProcessScoped(flow_id)(ctx);
  });
#endif
}

void trace::internal_to_cpu_end() {
#if TRACE == 1 && __has_include(<perfetto.h>)
  TRACE_EVENT_END("transfer");
#endif
}

void trace::internal_from_cpu_begin() {
#if TRACE == 1 && __has_include(<perfetto.h>)
  TRACE_EVENT_BEGIN("transfer", "dpu_vector::from_cpu");
#endif
}

void trace::internal_from_cpu_end() {
#if TRACE == 1 && __has_include(<perfetto.h>)
  TRACE_EVENT_END("transfer");
#endif
}

void trace::counter(const char* cat, const char* name, int64_t value) {
#if TRACE == 1 && __has_include(<perfetto.h>)
  if (std::string(cat) == "runtime") {
    TRACE_COUNTER("runtime", perfetto::DynamicString(name), value);
  } else if (std::string(cat) == "queue") {
    TRACE_COUNTER("queue", perfetto::DynamicString(name), value);
  }
#endif
}

void trace::event_begin(const char* cat, const char* name) {
#if TRACE == 1 && __has_include(<perfetto.h>)
  if (std::string(cat) == "runtime") {
    TRACE_EVENT_BEGIN("runtime", perfetto::DynamicString(name));
  } else if (std::string(cat) == "queue") {
    TRACE_EVENT_BEGIN("queue", perfetto::DynamicString(name));
  }
#endif
}

void trace::event_end(const char* cat) {
#if TRACE == 1 && __has_include(<perfetto.h>)
  if (std::string(cat) == "runtime") {
    TRACE_EVENT_END("runtime");
  } else if (std::string(cat) == "queue") {
    TRACE_EVENT_END("queue");
  }
#endif
}

std::string opcode_to_string(uint8_t op) {
  switch (op) {
    case OP_IDENTITY:
      return "IDENTITY";
    case OP_NEGATE:
      return "NEGATE";
    case OP_ABS:
      return "ABS";
    case OP_ADD:
      return "ADD";
    case OP_SUB:
      return "SUB";
    case OP_MUL:
      return "MUL";
    case OP_DIV:
      return "DIV";
    case OP_ASR:
      return "ASR";
    case OP_ADD_SCALAR:
      return "ADD_SCALAR";
    case OP_SUB_SCALAR:
      return "SUB_SCALAR";
    case OP_MUL_SCALAR:
      return "MUL_SCALAR";
    case OP_DIV_SCALAR:
      return "DIV_SCALAR";
    case OP_ASR_SCALAR:
      return "ASR_SCALAR";
    case OP_MIN:
      return "MIN";
    case OP_MAX:
      return "MAX";
    case OP_SUM:
      return "SUM";
    case OP_PRODUCT:
      return "PRODUCT";
    case OP_PUSH_INPUT:
    case OP_PUSH_OPERAND_0:
    case OP_PUSH_OPERAND_1:
    case OP_PUSH_OPERAND_2:
    case OP_PUSH_OPERAND_3:
    case OP_PUSH_OPERAND_4:
    case OP_PUSH_OPERAND_5:
    case OP_PUSH_OPERAND_6:
    case OP_PUSH_OPERAND_7:
      return "";
    default:
      return "UNK(" + std::to_string(op) + ")";
  }
}

#if TRACE == 1
static std::string vector_to_string(detail::VectorDescRef vec) {
  if (!vec) return "NULL";
  char buf[128];
  uint32_t addr = (vec->desc.empty() ? 0 : vec->desc[0].ptr);
  snprintf(buf, sizeof(buf), "[ptr=0x%x, size=%zu, elems=%zu]", addr,
           (size_t)vec->num_elements * vec->element_size, vec->num_elements);
  return std::string(buf);
}

static std::string get_pipeline_breakdown(const Event& e) {
  if (e.rpn_ops.empty()) return "";
  std::string breakdown;
  std::vector<std::string> stack;
  int op_idx = 1;
  for (uint8_t op : e.rpn_ops) {
    if (op == OP_PUSH_INPUT) {
      stack.push_back("In[0]");
    } else if (op >= OP_PUSH_OPERAND_0 && op <= OP_PUSH_OPERAND_7) {
      stack.push_back("In[" + std::to_string(op - OP_PUSH_OPERAND_0 + 1) + "]");
    } else if (IS_OP_UNARY(op)) {
      if (stack.size() < 1) {
        breakdown += "!!STK_ERR!!\n";
        break;
      }
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = "st[" + std::to_string(stack.size()) + "]";
      breakdown += std::to_string(op_idx++) + ". " + res + " = " +
                   opcode_to_string(op) + "(" + s1 + ")\n";
      stack.push_back(res);
    } else if (IS_OP_BINARY(op)) {
      if (stack.size() < 2) {
        breakdown += "!!STK_ERR!!\n";
        break;
      }
      std::string s2 = stack.back();
      stack.pop_back();
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = "st[" + std::to_string(stack.size()) + "]";
      breakdown += std::to_string(op_idx++) + ". " + res + " = " +
                   opcode_to_string(op) + "(" + s1 + ", " + s2 + ")\n";
      stack.push_back(res);
    } else if (IS_OP_REDUCTION(op)) {
      if (stack.size() < 1) {
        breakdown += "!!STK_ERR!!\n";
        break;
      }
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = "RED_RES";
      breakdown += std::to_string(op_idx++) + ". " + res + " = " +
                   opcode_to_string(op) + "(" + s1 + ")\n";
      stack.push_back(res);
    }
  }
  if (!stack.empty()) {
    breakdown += "Final Output: " + stack.back();
  }
  return breakdown;
}

static void add_event_metadata(perfetto::EventContext& ctx,
                               std::shared_ptr<Event> e) {
  if (e->op == Event::OperationType::COMPUTE) {
    for (size_t i = 0; i < e->inputs.size(); ++i) {
      ctx.AddDebugAnnotation(
          perfetto::DynamicString("in[" + std::to_string(i) + "]"),
          vector_to_string(e->inputs[i]));
    }
    if (e->output) {
      ctx.AddDebugAnnotation("out", vector_to_string(e->output));
    }
    std::string breakdown = get_pipeline_breakdown(*e);
    if (!breakdown.empty()) {
      ctx.AddDebugAnnotation("pipeline_breakdown", breakdown);
    }
  } else if (e->op == Event::OperationType::DPU_TRANSFER ||
             e->op == Event::OperationType::HOST_TRANSFER) {
    if (e->host_ptr) {
      char buf[64];
      snprintf(buf, sizeof(buf), "0x%p", e->host_ptr);
      ctx.AddDebugAnnotation("host_buffer", std::string(buf));
    }
    if (e->transfer_size > 0) {
      ctx.AddDebugAnnotation("size_bytes", (uint64_t)e->transfer_size);
    }
    if (e->op == Event::OperationType::DPU_TRANSFER) {
      ctx.AddDebugAnnotation("direction", "Host -> DPU");
      if (e->output) {
        ctx.AddDebugAnnotation("dpu_dest", vector_to_string(e->output));
      }
    } else {
      ctx.AddDebugAnnotation("direction", "DPU -> Host");
      if (!e->inputs.empty() && e->inputs[0]) {
        ctx.AddDebugAnnotation("dpu_src", vector_to_string(e->inputs[0]));
      }
    }
  }
}

#include <fstream>
#include <iostream>

static std::unique_ptr<perfetto::TracingSession> tracing_session_;

namespace trace {

void initialize() {
  perfetto::TracingInitArgs args;
  args.backends |= perfetto::kInProcessBackend;
  perfetto::Tracing::Initialize(args);
  perfetto::TrackEvent::Register();

  // Start tracing session
  perfetto::TraceConfig cfg;
  cfg.add_buffers()->set_size_kb(64 * 1024);  // 64MB
  auto* ds_cfg = cfg.add_data_sources()->mutable_config();
  ds_cfg->set_name("track_event");

  tracing_session_ = perfetto::Tracing::NewTrace(perfetto::kInProcessBackend);
  tracing_session_->Setup(cfg);
  tracing_session_->StartBlocking();

  // Initialize global tracks
  auto track_desc = perfetto::Track(DPU_TRACK_ID).Serialize();
  track_desc.set_name("DPU Hardware");
  perfetto::TrackEvent::SetTrackDescriptor(perfetto::Track(DPU_TRACK_ID),
                                           track_desc);
}

void shutdown() {
  if (tracing_session_) {
    std::cout << "[trace] Stopping tracing session..." << std::endl;
    tracing_session_->StopBlocking();
    std::cout << "[trace] Reading trace data..." << std::endl;
    std::vector<char> trace_data = tracing_session_->ReadTraceBlocking();

    std::ofstream out("trace.perfetto-trace", std::ios::binary);
    out.write(trace_data.data(), trace_data.size());
    out.close();

    std::cout << "Trace written to trace.perfetto-trace (" << trace_data.size()
              << " bytes)" << std::endl;
    std::cout << "[trace] Resetting tracing session..." << std::endl;
    tracing_session_.reset();
    std::cout << "[trace] Tracing session reset complete." << std::endl;
  }
}

void event_enqueued(std::shared_ptr<Event> e,
                    const std::deque<std::shared_ptr<Event>>& ops,
                    const std::list<std::shared_ptr<Event>>& running) {
  TRACE_EVENT_INSTANT("queue", "EventEnqueued",
                      perfetto::Track(EVENT_TRACK_BASE + e->id),
                      [e](perfetto::EventContext& ctx) {
                        perfetto::Flow::ProcessScoped(e->id)(ctx);
                      },
                      "type", operationtype_to_string(e->op), "id", e->id);

  // Calculate what we are waiting on (Running + Queued)
  std::string waiting_on;
  for (const auto& active : running) {
    if (!waiting_on.empty()) waiting_on += ", ";
    waiting_on += "Run[" + std::to_string(active->id) +
                  "]:" + operationtype_to_string(active->op);
  }
  for (const auto& queued : ops) {
    if (!waiting_on.empty()) waiting_on += ", ";
    waiting_on += "Wait[" + std::to_string(queued->id) +
                  "]:" + operationtype_to_string(queued->op);
  }

  std::string in_queue_name = "InQueue: " + operationtype_to_string(e->op);
  if (e->op == Event::OperationType::COMPUTE) {
    if (!e->rpn_ops.empty()) {
      in_queue_name = "InQueue: Fused Pipeline";
    } else if (e->opcode != 0) {
      in_queue_name = "InQueue: " + opcode_to_string(e->opcode);
    } else {
      in_queue_name = "InQueue: " + std::string(kernel_id_to_string(e->kid));
    }
  }

  TRACE_EVENT_BEGIN("queue", perfetto::DynamicString(in_queue_name),
                    perfetto::Track(EVENT_TRACK_BASE + e->id), "id", e->id,
                    [e](perfetto::EventContext& ctx) {
                      perfetto::Flow::ProcessScoped(e->id)(ctx);
                    },
                    "waiting_on_details", waiting_on, "queue_depth",
                    (int)ops.size());
}

void event_fused(std::shared_ptr<Event> e, std::shared_ptr<Event> into,
                 const std::string& fused_ops) {
  TRACE_EVENT_INSTANT(
      "queue",
      perfetto::DynamicString("Fused [" + fused_ops + "] into #" +
                              std::to_string(into->id)),
      perfetto::Track(EVENT_TRACK_BASE + e->id), "into_id", into->id,
      "new_ops_count", (int)into->rpn_ops.size());
  // End the InQueue slice for the fused event
  TRACE_EVENT_END("queue", perfetto::Track(EVENT_TRACK_BASE + e->id));
}

void inqueue_end(std::shared_ptr<Event> e) {
  TRACE_EVENT_END("queue", perfetto::Track(EVENT_TRACK_BASE + e->id));
}

void execution_begin(std::shared_ptr<Event> e) {
  auto base_lambda = [e](perfetto::EventContext& ctx) {
    perfetto::Flow::ProcessScoped(e->id)(ctx);
    for (size_t dep_id : e->dependencies) {
      perfetto::Flow::ProcessScoped(dep_id)(ctx);
    }
  };

  if (!e->rpn_ops.empty()) {
    std::string ops_str;
    for (size_t i = 0; i < e->rpn_ops.size(); ++i) {
      uint8_t op = e->rpn_ops[i];
      std::string s = opcode_to_string(op);
      if (s.empty()) continue;
      if (!ops_str.empty()) ops_str += ", ";
      ops_str += s;
      if (IS_OP_SCALAR(op)) {
        i += sizeof(uint32_t);
      }
    }

    auto event_lambda = [e, base_lambda, ops_str](perfetto::EventContext& ctx) {
      base_lambda(ctx);
      add_event_metadata(ctx, e);
    };

    TRACE_EVENT_BEGIN("events", perfetto::DynamicString(e->slice_name),
                      perfetto::Track(DPU_TRACK_ID), "id", e->id, "fused_ops",
                      perfetto::DynamicString(ops_str), event_lambda);
  } else {
    auto event_lambda = [e, base_lambda](perfetto::EventContext& ctx) {
      base_lambda(ctx);
      add_event_metadata(ctx, e);
    };
    TRACE_EVENT_BEGIN("events", perfetto::DynamicString(e->slice_name),
                      perfetto::Track(DPU_TRACK_ID), "id", e->id, event_lambda);
  }
}

void execution_end() {
  TRACE_EVENT_END("events", perfetto::Track(DPU_TRACK_ID));
}

void active_ops_counter(size_t count) {
  TRACE_COUNTER("queue", "Active DPU Ops", (int)count);
}

void ensure_callback_thread_named() {
  static thread_local bool thread_named = false;
  if (!thread_named) {
    auto track = perfetto::ThreadTrack::Current();
    auto desc = track.Serialize();
    desc.mutable_thread()->set_thread_name("UPMEM Callback");
    perfetto::TrackEvent::SetTrackDescriptor(track, desc);
    thread_named = true;
  }
}
}  // namespace trace

#endif
