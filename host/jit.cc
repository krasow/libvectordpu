#include "jit.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "common.h"
#include "fusion.h"
#include "logger.h"
#include "opcodes.h"
#include "queue.h"
#include "runtime.h"
#include "vectordpu.h"

#if JIT
#include <dlfcn.h>

#include <map>
#include <mutex>

#include "perfetto/trace.h"

namespace fs = std::filesystem;

namespace {
using Signature = std::pair<std::vector<uint8_t>, std::string>;
using CacheKey  = std::vector<Signature>;
static constexpr uint32_t KERNEL_COUNT_VAL = JIT_STATIC_KERNEL_COUNT;
// Number of result slots: one primary + one per extra horizontal chain.
static constexpr int MAX_RESULT_SLOTS = MAX_HORIZONTAL_CHAINS + 1;
std::map<CacheKey, std::string> g_jit_cache;
std::map<Signature, std::string> g_kernel_obj_cache;
std::recursive_mutex g_jit_cache_mutex;

std::string hash_signature(const Signature& sig) {
  size_t h = std::hash<std::string>{}(sig.second);
  for (uint8_t b : sig.first)
    h ^= std::hash<uint8_t>{}(b) + 0x9e3779b9 + (h << 6) + (h >> 2);
  char buf[32];
  sprintf(buf, "%016zx", h);
  return std::string(buf);
}
}  // namespace

// Anchor for dladdr
extern "C" void vectordpu_jit_dladdr_anchor() {}

// Only valid inside write_kernel_function
// out, stack_type, res, s1, s2, rhs are in scope.
#define EMIT_BINOP(sym)     out << s1 << " " #sym " " << s2 << ";\n"; break
#define EMIT_SCALAROP(sym)  out << s1 << " " #sym " (" << stack_type << ")" << rhs << ";\n"; break
#define EMIT_SHIFTOP        out << s1 << " >> " << rhs << ";\n"; break

static void write_dpu_main_header(std::ofstream& out) {
  out << R"(#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

__host DPU_LAUNCH_ARGS args;
BARRIER_INIT(my_barrier, NR_TASKLETS);
// Scratchpad for cross-tasklet reduction: one slot per tasklet per chain.
// Oversized for alignment safety; actual usage is MAX_RESULT_SLOTS * NR_TASKLETS.
uint64_t reduction_scratchpad[NR_TASKLETS * 16] __attribute__((aligned(8)));
__dma_aligned uint8_t dpu_workspace[NR_TASKLETS][TASKLET_WORKSPACE_SIZE];

)";
}

static void write_kernel_function(std::ofstream& out,
                                  const std::string& func_name,
                                  const std::vector<uint8_t>& rpn_ops,
                                  const std::string& type_name) {
  std::string stack_type = type_name;
  // necessary to include these headers for the generated kernel code
  // each fused kernel is a separate compilation unit
  out << R"(#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "common.h"
extern barrier_t my_barrier;
extern uint64_t reduction_scratchpad[NR_TASKLETS * 16];

)";

  out << "int " << func_name << "(void) {\n"
      << "    unsigned int id = me();\n"
      << "    uint32_t n = args.num_elements;\n"
      << "    __mram_ptr " << type_name << " *in_ptr = (__mram_ptr "
      << type_name << " *)(args.pipeline.init_offset);\n\n";
  
  // Horizontal fusion result support
  // Result pointers: [0] = primary output, [1..MAX_HORIZONTAL_CHAINS] = extra chains.
  out << "    __mram_ptr " << type_name << " *res_ptrs[" << MAX_RESULT_SLOTS << "];\n"
      << "    res_ptrs[0] = (__mram_ptr " << type_name << " *)(args.pipeline.res_offset);\n";
  for (int i = 0; i < MAX_HORIZONTAL_CHAINS; ++i) {
    out << "    res_ptrs[" << (i + 1) << "] = (__mram_ptr " << type_name
        << " *)(args.pipeline.extra_res_offsets[" << i << "]);\n";
  }

  // WRAM workspace layout:
  //   slot 0:                        input_blk
  //   slots 1..MAX_PIPELINE_OPERANDS: op_blks[0..MAX_PIPELINE_OPERANDS-1]
  //   slots MAX_PIPELINE_OPERANDS+1..+MAX_RESULT_SLOTS: res_blks (reuse scratch slots)
  out << "\n    " << type_name << " *input_blk = (" << type_name << " *)dpu_workspace[id];\n"
      << "    " << type_name << " *op_blks[MAX_PIPELINE_OPERANDS];\n"
      << "    for (int k = 0; k < MAX_PIPELINE_OPERANDS; k++)\n"
      << "        op_blks[k] = (" << type_name
      << " *)&dpu_workspace[id][(k + 1) * BLOCK_SIZE * MINIMUM_WRITE_SIZE];\n"
      << "    " << type_name << " *res_blks[" << MAX_RESULT_SLOTS << "];\n"
      << "    for (int k = 0; k < " << MAX_RESULT_SLOTS << "; k++)\n"
      << "        res_blks[k] = (" << type_name
      << " *)&dpu_workspace[id][(1 + MAX_PIPELINE_OPERANDS + k) * BLOCK_SIZE * MINIMUM_WRITE_SIZE];\n\n";

  // Scan RPN to find which operand slots are needed and where chain boundaries are.
  bool uses_input = false;
  bool uses_op[MAX_PIPELINE_OPERANDS] = {false};
  struct Chain {
    size_t start_op, end_op;
    bool    is_reduction;
    uint8_t reduction_op;
  };
  std::vector<Chain> chains;
  size_t current_chain_start = 0;

  auto identify_chain = [&](size_t start, size_t end) {
    Chain c{start, end, false, 0};
    for (size_t i = start; i < end; ++i) {
      uint8_t op = rpn_ops[i];
      if      (op == OP_PUSH_INPUT) uses_input = true;
      else if (op >= OP_PUSH_OPERAND_0 && op < OP_PUSH_OPERAND_0 + MAX_PIPELINE_OPERANDS)
        uses_op[op - OP_PUSH_OPERAND_0] = true;
      else if (IS_OP_SCALAR(op))    i += SCALAR_INLINE_BYTES;
      else if (IS_OP_SCALAR_VAR(op)) i += SCALAR_VAR_INDEX_BYTES;
      else if (IS_OP_REDUCTION(op)) { c.is_reduction = true; c.reduction_op = op; }
    }
    chains.push_back(c);
  };

  for (size_t i = 0; i < rpn_ops.size(); ++i) {
    uint8_t op = rpn_ops[i];
    if      (op == OP_NEXT_CHAIN)    { identify_chain(current_chain_start, i); current_chain_start = i + 1; }
    else if (IS_OP_SCALAR(op))       i += SCALAR_INLINE_BYTES;
    else if (IS_OP_SCALAR_VAR(op))   i += SCALAR_VAR_INDEX_BYTES;
  }
  identify_chain(current_chain_start, rpn_ops.size());

#if ENABLE_PROMOTION_REDUCTIONS == 1
  for (const auto& c : chains)
    if (c.is_reduction && type_name == "int32_t") { stack_type = "int64_t"; break; }
#endif

  // Reduction accumulators with identity values.
  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (!chains[c_idx].is_reduction) continue;
    const bool is_float = (stack_type == "float");
    out << "    " << stack_type << " acc_" << c_idx << " = ";
    switch (chains[c_idx].reduction_op) {
      case OP_SUM:     out << "0;\n"; break;
      case OP_PRODUCT: out << "1;\n"; break;
      case OP_MIN:     out << (is_float ? "3.402823466e+38f" : "INT32_MAX")  << ";\n"; break;
      case OP_MAX:     out << (is_float ? "-3.402823466e+38f" : "INT32_MIN") << ";\n"; break;
    }
  }

  // Main per-block loop.
  out << "    uint32_t blk, i, b_e, b_b, b_b_aligned;\n"
      << "    for (blk = id << BLOCK_SIZE_LOG2; blk < n; blk += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {\n"
      << "        b_e = (blk + BLOCK_SIZE >= n) ? (n - blk) : BLOCK_SIZE;\n"
      << "        b_b = b_e * sizeof(" << type_name << ");\n"
      << "        b_b_aligned = (b_b + 7) & ~7;\n\n";

  if (uses_input)
    out << "        mram_read((__mram_ptr void const *)(in_ptr + blk), input_blk, b_b_aligned);\n";
  for (int k = 0; k < MAX_PIPELINE_OPERANDS; k++) {
    if (!uses_op[k]) continue;
    out << "        {\n"
        << "            __mram_ptr " << type_name << " *p = (__mram_ptr "
        << type_name << " *)(args.pipeline.binary_operands[" << k << "]);\n"
        << "            if (p) mram_read((__mram_ptr void const *)(p + blk), op_blks["
        << k << "], b_b_aligned);\n"
        << "        }\n";
  }

  out << "        for (i = 0; i < b_e; i++) {\n";

  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    const auto& chain = chains[c_idx];
    out << "            // Chain " << c_idx << "\n";

    std::vector<std::string> stack;
    int tmp_n = 0;
    auto get_tmp = [&]() {
      return "t_" + std::to_string(c_idx) + "_" + std::to_string(tmp_n++);
    };

    for (size_t op_idx = chain.start_op; op_idx < chain.end_op; ++op_idx) {
      uint8_t op = rpn_ops[op_idx];

      if (op == OP_PUSH_INPUT) {
        stack.push_back("((" + stack_type + ")input_blk[i])");

      } else if (op >= OP_PUSH_OPERAND_0 && op < OP_PUSH_OPERAND_0 + MAX_PIPELINE_OPERANDS) {
        stack.push_back("((" + stack_type + ")op_blks["
                        + std::to_string(op - OP_PUSH_OPERAND_0) + "][i])");

      } else if (IS_OP_SCALAR(op) || IS_OP_SCALAR_VAR(op)) {
        std::string rhs;
        if (IS_OP_SCALAR(op)) {
          uint8_t b0 = rpn_ops[op_idx+1], b1 = rpn_ops[op_idx+2],
                  b2 = rpn_ops[op_idx+3], b3 = rpn_ops[op_idx+4];
          int32_t val = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
          op_idx += SCALAR_INLINE_BYTES;
          rhs = std::to_string(val);
        } else {
          uint8_t idx = rpn_ops[op_idx + 1];
          op_idx += SCALAR_VAR_INDEX_BYTES;
          rhs = "args.pipeline.scalars[" + std::to_string(idx) + "]";
        }
        std::string s1 = stack.back(); stack.pop_back();
        std::string res = get_tmp();
        out << "            " << stack_type << " " << res << " = ";
        // Normalize SCALAR_VAR opcode to the equivalent SCALAR opcode for a
        // unified switch; both forms share the same operator symbol.
        uint8_t base = IS_OP_SCALAR_VAR(op) ? (op - (OP_ADD_SCALAR_VAR - OP_ADD_SCALAR)) : op;
        switch (base) {
          case OP_ADD_SCALAR: EMIT_SCALAROP(+);
          case OP_SUB_SCALAR: EMIT_SCALAROP(-);
          case OP_MUL_SCALAR: EMIT_SCALAROP(*);
          case OP_DIV_SCALAR: EMIT_SCALAROP(/);
          case OP_ASR_SCALAR: EMIT_SHIFTOP;
          case OP_EQ_SCALAR:  EMIT_SCALAROP(==);
          case OP_LT_SCALAR:  EMIT_SCALAROP(<);
          case OP_GT_SCALAR:  EMIT_SCALAROP(>);
          case OP_GE_SCALAR:  EMIT_SCALAROP(>=);
          case OP_LE_SCALAR:  EMIT_SCALAROP(<=);
        }
        stack.push_back(res);

      } else if (op == OP_DUP) {
        stack.push_back(stack.back());

      } else if (IS_OP_UNARY(op)) {
        std::string s1 = stack.back(); stack.pop_back();
        std::string res = get_tmp();
        if (op == OP_NEGATE)
          out << "            " << stack_type << " " << res << " = -" << s1 << ";\n";
        else if (op == OP_ABS)
          out << "            " << stack_type << " " << res
              << " = (" << s1 << " < 0) ? -" << s1 << " : " << s1 << ";\n";
        stack.push_back(res);

      } else if (IS_OP_BINARY(op)) {
        if (stack.size() < 2) {
          fprintf(stderr, "[JIT-DBG] STACK UNDERFLOW at binary op %u, stack size=%zu\n",
                  (unsigned)op, stack.size());
          abort();
        }
        std::string s2 = stack.back(); stack.pop_back();
        std::string s1 = stack.back(); stack.pop_back();
        std::string res = get_tmp();
        out << "            " << stack_type << " " << res << " = ";
        switch (op) {
          case OP_ADD: EMIT_BINOP(+);
          case OP_SUB: EMIT_BINOP(-);
          case OP_MUL: EMIT_BINOP(*);
          case OP_DIV: EMIT_BINOP(/);
          case OP_ASR: out << s1 << " >> " << s2 << ";\n"; break;
          case OP_EQ:  EMIT_BINOP(==);
          case OP_LT:  EMIT_BINOP(<);
          case OP_GT:  EMIT_BINOP(>);
          case OP_GE:  EMIT_BINOP(>=);
          case OP_LE:  EMIT_BINOP(<=);
        }
        stack.push_back(res);

      } else if (IS_OP_TERNARY(op)) {
        std::string s1 = stack.back(); stack.pop_back();
        std::string s2 = stack.back(); stack.pop_back();
        std::string s3 = stack.back(); stack.pop_back();
        std::string res = get_tmp();
        if (op == OP_SELECT)
          out << "            " << stack_type << " " << res
              << " = (" << s3 << " != 0) ? " << s2 << " : " << s1 << ";\n";
        stack.push_back(res);

      } else if (IS_OP_REDUCTION(op)) {
        std::string s = stack.back(); stack.pop_back();
        switch (op) {
          case OP_SUM:     out << "            acc_" << c_idx << " += " << s << ";\n"; break;
          case OP_PRODUCT: out << "            acc_" << c_idx << " *= " << s << ";\n"; break;
          case OP_MIN:     out << "            if (" << s << " < acc_" << c_idx << ") acc_" << c_idx << " = " << s << ";\n"; break;
          case OP_MAX:     out << "            if (" << s << " > acc_" << c_idx << ") acc_" << c_idx << " = " << s << ";\n"; break;
        }
      }
    }  // op_idx

    if (!chain.is_reduction && !stack.empty())
      out << "            res_blks[" << c_idx << "][i] = " << stack.back() << ";\n";
  }  // c_idx

  out << "        }\n";  // end inner element loop

  // Write computed blocks back to MRAM for non-reduction chains.
  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (chains[c_idx].is_reduction) continue;
    out << "        if (res_ptrs[" << c_idx << "])\n"
        << "            mram_write(res_blks[" << c_idx
        << "], (__mram_ptr void *)(res_ptrs[" << c_idx << "] + blk), b_b_aligned);\n";
  }
  out << "    }\n";  // end block loop

  // Cross-tasklet reduction: each tasklet writes its partial result to the
  // scratchpad, then tasklet 0 reduces across all tasklets and writes to MRAM.
  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (!chains[c_idx].is_reduction) continue;
    const size_t scratchpad_row = c_idx;
    const bool is_first_chain   = (c_idx == 0);
    out << "    {\n"
        << "        uint64_t bf_scratch = 0;\n"
        << "        memcpy(&bf_scratch, &acc_" << c_idx << ", sizeof(" << stack_type << "));\n"
        << "        reduction_scratchpad[" << scratchpad_row << " * NR_TASKLETS + id] = bf_scratch;\n"
        << "        barrier_wait(&my_barrier);\n"
        << "        if (id == 0) {\n"
        << "            uint64_t tot_raw = reduction_scratchpad[" << scratchpad_row << " * NR_TASKLETS + 0];\n"
        << "            " << stack_type << " tot;\n"
        << "            memcpy(&tot, &tot_raw, sizeof(" << stack_type << "));\n"
        << "            for (int k = 1; k < NR_TASKLETS; k++) {\n"
        << "                uint64_t v_raw = reduction_scratchpad[" << scratchpad_row << " * NR_TASKLETS + k];\n"
        << "                " << stack_type << " v;\n"
        << "                memcpy(&v, &v_raw, sizeof(" << stack_type << "));\n";
    switch (chains[c_idx].reduction_op) {
      case OP_SUM:     out << "                tot += v;\n"; break;
      case OP_PRODUCT: out << "                tot *= v;\n"; break;
      case OP_MIN:     out << "                if (v < tot) tot = v;\n"; break;
      case OP_MAX:     out << "                if (v > tot) tot = v;\n"; break;
    }
    out << "            }\n"
        << "            uint64_t bf_final = 0;\n"
        << "            memcpy(&bf_final, &tot, sizeof(" << stack_type << "));\n";
    if (is_first_chain)
      out << "            mram_write(&bf_final, (__mram_ptr void *)args.pipeline.res_offset, "
          << MINIMUM_WRITE_SIZE << ");\n";
    else
      out << "            mram_write(&bf_final, (__mram_ptr void *)args.pipeline.extra_res_offsets["
          << (c_idx - 1) << "], " << MINIMUM_WRITE_SIZE << ");\n";
    out << "        }\n"
        << "        barrier_wait(&my_barrier);\n"
        << "    }\n";
  }

  out << "    return 0;\n}\n\n";
}

#undef EMIT_BINOP
#undef EMIT_SCALAROP
#undef EMIT_SHIFTOP

static std::string get_include_flags() {
  Dl_info dl_info;
  void* fptr = (void*)&vectordpu_jit_dladdr_anchor;
  std::vector<std::string> include_dirs;

  if (dladdr(fptr, &dl_info) != 0) {
    fs::path lib_path = fs::absolute(dl_info.dli_fname);
    fs::path base = lib_path.parent_path().parent_path();
    if (fs::exists(base / "include" / "vectordpu"))
      include_dirs.push_back((base / "include" / "vectordpu").string());
    if (fs::exists(base.parent_path() / "common"))
      include_dirs.push_back((base.parent_path() / "common").string());
    if (fs::exists(base / "common"))
      include_dirs.push_back((base / "common").string());
  }

  if (include_dirs.empty())
    include_dirs.push_back("include/vectordpu");

  std::string flags;
  for (const auto& dir : include_dirs) flags += " -I" + dir;
  return flags;
}

static bool compile_dpu_source(const std::string& filepath,
                               const std::string& binpath, bool is_object,
                               const std::string& include_flags) {
  std::string cmd = "dpu-upmem-dpurte-clang -DNR_TASKLETS="
      + std::to_string(DpuRuntime::get().num_tasklets())
      + include_flags + " -O3 " + (is_object ? "-c " : "") + "-o "
      + binpath + " " + filepath;

  if (system(cmd.c_str()) != 0) {
    std::cerr << "JIT Compilation failed: " << cmd << std::endl;
    return false;
  }
#if ENABLE_DPU_LOGGING >= 1
  DpuRuntime::get().get_logger().lock()
      << "[JIT] Compiled " << (is_object ? "object " : "kernel ")
      << "to " << binpath << std::endl;
#endif
  return true;
}

static bool link_dpu_objects(const std::string& main_path,
                             const std::vector<std::string>& objects,
                             const std::string& binpath,
                             const std::string& include_flags) {
  std::string cmd = "dpu-upmem-dpurte-clang -DNR_TASKLETS="
      + std::to_string(DpuRuntime::get().num_tasklets())
      + include_flags + " -O3 -o " + binpath + " " + main_path;
  for (const auto& obj : objects) cmd += " " + obj;

  if (system(cmd.c_str()) != 0) {
    std::cerr << "JIT Linking failed: " << cmd << std::endl;
    return false;
  }
#if ENABLE_DPU_LOGGING >= 1
  DpuRuntime::get().get_logger().lock()
      << "[JIT] Linked binary to " << binpath << std::endl;
#endif
  return true;
}

std::string jit_compile(
    const std::vector<std::pair<std::vector<uint8_t>, std::string>>& kernels) {
  std::cout << std::flush;

  {
    std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
    auto it = g_jit_cache.find(kernels);
    if (it != g_jit_cache.end()) {
#if ENABLE_DPU_LOGGING >= 1
      DpuRuntime::get().get_logger().lock()
          << "[JIT] Cache hit for batched binary with "
          << kernels.size() << " sub-kernels" << std::endl;
#endif
      return it->second;
    }
  }

  trace::jit_compile_begin(kernels);

  const std::string include_flags = get_include_flags();
  const std::string build_dir     = "build/jit";
  fs::create_directories(build_dir);

  // Compile each unique kernel to an object file (cached per signature).
  std::vector<std::string> object_files;
  for (const auto& sig : kernels) {
    std::string obj_path;
    {
      std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
      auto it = g_kernel_obj_cache.find(sig);
      if (it != g_kernel_obj_cache.end()) obj_path = it->second;
    }

    if (obj_path.empty()) {
      const std::string hash   = hash_signature(sig);
      const std::string c_path = build_dir + "/k_" + hash + ".c";
      obj_path                 = build_dir + "/k_" + hash + ".o";

      std::ofstream out(c_path);
      write_kernel_function(out, "k_" + hash, sig.first, sig.second);
      out.close();

      if (!compile_dpu_source(c_path, obj_path, true, include_flags)) {
        trace::jit_compile_end();
        throw std::runtime_error("JIT Compilation failed for " + c_path);
      }
      {
        std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
        g_kernel_obj_cache[sig] = obj_path;
      }
    }
    object_files.push_back(obj_path);
  }

  // Generate a main() that dispatches on args.kernel to the right sub-kernel.
  static int binary_counter = 0;
  const std::string main_c_path = build_dir + "/main_" + std::to_string(binary_counter++) + ".c";
  const std::string binpath     = main_c_path + ".dpu";

  {
    std::ofstream out(main_c_path);
    write_dpu_main_header(out);
    for (size_t k = 0; k < kernels.size(); ++k) {
      std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
      out << "extern int k_" << hash_signature(kernels[k]) << "(void);\n";
    }
    out << "\nint main() {\n  switch (args.kernel) {\n";
    for (size_t k = 0; k < kernels.size(); ++k) {
      std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
      out << "    case " << (KERNEL_COUNT_VAL + k) << ": return k_"
          << hash_signature(kernels[k]) << "();\n";
    }
    out << "    default: return -1;\n  }\n}\n";
  }

  if (!link_dpu_objects(main_c_path, object_files, binpath, include_flags)) {
    trace::jit_compile_end();
    throw std::runtime_error("JIT Linking failed for " + binpath);
  }

  {
    std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
    g_jit_cache[kernels] = binpath;
  }
  trace::jit_compile_end();
  return binpath;
}

void EventQueue::flush_jit_batch() {
  if (pending_unique_kernels_.empty()) return;

  std::vector<std::pair<std::vector<uint8_t>, std::string>> batch =
      pending_unique_kernels_;

#if ENABLE_DPU_LOGGING >= 1
  DpuRuntime::get().get_logger().lock()
      << "[queue-jit] Flushing " << batch.size()
      << " kernels to async JIT compiler." << std::endl;
#endif

  std::shared_future<std::string> future =
      std::async(std::launch::deferred, [batch]() { return jit_compile(batch); });
  for (auto& ev : pending_jit_events_) ev->jit_future = future;

  pending_jit_events_.clear();
  pending_unique_kernels_.clear();
}

void EventQueue::lock_for_jit(std::shared_ptr<Event> e) {
  if (e->op != Event::OperationType::COMPUTE || e->is_locked_for_jit) return;
  e->is_locked_for_jit = true;

  if (e->rpn_ops.empty()) {
    e->rpn_ops.push_back(OP_PUSH_INPUT);
    if (e->is_scalar) {
      e->rpn_ops.push_back(map_to_var_op(e->opcode));
      e->rpn_ops.push_back(0);
      e->scalars.push_back(e->scalar_value);
    } else {
      if (e->inputs.size() > 1) e->rpn_ops.push_back(OP_PUSH_OPERAND_0);
      e->rpn_ops.push_back(e->opcode);
    }
  }

  const char* type_name = "int32_t";
  if (e->output && e->output->type_name) {
    std::string tn = e->output->type_name;
    if      (tn == "i" || tn == "int")      type_name = "int32_t";
    else if (tn == "j" || tn == "uint32_t") type_name = "uint32_t";
    else if (tn == "f" || tn == "float")    type_name = "float";
    else if (tn == "d" || tn == "double")   type_name = "double";
    else                                    type_name = e->output->type_name;
  }

  Signature sig = {e->rpn_ops, type_name};

  // Check if this signature already has a slot in the current batch.
  for (size_t i = 0; i < pending_unique_kernels_.size(); ++i) {
    if (pending_unique_kernels_[i] == sig) {
      e->jit_sub_kernel_idx = i;
      pending_jit_events_.push_back(e);
      if (pending_jit_events_.size() >= MAX_JIT_QUEUE_DEPTH) flush_jit_batch();
      return;
    }
  }

  // New unique kernel — start a fresh epoch if the batch is full.
  if (pending_unique_kernels_.size() >= MAX_JIT_QUEUE_DEPTH) {
    flush_jit_batch();
    pending_unique_kernels_.clear();
  }

  e->jit_sub_kernel_idx = pending_unique_kernels_.size();
  pending_unique_kernels_.push_back(sig);
  pending_jit_events_.push_back(e);
  if (pending_jit_events_.size() >= MAX_JIT_QUEUE_DEPTH) flush_jit_batch();
}

void jit_cleanup() {
  std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
#if DEBUG_KEEP_JIT_DIR
  return;
#endif
  const std::string build_dir = "build/jit";
  if (fs::exists(build_dir)) {
    try { fs::remove_all(build_dir); } catch (...) {}
  }
}

#endif  // JIT
