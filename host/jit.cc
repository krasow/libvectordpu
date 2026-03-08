#include "jit.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "common.h"
#include "logger.h"
#include "opcodes.h"
#include "runtime.h"
#include "vectordpu.h"

std::string normalize_type_name(const char* name) {
  if (!name) return "int32_t";
  std::string tn(name);
  if (tn == "i" || tn == "int" || tn == "int32_t") return "int32_t";
  if (tn == "j" || tn == "uint32_t") return "uint32_t";
  if (tn == "f" || tn == "float") return "float";
  if (tn == "d" || tn == "double") return "double";
  if (tn == "l" || tn == "int64_t") return "int64_t";
  if (tn == "m" || tn == "uint64_t") return "uint64_t";
  if (tn == "x" || tn == "long long") return "int64_t";
  if (tn == "y" || tn == "unsigned long long") return "uint64_t";
  return tn;
}

#if JIT
#include <dlfcn.h>

#include <map>
#include <mutex>

#include "perfetto/trace.h"

namespace fs = std::filesystem;

namespace {
using Signature = std::tuple<std::vector<uint8_t>, std::string, std::string>;
using CacheKey = Signature;
std::map<std::vector<Signature>, std::string> g_jit_cache;
std::map<Signature, std::string> g_kernel_obj_cache;
std::mutex g_jit_cache_mutex;

std::string hash_signature(const Signature& sig) {
  size_t h = std::hash<std::string>{}(std::get<1>(sig));
  h ^= std::hash<std::string>{}(std::get<2>(sig)) + 0x9e3779b9 + (h << 6) + (h >> 2);
  for (uint8_t b : std::get<0>(sig)) {
    h ^= std::hash<uint8_t>{}(b) + 0x9e3779b9 + (h << 6) + (h >> 2);
  }
  char buf[32];
  sprintf(buf, "%016zx", h);
  return std::string(buf);
}
}  // namespace

// Anchor for dladdr
extern "C" void vectordpu_jit_dladdr_anchor() {}

static void write_kernel_header(std::ofstream& out) {
  out << "#include <stdint.h>\n";
  out << "#include <defs.h>\n";
  out << "#include <mram.h>\n";
  out << "#include <stdio.h>\n";
  out << "#include <barrier.h>\n";
  out << "#include <string.h>\n";
  out << "#include \"common.h\"\n\n";
}

static void write_dpu_main_header(std::ofstream& out, const std::string& stack_type) {
  out << "#include <alloc.h>\n";
  out << "#include <barrier.h>\n";
  out << "#include <defs.h>\n";
  out << "#include <mram.h>\n";
  out << "#include <stdint.h>\n";
  out << "#include <stdio.h>\n";
  out << "#include <stdlib.h>\n";
  out << "#include <string.h>\n";
  out << "#include \"common.h\"\n\n";

  out << "__host DPU_LAUNCH_ARGS args;\n";
  out << "BARRIER_INIT(my_barrier, NR_TASKLETS);\n";
  out << stack_type << " reduction_scratchpad[NR_TASKLETS] "
         "__attribute__((aligned(8)));\n\n";
  out << "__dma_aligned uint8_t "
         "dpu_workspace[NR_TASKLETS][TASKLET_WORKSPACE_SIZE];\n\n";
}

static void write_kernel_function(std::ofstream& out,
                                  const std::string& func_name,
                                  const std::vector<uint8_t>& rpn_ops,
                                  const std::string& output_type,
                                  const std::string& input_type) {
  fflush(stdout);
  std::string stack_type = input_type;

  out << "extern barrier_t my_barrier;\n";
  out << "extern " << output_type << " reduction_scratchpad[NR_TASKLETS];\n";
  out << "int " << func_name << "(void) {\n";
  out << "    unsigned int id = me();\n";
  out << "    uint32_t n = args.num_elements;\n";
  out << "    __mram_ptr " << input_type << " *in_ptr = (__mram_ptr "
      << input_type << " *)(args.pipeline.init_offset);\n";
  out << "    __mram_ptr " << output_type << " *rs_ptr = (__mram_ptr "
      << output_type << " *)(args.pipeline.res_offset);\n\n";

  // Setup Workspace pointers
  out << "    " << input_type << " *input_blk = (" << input_type
      << " *)dpu_workspace[id];\n";
  out << "    " << input_type << " *op_blks[MAX_WORKSPACE_OPERANDS];\n";
  out << "    for (int k = 0; k < MAX_WORKSPACE_OPERANDS; k++)\n";
  out << "      op_blks[k] = (" << input_type
      << " *)&dpu_workspace[id][(k + 1) * BLOCK_SIZE * MINIMUM_WRITE_SIZE];\n";

  // Reduction logic (detect reduction from last op)
  int num_reductions = 0;
  std::vector<uint8_t> reduction_ops;
  for (uint8_t op : rpn_ops) {
    if (IS_OP_REDUCTION(op)) {
      num_reductions++;
      reduction_ops.push_back(op);
    }
  }
  bool is_reduction = (num_reductions > 0);

  if (is_reduction) {
    out << "    " << output_type << " accs[" << num_reductions << "];\n";
    for (int k = 0; k < num_reductions; ++k) {
      switch (reduction_ops[k]) {
        case OP_SUM:
          out << "    accs[" << k << "] = 0;\n";
          break;
        case OP_PRODUCT:
          out << "    accs[" << k << "] = 1;\n";
          break;
        case OP_MIN:
          out << "    accs[" << k << "] = (" << stack_type << ")1e15;\n";
          break;
        case OP_MAX:
          out << "    accs[" << k << "] = (" << stack_type << ")-1e15;\n";
          break;
      }
    }
  }

  // Determine needed inputs
  bool uses_input = false;
  bool uses_op[MAX_PIPELINE_OPERANDS] = {false};
  int max_op_idx = -1;
  for (size_t i = 0; i < rpn_ops.size(); ++i) {
    uint8_t op = rpn_ops[i];
    if (op == OP_PUSH_INPUT)
      uses_input = true;
    else if (op >= OP_PUSH_OPERAND_0 && op <= OP_PUSH_OPERAND_7) {
      int idx = op - OP_PUSH_OPERAND_0;
      uses_op[idx] = true;
      if (idx > max_op_idx) max_op_idx = idx;
    } else if (IS_OP_SCALAR(op)) {
      i += 4;  // Skip scalar data
    } else if (IS_OP_SCALAR_VAR(op)) {
      i += 1;  // Skip scalar index
    }
  }
  int first_red_operand_idx = max_op_idx + 1;

  // Main Loop
  out << "    uint32_t blk, i, b_e, b_b;\n";
  out << "    for (blk = id << BLOCK_SIZE_LOG2; blk < n; blk += (NR_TASKLETS "
         "<< BLOCK_SIZE_LOG2)) {\n";
  out << "        b_e = (blk + BLOCK_SIZE >= n) ? (n - blk) : BLOCK_SIZE;\n";
  out << "        b_b = b_e * sizeof(" << input_type << ");\n\n";

  // Fetch Logic
  if (uses_input) {
    out << "        mram_read((__mram_ptr void const *)(in_ptr + blk), "
           "input_blk, b_b);\n";
  }
  for (int k = 0; k < MAX_WORKSPACE_OPERANDS; k++) {
    if (uses_op[k]) {
      out << "        {\n";
      out << "            __mram_ptr " << input_type << " *p = (__mram_ptr "
          << input_type << " *)(args.pipeline.binary_operands[" << k << "]);\n";
      out << "            mram_read((__mram_ptr void const *)(p + blk), "
             "op_blks["
          << k << "], b_b);\n";
      out << "        }\n";
    }
  }

  // Computation Loop
  out << "        " << input_type << " *w_out = (" << input_type
      << " *)op_blks[0]; // Reuse first operand block for intermediate output "
         "if needed\n";
  out << "        // Unrolled computation\n";
  out << "        for (i = 0; i < b_e; i++) {\n";

  // Stack simulation for code generation
  std::vector<std::string> stack;
  int tmp_var_counter = 0;

  auto get_tmp = [&]() { return "t" + std::to_string(tmp_var_counter++); };

  for (size_t i = 0; i < rpn_ops.size(); ++i) {
    uint8_t op = rpn_ops[i];
    if (op == OP_PUSH_INPUT) {
      stack.push_back("((" + stack_type + ")input_blk[i])");
    } else if (op >= OP_PUSH_OPERAND_0 && op <= OP_PUSH_OPERAND_11) {
      std::string idx = std::to_string(op - OP_PUSH_OPERAND_0);
      stack.push_back("((" + stack_type + ")op_blks[" + idx + "][i])");
    } else if (IS_OP_SCALAR(op)) {
      // Decode scalar
      int32_t val;
      uint8_t b0 = rpn_ops[i + 1];
      uint8_t b1 = rpn_ops[i + 2];
      uint8_t b2 = rpn_ops[i + 3];
      uint8_t b3 = rpn_ops[i + 4];
      val = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
      i += 4;

      std::string scalar_literal = std::to_string(val);
      if (stack.empty()) throw std::runtime_error("Stack underflow in JIT: scalar op");
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = get_tmp();
      out << "            " << stack_type << " " << res << " = ";

      switch (op) {
        case OP_ADD_SCALAR:
          out << s1 << " + (" << stack_type << ")" << scalar_literal << ";\n";
          break;
        case OP_SUB_SCALAR:
          out << s1 << " - (" << stack_type << ")" << scalar_literal << ";\n";
          break;
        case OP_MUL_SCALAR:
          out << s1 << " * (" << stack_type << ")" << scalar_literal << ";\n";
          break;
        case OP_DIV_SCALAR:
          out << s1 << " / (" << stack_type << ")" << scalar_literal << ";\n";
          break;
        case OP_ASR_SCALAR:
          out << s1 << " >> " << scalar_literal << ";\n";
          break;
      }
      stack.push_back(res);

    } else if (IS_OP_SCALAR_VAR(op)) {
      // Decode scalar index
      uint8_t idx = rpn_ops[i + 1];
      i += 1;

      std::string scalar_val = "args.pipeline.scalars[" + std::to_string(idx) + "]";
      if (stack.empty()) throw std::runtime_error("Stack underflow in JIT: scalar var op");
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = get_tmp();
      out << "            " << stack_type << " " << res << " = ";

      switch (op) {
        case OP_ADD_SCALAR_VAR:
          out << s1 << " + (" << stack_type << ")" << scalar_val << ";\n";
          break;
        case OP_SUB_SCALAR_VAR:
          out << s1 << " - (" << stack_type << ")" << scalar_val << ";\n";
          break;
        case OP_MUL_SCALAR_VAR:
          out << s1 << " * (" << stack_type << ")" << scalar_val << ";\n";
          break;
        case OP_DIV_SCALAR_VAR:
          out << s1 << " / (" << stack_type << ")" << scalar_val << ";\n";
          break;
        case OP_ASR_SCALAR_VAR:
          out << s1 << " >> " << scalar_val << ";\n";
          break;
      }
      stack.push_back(res);

    } else if (IS_OP_UNARY(op)) {
      if (stack.empty()) throw std::runtime_error("Stack underflow in JIT: unary op");
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = get_tmp();
      if (op == OP_NEGATE) {
        out << "            " << stack_type << " " << res << " = -" << s1
            << ";\n";
      } else if (op == OP_ABS) {
        out << "            " << stack_type << " " << res << ";\n";
        out << "            if (" << s1 << " < 0) " << res << " = -" << s1
            << ";\n";
        out << "            else " << res << " = " << s1 << ";\n";
      }
      stack.push_back(res);
    } else if (IS_OP_BINARY(op)) {
      if (stack.size() < 2) throw std::runtime_error("Stack underflow in JIT: binary op");
      std::string s2 = stack.back();
      stack.pop_back();
      std::string s1 = stack.back();
      stack.pop_back();
      std::string res = get_tmp();
      out << "            " << stack_type << " " << res << " = ";
      switch (op) {
        case OP_ADD:
          out << s1 << " + " << s2 << ";\n";
          break;
        case OP_SUB:
          out << s1 << " - " << s2 << ";\n";
          break;
        case OP_MUL:
          out << s1 << " * " << s2 << ";\n";
          break;
        case OP_DIV:
          out << s1 << " / " << s2 << ";\n";
          break;
        case OP_ASR:
          out << s1 << " >> " << s2 << ";\n";
          break;
      }
      stack.push_back(res);
    } else if (IS_OP_REDUCTION(op)) {
      // Actually we are emitting code inside the loop. 
      // We need to keep track of which reduction this is in the RPN sequence.
      // Let's count reductions encountered so far in this sequence.
      int red_idx = 0;
      for (size_t k = 0; k < i; ++k) {
        if (IS_OP_REDUCTION(rpn_ops[k])) red_idx++;
      }

      if (stack.empty()) throw std::runtime_error("Stack underflow in JIT: reduction op");
      std::string s = stack.back();
      stack.pop_back();
      switch (op) {
        case OP_SUM:
          out << "            accs[" << red_idx << "] += " << s << ";\n";
          break;
        case OP_PRODUCT:
          out << "            accs[" << red_idx << "] *= " << s << ";\n";
          break;
        case OP_MIN:
          out << "            if (" << s << " < accs[" << red_idx << "]) accs[" << red_idx << "] = " << s << ";\n";
          break;
        case OP_MAX:
          out << "            if (" << s << " > accs[" << red_idx << "]) accs[" << red_idx << "] = " << s << ";\n";
          break;
      }
    }
  }

  if (!stack.empty()) {
    out << "            w_out[i] = " << stack.back() << ";\n";
  }
  out << "        }\n";  // End compute loop

  // Write Back (BLOCK)
  out << "        if (args.pipeline.res_offset != 0) {\n";
  out << "            mram_write(w_out, (__mram_ptr void *)(rs_ptr + blk), "
         "b_b);\n";
  out << "        }\n";

  // Closing the main block loop
  out << "    }\n";  // End block loop

  // Reduction Writeback (replicated from pipeline.inl)
  if (is_reduction) {
    out << "    // Reduction Writeback\n";
    out << "    " << output_type << "* scratchpad = (" << output_type << "*)reduction_scratchpad;\n";
    out << "    for (int k = 0; k < " << num_reductions << "; ++k) {\n";
    out << "        scratchpad[id * " << num_reductions << " + k] = accs[k];\n";
    out << "    }\n";
    out << "    barrier_wait(&my_barrier);\n";
    out << "    if (id == 0) {\n";
    out << "        for (int k = 0; k < " << num_reductions << "; ++k) {\n";
    out << "            " << output_type << " tot = scratchpad[k];\n";
    out << "            for (int i = 1; i < NR_TASKLETS; i++) {\n";
    out << "              " << output_type << " v = scratchpad[i * " << num_reductions << " + k];\n";

    // Need to determine the op for EACH reduction from reduction_ops vector
    // This requires generating a switch inside the loop
    out << "              switch (k) {\n";
    for (int k = 0; k < num_reductions; ++k) {
        out << "                case " << k << ":\n";
        switch (reduction_ops[k]) {
            case OP_SUM: out << "                  tot += v; break;\n"; break;
            case OP_PRODUCT: out << "                  tot *= v; break;\n"; break;
            case OP_MIN: out << "                  if (v < tot) tot = v; break;\n"; break;
            case OP_MAX: out << "                  if (v > tot) tot = v; break;\n"; break;
        }
    }
            out << "              }\n";
    out << "            }\n";
    out << "            uint64_t res_val = 0;\n";
    out << "            *(" << output_type << "*)&res_val = tot;\n";
    out << "            mram_write(&res_val, (__mram_ptr void *)(args.pipeline.binary_operands[" << first_red_operand_idx << " + k]), 8);\n";
    out << "        }\n";
    out << "    }\n";
  }

  out << "    return 0;\n";
  out << "}\n\n";
}

static std::string get_include_flags() {
  Dl_info dl_info;
  void* fptr = (void*)&vectordpu_jit_dladdr_anchor;
  std::vector<std::string> include_dirs;

  if (dladdr(fptr, &dl_info) != 0 && dl_info.dli_fname != nullptr) {
    fs::path lib_path = fs::absolute(dl_info.dli_fname);
    fs::path base = lib_path.parent_path().parent_path();
    if (fs::exists(base / "include" / "vectordpu"))
      include_dirs.push_back((base / "include" / "vectordpu").string());
    if (fs::exists(base.parent_path() / "common"))
      include_dirs.push_back((base.parent_path() / "common").string());
    if (fs::exists(base / "common"))
      include_dirs.push_back((base / "common").string());
  }

  if (include_dirs.empty()) {
    include_dirs.push_back("include/vectordpu");
  }

  std::string include_flags;
  for (const auto& dir : include_dirs) {
    include_flags += " -I" + dir;
  }
  return include_flags;
}

static bool compile_dpu_source(const std::string& filepath,
                               const std::string& binpath, bool is_object,
                               const std::string& include_flags) {
  std::string cmd = "dpu-upmem-dpurte-clang -DNR_TASKLETS=" +
                    std::to_string(DpuRuntime::get().num_tasklets()) +
                    include_flags + " -O3 " + (is_object ? "-c " : "") + "-o " +
                    binpath + " " + filepath;

  int ret = system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "JIT Compilation failed: " << cmd << std::endl;
    return false;
  }

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[JIT] Compiled " << (is_object ? "object " : "kernel ")
                << "to " << binpath << std::endl;
#endif

  return true;
}

static bool link_dpu_objects(const std::string& main_path,
                             const std::vector<std::string>& objects,
                             const std::string& binpath,
                             const std::string& include_flags) {
  std::string cmd = "dpu-upmem-dpurte-clang -DNR_TASKLETS=" +
                    std::to_string(DpuRuntime::get().num_tasklets()) +
                    include_flags + " -O3 -o " + binpath + " " + main_path;
  for (const auto& obj : objects) {
    cmd += " " + obj;
  }

  int ret = system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "JIT Linking failed: " << cmd << std::endl;
    return false;
  }

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  logger.lock() << "[JIT] Linked binary to " << binpath << std::endl;
#endif
  return true;
}

std::string jit_compile(
    const std::vector<std::tuple<std::vector<uint8_t>, std::string, std::string>>&
        kernels) {
  static std::mutex jit_mtx;
  std::lock_guard<std::mutex> lock(jit_mtx);
  std::cout << std::flush;

#if ENABLE_DPU_LOGGING >= 1
  // Logger& logger = DpuRuntime::get().get_logger();
#endif

  {
    std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
    if (g_jit_cache.find(kernels) != g_jit_cache.end()) {
        return g_jit_cache[kernels];
    }
    printf("[JIT] Cache miss! Compiling new kernel batch of %zu kernels...\n", kernels.size());
    for (size_t i = 0; i < kernels.size(); ++i) {
        printf("    K%zu: %s\n", i, std::get<1>(kernels[i]).c_str());
    }
  }

  try {
  trace::jit_compile_begin(kernels);

  std::string include_flags = get_include_flags();
  std::string build_dir = "build/jit";
  fs::create_directories(build_dir);

  std::vector<std::string> object_files;
  for (size_t k_idx = 0; k_idx < kernels.size(); ++k_idx) {
    const auto& sig = kernels[k_idx];
    std::string obj_path;

    {
      std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
      if (g_kernel_obj_cache.count(sig)) {
        obj_path = g_kernel_obj_cache[sig];
      }
    }

    if (obj_path.empty()) {
      std::string hash = hash_signature(sig);
      std::string c_path = build_dir + "/k_" + hash + ".c";
      obj_path = build_dir + "/k_" + hash + ".o";
      std::ofstream out(c_path);
      write_kernel_header(out);
      write_kernel_function(out, "k_" + hash, std::get<0>(sig), std::get<1>(sig), std::get<2>(sig));
      out.close();

      if (!compile_dpu_source(c_path, obj_path, true, include_flags)) {
        trace::jit_compile_end();
        throw std::runtime_error("JIT Compilation failed for " + c_path);
      }

      {
        std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
        g_kernel_obj_cache[sig] = obj_path;
      }
    }
    object_files.push_back(obj_path);
  }

  static int binary_counter = 0;
  std::string main_c_path =
      build_dir + "/main_" + std::to_string(binary_counter++) + ".c";
  std::string binpath = main_c_path + ".dpu";

  std::string batch_stack_type = "int32_t";
  for (const auto& sig : kernels) {
    if (std::get<1>(sig) == "int64_t") {
      batch_stack_type = "int64_t";
      break;
    }
  }

  std::ofstream out(main_c_path);
  write_dpu_main_header(out, batch_stack_type);

  // extern declarations
  for (size_t k_idx = 0; k_idx < kernels.size(); ++k_idx) {
    std::string hash;
    {
      std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
      hash = hash_signature(kernels[k_idx]);
    }
    out << "extern int k_" << hash << "(void);\n";
  }

  out << "\nint main() {\n";
  out << "  switch (args.kernel) {\n";
  for (size_t k_idx = 0; k_idx < kernels.size(); ++k_idx) {
    std::string hash;
    {
      std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
      hash = hash_signature(kernels[k_idx]);
    }
    out << "    case " << k_idx << ": return k_"
        << hash << "();\n";
  }
  out << "    default: return -1;\n";
  out << "  }\n";
  out << "}\n";
  out.close();

  if (!link_dpu_objects(main_c_path, object_files, binpath, include_flags)) {
    trace::jit_compile_end();
    throw std::runtime_error("JIT Linking failed for " + binpath);
  }

  {
    std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
    g_jit_cache[kernels] = binpath;
  }
  trace::jit_compile_end();
  return binpath;
  } catch (const std::exception& e) {
    fprintf(stderr, "[JIT] EXCEPTION in jit_compile: %s\n", e.what());
    throw;
  } catch (...) {
    fprintf(stderr, "[JIT] UNKNOWN EXCEPTION in jit_compile\n");
    throw;
  }
}

void jit_cleanup() {
  std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
  return; // DEBUG: Keep JIT files
  std::string build_dir = "build/jit";
  if (fs::exists(build_dir)) {
    try {
      fs::remove_all(build_dir);
    } catch (...) {
      // Ignore cleanup errors during shutdown
    }
  }
}

#endif
