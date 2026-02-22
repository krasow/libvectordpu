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

#if JIT
#include <dlfcn.h>

#include <map>
#include <mutex>

#include "perfetto/trace.h"

namespace fs = std::filesystem;

namespace {
using Signature = std::pair<std::vector<uint8_t>, std::string>;
using CacheKey = std::vector<Signature>;
std::map<CacheKey, std::string> g_jit_cache;
std::map<Signature, std::string> g_kernel_obj_cache;
std::mutex g_jit_cache_mutex;

std::string hash_signature(const Signature& sig) {
  size_t h = std::hash<std::string>{}(sig.second);
  for (uint8_t b : sig.first) {
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
  out << "#include \"common.h\"\n\n";
}

static void write_dpu_main_header(std::ofstream& out) {
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
  out << "uint64_t reduction_scratchpad[NR_TASKLETS] "
         "__attribute__((aligned(8)));\n\n";
  out << "#define TASKLET_WORKSPACE_SIZE (8 * BLOCK_SIZE * "
         "MINIMUM_WRITE_SIZE)\n";
  out << "__dma_aligned uint8_t "
         "dpu_workspace[NR_TASKLETS][TASKLET_WORKSPACE_SIZE];\n\n";
}

static void write_kernel_function(std::ofstream& out,
                                  const std::string& func_name,
                                  const std::vector<uint8_t>& rpn_ops,
                                  const std::string& type_name) {
  std::string stack_type = type_name;

  out << "#include <mram.h>\n";
  out << "#include <defs.h>\n";
  out << "#include <stdio.h>\n";
  out << "#include <barrier.h>\n";
  out << "#include <string.h>\n";
  out << "#include \"common.h\"\n";
  out << "extern barrier_t my_barrier;\n";
  out << "extern uint64_t reduction_scratchpad[NR_TASKLETS];\n\n";
  out << "int " << func_name << "(void) {\n";
  out << "    unsigned int id = me();\n";
  out << "    uint32_t n = args.num_elements;\n";
  out << "    __mram_ptr " << type_name << " *in_ptr = (__mram_ptr "
      << type_name << " *)(args.pipeline.init_offset);\n";
  out << "    __mram_ptr " << type_name << " *rs_ptr = (__mram_ptr "
      << type_name << " *)(args.pipeline.res_offset);\n\n";

  // Setup Workspace pointers
  out << "    " << type_name << " *input_blk = (" << type_name
      << " *)dpu_workspace[id];\n";
  out << "    " << type_name << " *op_blks[MAX_PIPELINE_OPERANDS];\n";
  out << "    for (int k = 0; k < MAX_PIPELINE_OPERANDS; k++)\n";
  out << "      op_blks[k] = (" << type_name
      << " *)&dpu_workspace[id][(k + 1) * BLOCK_SIZE * MINIMUM_WRITE_SIZE];\n";

  // Reduction logic (detect reduction from last op)
  bool is_reduction = false;
  uint8_t reduction_op = 0;
  if (!rpn_ops.empty() && IS_OP_REDUCTION(rpn_ops.back())) {
    is_reduction = true;
    reduction_op = rpn_ops.back();
#if ENABLE_PROMOTION_REDUCTIONS == 1
    if (type_name == "int32_t") stack_type = "int64_t";
#endif
  }

  if (is_reduction) {
    out << "    " << stack_type << " acc;\n";
    switch (reduction_op) {
      case OP_SUM:
        out << "    acc = 0;\n";
        break;
      case OP_PRODUCT:
        out << "    acc = 1;\n";
        break;
      case OP_MIN:
        out << "    acc = (" << stack_type << ")1e9;\n";
        break;
      case OP_MAX:
        out << "    acc = (" << stack_type << ")-1e9;\n";
        break;
    }
  }

  // Determine needed inputs
  bool uses_input = false;
  bool uses_op[8] = {false};
  for (size_t i = 0; i < rpn_ops.size(); ++i) {
    uint8_t op = rpn_ops[i];
    if (op == OP_PUSH_INPUT)
      uses_input = true;
    else if (op >= OP_PUSH_OPERAND_0 && op <= OP_PUSH_OPERAND_7) {
      uses_op[op - OP_PUSH_OPERAND_0] = true;
    } else if (IS_OP_SCALAR(op)) {
      i += 4;  // Skip scalar data
    }
  }

  // Main Loop
  out << "    uint32_t blk, i, b_e, b_b;\n";
  out << "    for (blk = id << BLOCK_SIZE_LOG2; blk < n; blk += (NR_TASKLETS "
         "<< BLOCK_SIZE_LOG2)) {\n";
  out << "        b_e = (blk + BLOCK_SIZE >= n) ? (n - blk) : BLOCK_SIZE;\n";
  out << "        b_b = b_e * sizeof(" << type_name << ");\n\n";

  // Fetch Logic
  if (uses_input) {
    out << "        mram_read((__mram_ptr void const *)(in_ptr + blk), "
           "input_blk, b_b);\n";
  }
  for (int k = 0; k < 3; k++) {  // Max 3 operands per design
    if (uses_op[k]) {
      out << "        {\n";
      out << "            __mram_ptr " << type_name << " *p = (__mram_ptr "
          << type_name << " *)(args.pipeline.binary_operands[" << k << "]);\n";
      out << "            mram_read((__mram_ptr void const *)(p + blk), "
             "op_blks["
          << k << "], b_b);\n";
      out << "        }\n";
    }
  }

  // Computation Loop
  out << "        " << type_name << " *w_out = (" << type_name
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
    } else if (op >= OP_PUSH_OPERAND_0 && op <= OP_PUSH_OPERAND_7) {
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

    } else if (IS_OP_UNARY(op)) {
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
      // Reduction consumes the stack top
      std::string s = stack.back();
      stack.pop_back();
      switch (op) {
        case OP_SUM:
          out << "            acc += " << s << ";\n";
          break;
        case OP_PRODUCT:
          out << "            acc *= " << s << ";\n";
          break;
        case OP_MIN:
          out << "            if (" << s << " < acc) acc = " << s << ";\n";
          break;
        case OP_MAX:
          out << "            if (" << s << " > acc) acc = " << s << ";\n";
          break;
      }
    }
  }

  if (!is_reduction && !stack.empty()) {
    out << "            w_out[i] = " << stack.back() << ";\n";
  }
  out << "        }\n";  // End compute loop

  // Write Back (BLOCK)
  if (!is_reduction && !stack.empty()) {
    out << "        mram_write(w_out, (__mram_ptr void *)(rs_ptr + blk), "
           "b_b);\n";
  }

  // Closing the main block loop
  out << "    }\n";  // End block loop

  // Reduction Writeback (replicated from pipeline.inl)
  if (is_reduction) {
    out << "    // Reduction Writeback\n";
    out << "    enum { sd = (MINIMUM_WRITE_SIZE / sizeof(" << stack_type
        << ")) };\n";
    out << "    uint64_t bf_scratch = 0;\n";
    out << "    memcpy(&bf_scratch, &acc, sizeof(" << stack_type << "));\n";
    out << "    reduction_scratchpad[id] = bf_scratch;\n";
    out << "    barrier_wait(&my_barrier);\n";
    out << "    if (id == 0) {\n";
    out << "        " << stack_type << " tot = (" << stack_type << ")*("<< stack_type << "*)&reduction_scratchpad[0];\n";
    out << "        for (int i = 1; i < NR_TASKLETS; i++) {\n";
    out << "          " << stack_type << " v = *(" << stack_type << "*)&reduction_scratchpad[i];\n";

    switch (reduction_op) {
      case OP_SUM:
        out << "          tot += v;\n";
        break;
      case OP_PRODUCT:
        out << "          tot *= v;\n";
        break;
      case OP_MIN:
        out << "          if (v < tot) tot = v;\n";
        break;
      case OP_MAX:
        out << "          if (v > tot) tot = v;\n";
        break;
    }
    out << "        }\n";
    out << "        " << stack_type << " bf_final = 0;\n";
    out << "        memcpy(&bf_final, &tot, sizeof(" << stack_type << "));\n";
    out << "        mram_write(&bf_final, (__mram_ptr void *)rs_ptr, "
           "MINIMUM_WRITE_SIZE);\n";
    out << "    }\n";
  }

  out << "    return 0;\n";
  out << "}\n\n";
}

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
    const std::vector<std::pair<std::vector<uint8_t>, std::string>>& kernels) {
  std::cout << std::flush;

#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
#endif

  {
    std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
    if (g_jit_cache.find(kernels) != g_jit_cache.end()) {
#if ENABLE_DPU_LOGGING >= 1
      logger.lock() << "[JIT] Cache hit for batched binary with "
                    << kernels.size() << " sub-kernels" << std::endl;
#endif
      return g_jit_cache[kernels];
    }
  }

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
      write_kernel_function(out, "k_" + hash, sig.first, sig.second);
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

  std::ofstream out(main_c_path);
  write_dpu_main_header(out);

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
    std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
    out << "    case " << k_idx << ": return k_"
        << hash_signature(kernels[k_idx]) << "();\n";
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
}

void jit_cleanup() {
  std::lock_guard<std::mutex> lock(g_jit_cache_mutex);
#if DEBUG_KEEP_JIT_DIR
  return;
#endif
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
