#include "jit.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "common.h"
#include "opcodes.h"
#include "vectordpu.h"
#include "runtime.h"
#include "logger.h"

#if JIT
#include <dlfcn.h>

namespace fs = std::filesystem;

#include <map>

// Anchor for dladdr
extern "C" void vectordpu_jit_dladdr_anchor() {}

std::string jit_compile(const std::vector<uint8_t>& rpn_ops,
                        const char* type_name) {
  // Cache Key: RPN sequence + Type name
  using CacheKey = std::pair<std::vector<uint8_t>, std::string>;
  static std::map<CacheKey, std::string> jit_cache;

  std::cout << std::flush;
  
#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
#endif

  CacheKey key = {rpn_ops, std::string(type_name)};
  if (jit_cache.find(key) != jit_cache.end()) {
#if ENABLE_DPU_LOGGING >= 1
      logger.lock() << "[JIT] Cache hit for kernel type=" << type_name << std::endl;
#endif
    return jit_cache[key];
  }

  // 1. Generate unique filename based on ops hash or simple counter
  // For simplicity, using a static counter or hash
  static int counter = 0;
  std::stringstream ss_fn;
  ss_fn << "jit_kernel_" << counter++ << ".c";
  std::string filename = ss_fn.str();
  std::string build_dir = "build/jit";
  fs::create_directories(build_dir);
  std::string filepath = build_dir + "/" + filename;
  std::string binpath = build_dir + "/" + filename + ".dpu";

  // Check if binary already exists (optional optimization, skip for now)

  std::ofstream out(filepath);

  // 2. Write Header
  out << "#include <alloc.h>\n";
  out << "#include <barrier.h>\n";
  out << "#include <defs.h>\n";
  out << "#include <mram.h>\n";
  out << "#include <stdint.h>\n";
  out << "#include <stdio.h>\n";
  out << "#include <string.h>\n";
  out << "#include \"common.h\"\n\n";

  out << "__host DPU_LAUNCH_ARGS args;\n";
  out << "BARRIER_INIT(my_barrier, NR_TASKLETS);\n\n";
  out << "#define TASKLET_WORKSPACE_SIZE (8 * BLOCK_SIZE * "
         "MINIMUM_WRITE_SIZE)\n";
  out << "__dma_aligned uint8_t "
         "dpu_workspace[NR_TASKLETS][TASKLET_WORKSPACE_SIZE];\n\n";

  // 3. Write Kernel
  out << "int jit_main_kernel(void) {\n";
  out << "    unsigned int id = me();\n";
  out << "    uint32_t n = args.num_elements;\n";
  out << "    " << type_name << " *in_ptr = (" << type_name
      << " *)(args.pipeline.init_offset);\n";
  out << "    " << type_name << " *rs_ptr = (" << type_name
      << " *)(args.pipeline.res_offset);\n\n";

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
  }

  if (is_reduction) {
    out << "    " << type_name << " acc;\n";
    switch (reduction_op) {
      case OP_SUM:
        out << "    acc = 0;\n";
        break;
      case OP_PRODUCT:
        out << "    acc = 1;\n";
        break;
      case OP_MIN:
        out << "    acc = (" << type_name << ")1e9;\n";
        break;  // TODO: limits
      case OP_MAX:
        out << "    acc = (" << type_name << ")-1e9;\n";
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
      stack.push_back("input_blk[i]");
    } else if (op >= OP_PUSH_OPERAND_0 && op <= OP_PUSH_OPERAND_7) {
      std::string idx = std::to_string(op - OP_PUSH_OPERAND_0);
      stack.push_back("op_blks[" + idx + "][i]");
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
      out << "            " << type_name << " " << res << " = ";

      switch (op) {
        case OP_ADD_SCALAR:
          out << s1 << " + " << scalar_literal << ";\n";
          break;
        case OP_SUB_SCALAR:
          out << s1 << " - " << scalar_literal << ";\n";
          break;
        case OP_MUL_SCALAR:
          out << s1 << " * " << scalar_literal << ";\n";
          break;
        case OP_DIV_SCALAR:
          out << s1 << " / " << scalar_literal << ";\n";
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
        out << "            " << type_name << " " << res << " = -" << s1
            << ";\n";
      } else if (op == OP_ABS) {
        out << "            " << type_name << " " << res << ";\n";
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
      out << "            " << type_name << " " << res << " = ";
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
    out << "    enum { sd = (MINIMUM_WRITE_SIZE / sizeof(" << type_name
        << ")) };\n";
    out << "    uint64_t bf = 0;\n";
    out << "    memcpy(&bf, &acc, sizeof(" << type_name << "));\n";
    out << "    mram_write((void *)&bf, (__mram_ptr uint64_t *)rs_ptr + id, "
           "MINIMUM_WRITE_SIZE);\n";
    out << "    barrier_wait(&my_barrier);\n";
    out << "    if (id == 0) {\n";
    out << "        " << type_name << " rb[NR_TASKLETS * sd];\n";
    out << "        mram_read((__mram_ptr void const *)rs_ptr, rb, "
           "NR_TASKLETS * MINIMUM_WRITE_SIZE);\n";
    out << "        " << type_name << " tot = rb[0];\n";
    out << "        for (i = 1; i < NR_TASKLETS; i++) {\n";
    out << "          " << type_name << " v = rb[i * sd];\n";

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
    out << "        bf = 0;\n";
    out << "        memcpy(&bf, &tot, sizeof(" << type_name << "));\n";
    out << "        mram_write(&bf, (__mram_ptr void *)rs_ptr, "
           "MINIMUM_WRITE_SIZE);\n";
    out << "    }\n";
  }

  out << "    return 0;\n";
  out << "}\n\n";

  out << "int main() { return jit_main_kernel(); }\n";

  out.close();

  // 4. Compile
  // Resolve include path relative to library
  Dl_info dl_info;
  void* fptr = (void*)&vectordpu_jit_dladdr_anchor;
  std::vector<std::string> include_dirs;

  if (dladdr(fptr, &dl_info) != 0) {
      fs::path lib_path = fs::absolute(dl_info.dli_fname);
      fs::path base = lib_path.parent_path().parent_path();

      // Case 1: Installed mode (e.g. /usr/local/include/vectordpu)
      if (fs::exists(base / "include" / "vectordpu"))
          include_dirs.push_back((base / "include" / "vectordpu").string());

      // Case 2: Source tree (from build/lib/, headers are in ../common)
      if (fs::exists(base.parent_path() / "common"))
          include_dirs.push_back((base.parent_path() / "common").string());

      // Case 3: Library is in project root/lib (less common but possible)
      if (fs::exists(base / "common"))
          include_dirs.push_back((base / "common").string());
  }

  // Fallback to current directory if nothing found
  if (include_dirs.empty()) {
      include_dirs.push_back("include/vectordpu");
  }

  std::string include_flags;
  for (const auto& dir : include_dirs) {
      include_flags += " -I" + dir;
  }

  std::string cmd =
      "dpu-upmem-dpurte-clang -DNR_TASKLETS=" + std::to_string(DpuRuntime::get().num_tasklets()) + 
      include_flags + " -O3 -o " + binpath + " " + filepath;

      int ret = system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "JIT Compilation failed: " << cmd << std::endl;
    exit(1);
  }

  #if ENABLE_DPU_LOGGING >= 1
  logger.lock() << "[JIT] Compiled kernel to " << binpath << std::endl;
  #endif

  jit_cache[key] = binpath;
  return binpath;
}

#endif
