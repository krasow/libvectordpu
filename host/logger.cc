#include "logger.h"

#include "allocator.h"
#include "kernelids.h"
#include "queue.h"

const char* kernel_id_to_string(KernelID id) { return kernel_infos[id].name; }

const char* ktype_to_string(KernelCategory ktype) {
  switch (ktype) {
    case KERNEL_BINARY:
      return "BINARY";
    case KERNEL_UNARY:
      return "UNARY";
    case KERNEL_REDUCTION:
      return "REDUCTION";
    default:
      return "UNKNOWN_KTYPE";
  }
}

void print_vector_desc(Logger& logger, detail::VectorDesc desc, uint32_t reserved) {
  auto out = logger.lock();
  out << "  " << std::left << std::setw(6) << "DPU" << std::setw(14) << "PTR"
      << std::setw(14) << "ALLOC(bytes)" << std::setw(14) << "VEC_SIZE(bytes)\n"
      << std::string(51, '-') << "\n";

  for (size_t i = 0; i < desc.num_elements; i++) {
    std::ostringstream ptr_hex;
    ptr_hex << "0x" << std::hex << std::setw(8) << std::setfill('0')
            << desc.desc[i].ptr;

    out << "  " << std::left << std::setw(6) << i << std::setw(14)
        << ptr_hex.str() << std::setw(14) << std::dec << desc.desc[i].size_bytes
        << std::dec << (desc.desc[i].size_bytes - reserved) << "\n";
  }
}

void log_allocation(Logger& logger, const std::type_info& type, uint32_t n,
                    std::string_view debug_name, const char* debug_file,
                    int debug_line, bool is_allocation) {
  auto log = logger.lock();
  log << "[mem-logger] action=" << (is_allocation ? "allocate  " : "deallocate")
      << " type=dpu_vector<" << type.name() << ">"
      << " size=" << n;
  if (!debug_name.empty()) {
    log << " (name=\"" << debug_name << "\")";
  }
  if (debug_file != nullptr && debug_line >= 0) {
    log << " at " << debug_file << ":" << debug_line;
  }
  log << std::endl;
}

#if ENABLE_DPU_LOGGING >= 1
void log_dpu_launch_args(Logger& logger, const DPU_LAUNCH_ARGS* args,
                         uint32_t nr_of_dpus) {
  auto log = logger.lock();
  log << "[task-logger] kernel="
      << kernel_id_to_string(static_cast<KernelID>(args->kernel))
      << " dpus=" << nr_of_dpus << std::endl;

// the following code is gross, but it's just for logging purposes
// it creates a table of the launch arguments for each DPU
#if ENABLE_DPU_LOGGING >= 2
  // Determine which columns to show
  bool show_rhs = false;
  bool show_lhs = false;
  bool show_src = false;
  bool show_res = false;

  const auto& a = args[0];
  if (a.ktype == KERNEL_BINARY) {
    show_rhs = true;
    show_lhs = true;
    show_res = true;
  } else if (a.ktype == KERNEL_UNARY || a.ktype == KERNEL_REDUCTION) {
    show_rhs = true;
    show_res = true;
  }

  log << "  " << std::left << std::setw(6) << "DPU" << std::setw(12) << "KTYPE"
      << std::setw(12) << "NUM_ELEMS" << std::setw(9) << "SIZE_T";

  if (show_rhs) log << std::setw(13) << "RHS_OFFSET";
  if (show_lhs) log << std::setw(13) << "LHS_OFFSET";
  if (show_src) log << std::setw(13) << "SRC_OFFSET";
  if (show_res) log << std::setw(13) << "RES_OFFSET";

  log << "\n" << std::string(38, '-');
  if (a.ktype == KERNEL_BINARY) {
    log << std::string(39, '-');
  } else {
    log << std::string(26, '-');
  }
  log << "\n";

  auto fmt_hex = [](uint32_t v) {
    std::ostringstream ss;
    ss << "0x" << std::hex << std::setw(8) << std::setfill('0') << v;
    return ss.str();
  };

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    const auto& a = args[i];

    std::ostringstream rhs, lhs, src, res;
    rhs << "";
    lhs << "";
    src << "";
    res << "";

    if (a.ktype == KERNEL_BINARY) {
      rhs << fmt_hex(a.binary.rhs_offset);
      lhs << fmt_hex(a.binary.lhs_offset);
      res << fmt_hex(a.binary.res_offset);
    } else if (a.ktype == KERNEL_UNARY) {
      rhs << fmt_hex(a.unary.rhs_offset);
      res << fmt_hex(a.unary.res_offset);
    } else if (a.ktype == KERNEL_REDUCTION) {
      rhs << fmt_hex(a.reduction.rhs_offset);
      res << fmt_hex(a.reduction.res_offset);
    }

    log << "  " << std::left << std::setw(6) << i << std::setw(12)
        << ktype_to_string(static_cast<KernelCategory>(a.ktype)) << std::setw(12) << a.num_elements
        << std::setw(9) << a.size_type;

    if (show_rhs) log << std::setw(13) << rhs.str();
    if (show_lhs) log << std::setw(13) << lhs.str();
    if (show_src) log << std::setw(13) << src.str();
    if (show_res) log << std::setw(13) << res.str();

    log << "\n";
  }
#endif
}
#endif