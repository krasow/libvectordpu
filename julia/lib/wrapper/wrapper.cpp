#include <jlcxx/jlcxx.hpp>
#include <jlcxx/array.hpp>
#include <jlcxx/stl.hpp>

#include <vectordpu.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

// ============================================================
// Opcode-to-KernelID lookup tables (one per operation category)
// ============================================================

struct OpEntry { KernelID kid; uint8_t opcode; };

// Binary vector-vector:  ADD=0, SUB=1, MUL=2, DIV=3, ASR=4
static constexpr OpEntry binary_ops[] = {
    { OpInfo<int32_t>::add, OpInfo<int32_t>::add_op },
    { OpInfo<int32_t>::sub, OpInfo<int32_t>::sub_op },
    { OpInfo<int32_t>::mul, OpInfo<int32_t>::mul_op },
    { OpInfo<int32_t>::div, OpInfo<int32_t>::div_op },
    { OpInfo<int32_t>::asr, OpInfo<int32_t>::asr_op },
};
static constexpr int NUM_BINARY_OPS = sizeof(binary_ops) / sizeof(binary_ops[0]);

// Binary vector-scalar:  ADD=0, SUB=1, MUL=2, DIV=3, ASR=4
static constexpr OpEntry scalar_ops[] = {
    { OpInfo<int32_t>::add_scalar, OpInfo<int32_t>::add_scalar_op },
    { OpInfo<int32_t>::sub_scalar, OpInfo<int32_t>::sub_scalar_op },
    { OpInfo<int32_t>::mul_scalar, OpInfo<int32_t>::mul_scalar_op },
    { OpInfo<int32_t>::div_scalar, OpInfo<int32_t>::div_scalar_op },
    { OpInfo<int32_t>::asr_scalar, OpInfo<int32_t>::asr_scalar_op },
};
static constexpr int NUM_SCALAR_OPS = sizeof(scalar_ops) / sizeof(scalar_ops[0]);

// Unary:  NEGATE=0, ABS=1
static constexpr OpEntry unary_ops[] = {
    { OpInfo<int32_t>::negate, OpInfo<int32_t>::negate_op },
    { OpInfo<int32_t>::abs,    OpInfo<int32_t>::abs_op },
};
static constexpr int NUM_UNARY_OPS = sizeof(unary_ops) / sizeof(unary_ops[0]);

// Reduction:  MIN=0, MAX=1, SUM=2, PRODUCT=3
static constexpr OpEntry reduction_ops[] = {
    { OpInfo<int32_t>::min,     OpInfo<int32_t>::min_op },
    { OpInfo<int32_t>::max,     OpInfo<int32_t>::max_op },
    { OpInfo<int32_t>::sum,     OpInfo<int32_t>::sum_op },
    { OpInfo<int32_t>::product, OpInfo<int32_t>::product_op },
};
static constexpr int NUM_REDUCTION_OPS = sizeof(reduction_ops) / sizeof(reduction_ops[0]);

// ============================================================

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    // ---- wrapped type ----
    mod.add_type<dpu_vector<int32_t>>("DpuVectorInt32")
        .constructor<uint32_t>()
        .method("cpp_length", [](const dpu_vector<int32_t>& v) -> int64_t {
            return static_cast<int64_t>(v.size());
        });

    // ---- host <-> DPU transfers ----

    mod.method("from_cpu_int32", [](jlcxx::ArrayRef<int32_t> arr) {
        auto result = dpu_vector<int32_t>::from_cpu(arr.data(), static_cast<uint32_t>(arr.size()));
        // Wait for transfer to complete before returning to Julia 
        // to ensure Julia doesn't move/reclaim the buffer.
        DpuRuntime::get().get_event_queue().process_events(result.data_desc().last_producer_id);
        return result;
    });

    mod.method("to_cpu!", [](dpu_vector<int32_t>& v, jlcxx::ArrayRef<int32_t> out) {
        size_t n = std::min(static_cast<size_t>(v.size()),
                            static_cast<size_t>(out.size()));
        v.to_cpu(out.data(), static_cast<uint32_t>(n));
    });

    // ---- modular dispatchers ----

    // Binary vector-vector: op_idx in {0..4} = {ADD,SUB,MUL,DIV,ASR}
    mod.method("launch_binary", [](const dpu_vector<int32_t>& lhs,
                                   const dpu_vector<int32_t>& rhs,
                                   int32_t op_idx) {
        assert(op_idx >= 0 && op_idx < NUM_BINARY_OPS);
        auto& e = binary_ops[op_idx];
        dpu_vector<int32_t> res(lhs.size(), 0, true);
        detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(),
                              rhs.data_desc_ref(), e.kid, e.opcode,
                              OpInfo<int32_t>::universal_pipeline);
        return res;
    });

    // Binary vector-scalar: op_idx in {0..4} = {ADD,SUB,MUL,DIV,ASR}
    mod.method("launch_binary_scalar", [](const dpu_vector<int32_t>& lhs,
                                          int32_t scalar,
                                          int32_t op_idx) {
        assert(op_idx >= 0 && op_idx < NUM_SCALAR_OPS);
        auto& e = scalar_ops[op_idx];
        uint32_t scalar_bits = 0;
        std::memcpy(&scalar_bits, &scalar, sizeof(int32_t));
        dpu_vector<int32_t> res(lhs.size(), 0, true);
        detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(),
                                     scalar_bits, e.kid, e.opcode,
                                     OpInfo<int32_t>::universal_pipeline);
        return res;
    });

    // Unary: op_idx in {0,1} = {NEGATE, ABS}
    mod.method("launch_unary", [](const dpu_vector<int32_t>& input,
                                  int32_t op_idx) {
        assert(op_idx >= 0 && op_idx < NUM_UNARY_OPS);
        auto& e = unary_ops[op_idx];
        dpu_vector<int32_t> res(input.size(), 0, true);
        detail::launch_unary(res.data_desc_ref(), input.data_desc_ref(),
                             e.kid, e.opcode,
                             OpInfo<int32_t>::universal_pipeline);
        return res;
    });

    // Reduction: op_idx in {0..3} = {MIN, MAX, SUM, PRODUCT}
    // Always returns int64_t (Julia can narrow if needed).
    mod.method("launch_reduction", [](const dpu_vector<int32_t>& input,
                                      int32_t op_idx) -> int64_t {
        assert(op_idx >= 0 && op_idx < NUM_REDUCTION_OPS);
        auto& e = reduction_ops[op_idx];

        auto& runtime = DpuRuntime::get();
#if ENABLE_PROMOTION_REDUCTIONS == 1
        // Use 64-bit buffer to avoid truncation of partial results
        dpu_vector<int64_t> buf(runtime.num_dpus(),
                                runtime.num_tasklets() * 8);
        detail::launch_reduction(buf.data_desc_ref(), input.data_desc_ref(),
                                 e.kid, e.opcode,
                                 OpInfo<int32_t>::universal_pipeline);
        return reduction_cpu(buf, e.kid);
#else
        dpu_vector<int32_t> buf(runtime.num_dpus(),
                                runtime.num_tasklets() * sizeof(size_t));
        detail::launch_reduction(buf.data_desc_ref(), input.data_desc_ref(),
                                 e.kid, e.opcode,
                                 OpInfo<int32_t>::universal_pipeline);
        return reduction_cpu(buf, e.kid);
#endif
    });

    // ---- synchronization ----

    mod.method("dpu_fence", [](dpu_vector<int32_t>& v) {
        v.add_fence();
    });
}
