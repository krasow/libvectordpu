#include <jlcxx/jlcxx.hpp>
#include <jlcxx/array.hpp>
#include <jlcxx/stl.hpp>

#include <vectordpu.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

// ============================================================
// Templated helpers for type registration and dispatch
// ============================================================

template<typename T>
void register_vector_ops(jlcxx::Module& mod, const std::string& type_suffix) {
    std::string type_name = "DpuVector" + type_suffix;
    mod.add_type<dpu_vector<T>>(type_name)
        .template constructor<uint32_t>()
        .method("cpp_length", [](const dpu_vector<T>& v) -> int64_t {
            return static_cast<int64_t>(v.size());
        });

    // Also register cpp_length as a free function so Julia can call UpmemVector.cpp_length_Int32(handle)
    mod.method("cpp_length_" + type_suffix, [](const dpu_vector<T>& v) -> int64_t {
        return static_cast<int64_t>(v.size());
    });

    mod.method("from_cpu_" + type_suffix, [](jlcxx::ArrayRef<T> arr) {
        auto result = dpu_vector<T>::from_cpu(arr.data(), static_cast<uint32_t>(arr.size()));
        DpuRuntime::get().get_event_queue().process_events(result.data_desc().last_producer_id);
        return result;
    });

    mod.method("to_cpu_" + type_suffix + "!", [](dpu_vector<T>& v, jlcxx::ArrayRef<T> out) {
        size_t n = std::min(static_cast<size_t>(v.size()), static_cast<size_t>(out.size()));
        v.to_cpu(out.data(), static_cast<uint32_t>(n));
    });

    mod.method("free_vector_" + type_suffix, [](dpu_vector<T>& v) {
        v.free();
    });

    // Dispatchers for this type
    mod.method("launch_binary_" + type_suffix, [](const dpu_vector<T>& lhs, const dpu_vector<T>& rhs, int32_t op_idx) {
        // We use a fixed set of opcodes, kid comes from OpInfo<T>
        uint8_t opcode = 0;
        KernelID kid = 0;
        switch(op_idx) {
            case 0: kid = OpInfo<T>::add; opcode = OpInfo<T>::add_op; break;
            case 1: kid = OpInfo<T>::sub; opcode = OpInfo<T>::sub_op; break;
            case 2: kid = OpInfo<T>::mul; opcode = OpInfo<T>::mul_op; break;
            case 3: kid = OpInfo<T>::div; opcode = OpInfo<T>::div_op; break;
            case 4: kid = OpInfo<T>::asr; opcode = OpInfo<T>::asr_op; break;
            default: assert(false);
        }
        dpu_vector<T> res(lhs.size(), 0, true);
        detail::launch_binary(res.data_desc_ref(), lhs.data_desc_ref(), rhs.data_desc_ref(), kid, opcode, OpInfo<T>::universal_pipeline);
        return res;
    });

    mod.method("launch_binary_inplace_" + type_suffix, [](dpu_vector<T>& lhs, const dpu_vector<T>& rhs, int32_t op_idx) {
        uint8_t opcode = 0;
        KernelID kid = 0;
        switch(op_idx) {
            case 0: kid = OpInfo<T>::add; opcode = OpInfo<T>::add_op; break;
            case 1: kid = OpInfo<T>::sub; opcode = OpInfo<T>::sub_op; break;
            case 2: kid = OpInfo<T>::mul; opcode = OpInfo<T>::mul_op; break;
            case 3: kid = OpInfo<T>::div; opcode = OpInfo<T>::div_op; break;
            case 4: kid = OpInfo<T>::asr; opcode = OpInfo<T>::asr_op; break;
            default: assert(false);
        }
        detail::launch_binary(lhs.data_desc_ref(), lhs.data_desc_ref(), rhs.data_desc_ref(), kid, opcode, OpInfo<T>::universal_pipeline);
    });

    mod.method("launch_binary_scalar_" + type_suffix, [](const dpu_vector<T>& lhs, int64_t scalar, int32_t op_idx) {
        uint8_t opcode = 0;
        KernelID kid = 0;
        switch(op_idx) {
            case 0: kid = OpInfo<T>::add_scalar; opcode = OpInfo<T>::add_scalar_op; break;
            case 1: kid = OpInfo<T>::sub_scalar; opcode = OpInfo<T>::sub_scalar_op; break;
            case 2: kid = OpInfo<T>::mul_scalar; opcode = OpInfo<T>::mul_scalar_op; break;
            case 3: kid = OpInfo<T>::div_scalar; opcode = OpInfo<T>::div_scalar_op; break;
            case 4: kid = OpInfo<T>::asr_scalar; opcode = OpInfo<T>::asr_scalar_op; break;
            default: assert(false);
        }
        dpu_vector<T> res(lhs.size(), 0, true);
        detail::launch_binary_scalar(res.data_desc_ref(), lhs.data_desc_ref(), static_cast<uint64_t>(scalar), kid, opcode, OpInfo<T>::universal_pipeline);
        return res;
    });

    mod.method("launch_unary_" + type_suffix, [](const dpu_vector<T>& input, int32_t op_idx) {
        uint8_t opcode = 0;
        KernelID kid = 0;
        switch(op_idx) {
            case 0: kid = OpInfo<T>::negate; opcode = OpInfo<T>::negate_op; break;
            case 1: kid = OpInfo<T>::abs;    opcode = OpInfo<T>::abs_op; break;
            default: assert(false);
        }
        dpu_vector<T> res(input.size(), 0, true);
        detail::launch_unary(res.data_desc_ref(), input.data_desc_ref(), kid, opcode, OpInfo<T>::universal_pipeline);
        return res;
    });

    mod.method("launch_reduction_" + type_suffix, [](const dpu_vector<T>& input, int32_t op_idx) -> int64_t {
         uint8_t opcode = 0;
         KernelID kid = 0;
         switch(op_idx) {
             case 0: kid = OpInfo<T>::min;     opcode = OpInfo<T>::min_op; break;
             case 1: kid = OpInfo<T>::max;     opcode = OpInfo<T>::max_op; break;
             case 2: kid = OpInfo<T>::sum;     opcode = OpInfo<T>::sum_op; break;
             case 3: kid = OpInfo<T>::product; opcode = OpInfo<T>::product_op; break;
             default: assert(false);
         }
         auto& runtime = DpuRuntime::get();
         // Always promote to 64-bit for reductions returned to Julia
         dpu_vector<int64_t> buf(runtime.num_dpus(), runtime.num_tasklets() * 8);
         detail::launch_reduction(buf.data_desc_ref(), input.data_desc_ref(), kid, opcode, OpInfo<T>::universal_pipeline);
         return reduction_cpu(buf, kid);
    });

    mod.method("dpu_fence_" + type_suffix, [](dpu_vector<T>& v) {
        v.add_fence();
    });
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    register_vector_ops<int32_t>(mod, "Int32");
    register_vector_ops<int64_t>(mod, "Int64");

    mod.method("dpu_wait_running_events", []() {
        DpuRuntime::get().get_event_queue().wait_running_events();
    });

    mod.method("dpu_shutdown", []() {
        DpuRuntime::get().shutdown();
    });
}
