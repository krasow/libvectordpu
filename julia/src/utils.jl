# Shared utilities for UpmemVector


const MAX_GC_RETRIES = 2

"""
    retry_on_oom(f)

Executes function `f`. If a DPU OOM or Timeout exception is caught, 
triggers a full Julia Garbage Collection cycle and retries once.
This breaks deadlocks between managed GC and DPU MRAM backpressure.
"""
function retry_on_oom(f)
    retries = 0
    while true
        try
            return f()
        catch e
            msg = sprint(showerror, e)
            if occursin("DPU Timeout", msg) || occursin("DPU OOM", msg)
                if retries >= MAX_GC_RETRIES
                    @error "Exhausted $MAX_GC_RETRIES OOM retries. Giving up."
                    rethrow(e)
                end
                retries += 1
                UpmemVector.dpu_wait_running_events()
                GC.gc(true)
            else
                rethrow(e)
            end
        end
    end
end
# Dispatch helpers for type stability
cpp_length(::Type{Int32}, h) = UpmemVector.cpp_length_Int32(h)
cpp_length(::Type{Int64}, h) = UpmemVector.cpp_length_Int64(h)

from_cpu(::Type{Int32}, data) = UpmemVector.from_cpu_Int32(data)
from_cpu(::Type{Int64}, data) = UpmemVector.from_cpu_Int64(data)

cpp_alloc(::Type{Int32}, n) = UpmemVector.DpuVectorInt32(UInt32(n))
cpp_alloc(::Type{Int64}, n) = UpmemVector.DpuVectorInt64(UInt32(n))

to_cpu!(::Type{Int32}, h, out) = UpmemVector.to_cpu_Int32!(h, out)
to_cpu!(::Type{Int64}, h, out) = UpmemVector.to_cpu_Int64!(h, out)

dpu_fence(::Type{Int32}, h) = UpmemVector.dpu_fence_Int32(h)
dpu_fence(::Type{Int64}, h) = UpmemVector.dpu_fence_Int64(h)

free_vector(::Type{Int32}, h) = UpmemVector.free_vector_Int32(h)
free_vector(::Type{Int64}, h) = UpmemVector.free_vector_Int64(h)

launch_binary(::Type{Int32}, args...) = UpmemVector.launch_binary_Int32(args...)
launch_binary(::Type{Int64}, args...) = UpmemVector.launch_binary_Int64(args...)

launch_binary_inplace(::Type{Int32}, args...) = UpmemVector.launch_binary_inplace_Int32(args...)
launch_binary_inplace(::Type{Int64}, args...) = UpmemVector.launch_binary_inplace_Int64(args...)

launch_binary_scalar(::Type{Int32}, args...) = UpmemVector.launch_binary_scalar_Int32(args...)
launch_binary_scalar(::Type{Int64}, args...) = UpmemVector.launch_binary_scalar_Int64(args...)

launch_unary(::Type{Int32}, args...) = UpmemVector.launch_unary_Int32(args...)
launch_unary(::Type{Int64}, args...) = UpmemVector.launch_unary_Int64(args...)

launch_reduction(::Type{Int32}, args...) = UpmemVector.launch_reduction_Int32(args...)
launch_reduction(::Type{Int64}, args...) = UpmemVector.launch_reduction_Int64(args...)
