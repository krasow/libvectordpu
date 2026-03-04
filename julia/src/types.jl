# DpuVector -- Julia wrapper around the CxxWrap-managed dpu_vector<int32_t>

"""
    DpuVector(data::AbstractVector{<:Integer})
    DpuVector(n::Integer)

A 1-D vector stored in UPMEM DPU memory.

Construct from a Julia array to transfer data to the DPUs, or pass an integer
to allocate an uninitialised vector of that length.

# Examples
```julia
v = DpuVector(Int32[1, 2, 3, 4])
w = DpuVector(1024)                 # uninitialised, length 1024
```
"""
mutable struct DpuVector
    handle::UpmemVector.DpuVectorInt32   # CxxWrap-managed C++ object
    len::Int

    function DpuVector(handle::UpmemVector.DpuVectorInt32)
        v = new(handle, Int(UpmemVector.cpp_length(handle)))
        return v
    end
end

# Construct from a Julia vector -- transfer to DPU memory
function DpuVector(data::Vector{Int32})
    handle = retry_on_oom() do
        UpmemVector.from_cpu_int32(data)
    end
    return DpuVector(handle)
end

function DpuVector(data::AbstractVector{Int32})
    return DpuVector(collect(Int32, data))
end

# Accept any integer array by converting to Int32
function DpuVector(data::AbstractVector{<:Integer})
    return DpuVector(convert(Vector{Int32}, data))
end

# Allocate an uninitialised DPU vector of length n
function DpuVector(n::Integer)
    handle = retry_on_oom() do
        UpmemVector.DpuVectorInt32(UInt32(n))
    end
    return DpuVector(handle)
end

# ---- Conversions: DPU -> Julia ----

"""
    Array(v::DpuVector) -> Vector{Int32}

Transfer DPU vector contents back to the host as a Julia `Vector{Int32}`.
"""
function Base.Array(v::DpuVector)
    out = Vector{Int32}(undef, v.len)
    retry_on_oom() do
        UpmemVector.to_cpu!(v.handle, out)
    end
    return out
end

Base.Vector(v::DpuVector) = Array(v)
Base.collect(v::DpuVector) = Array(v)

# ---- Basic queries ----

Base.length(v::DpuVector) = v.len
Base.size(v::DpuVector) = (v.len,)
Base.ndims(::Type{DpuVector}) = 1
Base.axes(v::DpuVector) = (Base.OneTo(v.len),)
Base.broadcastable(v::DpuVector) = v
Base.eltype(::DpuVector) = Int32
Base.BroadcastStyle(::Type{<:DpuVector}) = Base.Broadcast.DefaultArrayStyle{1}()

# Scalar indexing (requires full transfer -- use sparingly)
function Base.getindex(v::DpuVector, i::Int)
    @boundscheck 1 <= i <= v.len || throw(BoundsError(v, i))
    return Array(v)[i]
end

"""
    fence(v::DpuVector)

Explicitly synchronize: block until all pending DPU operations on `v` complete.
"""
function fence(v::DpuVector)
    retry_on_oom() do
        UpmemVector.dpu_fence(v.handle)
    end
end

"""
    free!(v::DpuVector)

Explicitly drops the underlying C++ reference so memory can be freed before GC.
"""
function free!(v::DpuVector)
    UpmemVector.free_vector(v.handle)
end

export fence, free!
