"""
    DpuVector{T}(data::AbstractVector{T})
    DpuVector{T}(n::Integer)

A 1-D vector stored in UPMEM DPU memory with element type T (Int32 or Int64).
"""
mutable struct DpuVector{T}
    handle::Any   # CxxWrap-managed C++ object (DpuVectorInt32 or DpuVectorInt64)
    len::Int

    function DpuVector{T}(handle) where T
        new{T}(handle, Int(cpp_length(T, handle)))
    end
end

# Shorthand for DpuVector(Int32[...])
DpuVector(data::Vector{T}) where T = DpuVector{T}(data)

# Construct from a Julia vector -- transfer to DPU memory
function DpuVector{T}(data::Vector{T}) where T
    handle = retry_on_oom() do
        from_cpu(T, data)
    end
    return DpuVector{T}(handle)
end

function DpuVector{T}(data::AbstractVector) where T
    return DpuVector{T}(collect(T, data))
end

# Default to Int32 if no type provided and no data to infer from
DpuVector(n::Integer) = DpuVector{Int32}(n)

# Allocate an uninitialised DPU vector of length n
function DpuVector{T}(n::Integer) where T
    handle = retry_on_oom() do
        cpp_alloc(T, n)
    end
    return DpuVector{T}(handle)
end

# ---- Conversions: DPU -> Julia ----

"""
    Array(v::DpuVector{T}) -> Vector{T}

Transfer DPU vector contents back to the host as a Julia `Vector{T}`.
"""
function Base.Array(v::DpuVector{T}) where T
    out = Vector{T}(undef, v.len)
    retry_on_oom() do
        to_cpu!(T, v.handle, out)
    end
    return out
end

Base.Vector(v::DpuVector) = Array(v)
Base.collect(v::DpuVector) = Array(v)

# ---- Basic queries ----

Base.length(v::DpuVector) = v.len
Base.size(v::DpuVector) = (v.len,)
Base.ndims(::Type{<:DpuVector}) = 1
Base.axes(v::DpuVector) = (Base.OneTo(v.len),)
Base.broadcastable(v::DpuVector) = v
Base.eltype(::DpuVector{T}) where T = T
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
function fence(v::DpuVector{T}) where T
    retry_on_oom() do
        dpu_fence(T, v.handle)
    end
end

"""
    free!(v::DpuVector)

Explicitly drops the underlying C++ reference so memory can be freed before GC.
"""
function free!(v::DpuVector{T}) where T
    free_vector(T, v.handle)
end

export fence, free!
