# Modular opcode-based operation dispatch for DpuVector
#
# Each operation category has its own enum that maps 1:1 to the C++ OpEntry
# lookup tables in wrapper.cpp.  Julia Base overloads simply call the
# appropriate launch_* function with the right enum index.

# ---- operation enums (indices match the C++ OpEntry arrays) ----

module Ops

@enum BinaryOp::Int32 begin
    BINARY_ADD = 0
    BINARY_SUB = 1
    BINARY_MUL = 2
    BINARY_DIV = 3
    BINARY_ASR = 4
end

@enum ScalarOp::Int32 begin
    SCALAR_ADD = 0
    SCALAR_SUB = 1
    SCALAR_MUL = 2
    SCALAR_DIV = 3
    SCALAR_ASR = 4
end

@enum UnaryOp::Int32 begin
    UNARY_NEGATE = 0
    UNARY_ABS    = 1
end

@enum ReductionOp::Int32 begin
    REDUCE_MIN     = 0
    REDUCE_MAX     = 1
    REDUCE_SUM     = 2
    REDUCE_PRODUCT = 3
end

end # module Ops

using .Ops

export Ops

# ---- generic dispatch functions ----

function binary_op(a::DpuVector{T}, b::DpuVector{T}, op::Ops.BinaryOp) where T
    handle = retry_on_oom() do
        launch_binary(T, a.handle, b.handle, Int32(op))
    end
    return DpuVector{T}(handle)
end

function scalar_op(a::DpuVector{T}, s::Integer, op::Ops.ScalarOp) where T
    handle = retry_on_oom() do
        launch_binary_scalar(T, a.handle, Int64(s), Int32(op))
    end
    return DpuVector{T}(handle)
end

function unary_op(a::DpuVector{T}, op::Ops.UnaryOp) where T
    handle = retry_on_oom() do
        launch_unary(T, a.handle, Int32(op))
    end
    return DpuVector{T}(handle)
end

function reduce_op(a::DpuVector{T}, op::Ops.ReductionOp) where T
    return retry_on_oom() do
        launch_reduction(T, a.handle, Int32(op))
    end
end

# ---- Base overloads: binary vector ⊕ vector ----

Base.:+(a::DpuVector{T}, b::DpuVector{T}) where T = binary_op(a, b, Ops.BINARY_ADD)
Base.:-(a::DpuVector{T}, b::DpuVector{T}) where T = binary_op(a, b, Ops.BINARY_SUB)
Base.:*(a::DpuVector{T}, b::DpuVector{T}) where T = binary_op(a, b, Ops.BINARY_MUL)
Base.div(a::DpuVector{T}, b::DpuVector{T}) where T = binary_op(a, b, Ops.BINARY_DIV)

# ---- In-place overloads ----

const INPLACE_OPS = Dict(
    (+) => Ops.BINARY_ADD,
    (-) => Ops.BINARY_SUB,
    (*) => Ops.BINARY_MUL,
    (div) => Ops.BINARY_DIV
)

function Base.materialize!(dest::DpuVector{T}, bc::Base.Broadcast.Broadcasted) where T
    if length(bc.args) == 2 && haskey(INPLACE_OPS, bc.f)
        a, b = bc.args
        if a === dest
            op = INPLACE_OPS[bc.f]
            retry_on_oom() do
                launch_binary_inplace(T, dest.handle, b.handle, Int32(op))
            end
            return dest
        end
    end
    # Fallback to out-of-place and assign
    val = Base.materialize(bc)
    dest.handle = val.handle
    return dest
end

# ---- Base overloads: vector ⊕ scalar / scalar ⊕ vector ----

Base.:+(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_ADD)
Base.:+(s::Integer, a::DpuVector) = scalar_op(a, s, Ops.SCALAR_ADD)
Base.:-(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_SUB)
Base.:*(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_MUL)
Base.:*(s::Integer, a::DpuVector) = scalar_op(a, s, Ops.SCALAR_MUL)
Base.div(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_DIV)
Base.:>>(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_ASR)

# ---- Base overloads: unary ----

Base.:-(a::DpuVector)  = unary_op(a, Ops.UNARY_NEGATE)
Base.abs(a::DpuVector) = unary_op(a, Ops.UNARY_ABS)

# ---- Base overloads: reductions ----

Base.sum(v::DpuVector)     = reduce_op(v, Ops.REDUCE_SUM)
Base.prod(v::DpuVector)    = reduce_op(v, Ops.REDUCE_PRODUCT)
Base.minimum(v::DpuVector) = reduce_op(v, Ops.REDUCE_MIN)
Base.maximum(v::DpuVector) = reduce_op(v, Ops.REDUCE_MAX)
