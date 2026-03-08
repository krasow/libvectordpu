module UpmemVector

using CxxWrap

# Path to the compiled wrapper shared library (without extension)
const _wrapper_lib = joinpath(@__DIR__, "..", "lib", "wrapper", "build", "libupmemvector_wrapper")

@wrapmodule(() -> _wrapper_lib)

function __init__()
    @initcxx
    atexit() do
        dpu_shutdown()
    end
end

include("utils.jl")
include("types.jl")
include("operations.jl")
include("display.jl")

export DpuVector

end # module UpmemVector
