module UpmemVector

using CxxWrap

# Path to the compiled wrapper shared library (without extension)
const _wrapper_lib = joinpath(@__DIR__, "..", "lib", "wrapper", "build", "libupmemvector_wrapper")

@wrapmodule(() -> _wrapper_lib)

function __init__()
    @initcxx
    atexit() do
        try
            GC.gc(true) # we need to ensure all vectors are destructed before cleanup is called
            UpmemVector.cleanup()
        catch e # ignore errors during shutdown
        end
    end
end

"""
    retry_on_oom(f)

Executes function `f`. If a DPU OOM exception is caught, triggers Julia GC 
and retries once.
"""
function retry_on_oom(f)
    try
        return f()
    catch e
        @info "HELLOOOWOOFGWHJ"
        # CxxWrap throws exceptions as CxxException. 
        # We check the message for "DPU OOM".
        if occursin("DPU OOM", sprint(showerror, e))
            @warn "DPU OOM detected. Syncing and triggering GC..."
            GC.gc(true) # Major GC
            yield()
            return f()
        else
            rethrow(e)
        end
    end
end

include("types.jl")
include("operations.jl")
include("display.jl")

export DpuVector

end # module UpmemVector
