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
