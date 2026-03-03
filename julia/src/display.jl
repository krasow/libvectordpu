# REPL display for DpuVector

const _MAX_DISPLAY_ELEMENTS = 10

function Base.show(io::IO, v::DpuVector)
    print(io, "DpuVector{Int32}(", v.len, ")")
end

function Base.show(io::IO, ::MIME"text/plain", v::DpuVector)
    println(io, v.len, "-element DpuVector{Int32}:")
    if v.len == 0
        return
    end

    data = Array(v)

    if v.len <= 2 * _MAX_DISPLAY_ELEMENTS
        # small enough to show everything
        for i in eachindex(data)
            print(io, " ", data[i])
            i < length(data) && println(io)
        end
    else
        # show first and last _MAX_DISPLAY_ELEMENTS entries
        for i in 1:_MAX_DISPLAY_ELEMENTS
            println(io, " ", data[i])
        end
        println(io, " \u22ee")                  # vertical ellipsis
        for i in (v.len - _MAX_DISPLAY_ELEMENTS + 1):v.len
            print(io, " ", data[i])
            i < v.len && println(io)
        end
    end
end
