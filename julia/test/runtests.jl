using Test
using UpmemVector

# DPU vectors need at least num_dpus elements to work correctly.
# With NR_DPUS=64 (one full rank), use vectors of 64+ elements.
# For reductions, use even larger vectors to span all tasklets.

const N = 4096  # safe vector length for all DPU counts/tasklets

@testset "UpmemVector" begin

    @testset "construction and conversion" begin
        data = Int32.(collect(1:N))
        v = DpuVector(data)
        @test length(v) == N
        @test size(v) == (N,)
        @test eltype(v) == Int32

        back = Array(v)
        @test back == data
        @test Vector(v) == data
        @test collect(v) == data

        data2 = Int32.(collect(1:2N))
        v2 = DpuVector(data2)
        @test length(v2) == 2N
        @test Array(v2) == data2
    end

    @testset "scalar indexing" begin
        data = Int32.(collect(10:10:10N))
        v = DpuVector(data)
        @test v[1] == Int32(10)
        @test v[N] == Int32(10N)
        @test_throws BoundsError v[0]
        @test_throws BoundsError v[N+1]
    end

    @testset "binary vector-vector" begin
        a_data = Int32.(collect(1:N))
        b_data = Int32.(collect(N:-1:1))
        a = DpuVector(a_data)
        b = DpuVector(b_data)

        @test Array(a + b) == a_data .+ b_data
        @test Array(a - b) == a_data .- b_data
        @test Array(a * b) == a_data .* b_data
        @test Array(div(a, b)) == a_data .÷ b_data
    end

    @testset "binary vector-scalar" begin
        a_data = Int32.(collect(2:2:2N))
        a = DpuVector(a_data)

        @test Array(a + 10)    == a_data .+ Int32(10)
        @test Array(10 + a)    == a_data .+ Int32(10)
        @test Array(a - 1)     == a_data .- Int32(1)
        @test Array(a * 3)     == a_data .* Int32(3)
        @test Array(3 * a)     == a_data .* Int32(3)
        @test Array(div(a, 2)) == a_data .÷ Int32(2)
        @test Array(a >> 1)    == a_data .>> Int32(1)
    end

    @testset "unary operations" begin
        a_data = Int32.(vcat(collect(-N÷2:-1), collect(1:N÷2)))
        a = DpuVector(a_data)

        @test Array(-a)    == -a_data
        @test Array(abs(a)) == abs.(a_data)
    end

    @testset "reductions" begin
        n = 4096
        a = DpuVector(Int32.(collect(1:n)))

        @test sum(a) == Int64(n) * Int64(n + 1) ÷ 2
        @test minimum(a) == 1
        @test maximum(a) == n

        b = DpuVector(fill(Int32(1), n))
        @test sum(b) == n
        @test prod(b) == 1
        @test minimum(b) == 1
        @test maximum(b) == 1
    end

    @testset "chained operations" begin
        a_data = Int32.(collect(1:N))
        b_data = Int32.(collect(N:-1:1))
        a = DpuVector(a_data)
        b = DpuVector(b_data)

        result = abs(-((a + b) - a))
        @test Array(result) == abs.(-(((a_data .+ b_data) .- a_data)))
    end

    @testset "display" begin
        v = DpuVector(Int32.(collect(1:N)))
        buf = IOBuffer()
        show(buf, v)
        @test String(take!(buf)) == "DpuVector{Int32}($N)"

        show(buf, MIME("text/plain"), v)
        s = String(take!(buf))
        @test occursin("$N-element DpuVector{Int32}:", s)
        @test occursin("1", s)
    end

end
