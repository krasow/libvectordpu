using CxxWrap

wrapper_dir = joinpath(@__DIR__, "..", "lib", "wrapper")
build_dir   = joinpath(wrapper_dir, "build")
mkpath(build_dir)

# CxxWrap / JlCxx paths
jlcxx_prefix = CxxWrap.prefix_path()
julia_prefix = joinpath(Sys.BINDIR, "..")

# vectordpu install location (override via VECTORDPU_DIR env var)
vectordpu_dir = get(ENV, "VECTORDPU_DIR", joinpath(homedir(), "vectordpu"))

if !isdir(vectordpu_dir)
    error("""
    vectordpu installation not found at: $vectordpu_dir
    Build and install upmem-vector first:
        source /usr/upmem_env.sh
        cd /path/to/upmem-vector
        make install BACKEND=hw JIT=1
    Or set VECTORDPU_DIR to point to your installation.""")
end

@info "Building UpmemVector C++ wrapper" jlcxx_prefix vectordpu_dir

cd(build_dir) do
    run(`cmake $(wrapper_dir)
        -DCMAKE_PREFIX_PATH=$(jlcxx_prefix)
        -DJulia_PREFIX=$(julia_prefix)
        -DVECTORDPU_DIR=$(vectordpu_dir)
        -DCMAKE_BUILD_TYPE=Release`)
    run(`cmake --build . --config Release`)
end

@info "UpmemVector C++ wrapper built successfully"
