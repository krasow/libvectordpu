# UpmemVector.jl -- convenience Makefile
#
# Prerequisites:
#   1. Source the UPMEM environment:
#        source /usr/upmem_env.sh
#
#   2. Build and install upmem-vector (from the upmem-vector repo):
#        make install BACKEND=hw JIT=1
#      This installs to ../vectordpu/ by default.
#
#   3. Install Julia (>= 1.10) and the CxxWrap package:
#        julia -e 'using Pkg; Pkg.add("CxxWrap")'
#
# Then from this directory:
#   make          # build the C++ wrapper
#   make test     # run Julia tests
#   make clean    # remove build artefacts

JULIA       ?= julia
VECTORDPU_DIR ?= $(HOME)/vectordpu

.PHONY: all build test clean

all: build

build:
	VECTORDPU_DIR=$(VECTORDPU_DIR) $(JULIA) --project=. deps/build.jl

test: build
	VECTORDPU_DIR=$(VECTORDPU_DIR) $(JULIA) --project=. test/runtests.jl

clean:
	rm -rf lib/wrapper/build
