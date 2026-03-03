cmake_minimum_required(VERSION 3.13)
project(upmemvector_wrapper CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find JlCxx (CxxWrap C++ library) ----
find_package(JlCxx REQUIRED)
get_target_property(JLCXX_LOCATION JlCxx::cxxwrap_julia LOCATION)
message(STATUS "Found JlCxx at: ${JLCXX_LOCATION}")

# ---- vectordpu installation path ----
set(VECTORDPU_DIR "$ENV{HOME}/vectordpu" CACHE PATH "Path to installed vectordpu library")

if(NOT EXISTS "${VECTORDPU_DIR}/include/vectordpu/vectordpu.h")
    message(FATAL_ERROR
        "vectordpu not found at ${VECTORDPU_DIR}.\n"
        "Build and install it first:\n"
        "  cd /path/to/upmem-vector && make install BACKEND=hw JIT=1\n"
        "Or set -DVECTORDPU_DIR=/path/to/vectordpu")
endif()

# ---- UPMEM SDK (for transitive DPU library dependency) ----
set(UPMEM_HOME "$ENV{UPMEM_HOME}" CACHE PATH "Path to UPMEM SDK")

if(UPMEM_HOME STREQUAL "")
    message(FATAL_ERROR
        "UPMEM_HOME is not set. Source the UPMEM environment first:\n"
        "  source /usr/upmem_env.sh")
endif()

# ---- Build the wrapper shared library ----
add_library(upmemvector_wrapper SHARED wrapper.cpp)

target_include_directories(upmemvector_wrapper PRIVATE
    ${VECTORDPU_DIR}/include/vectordpu
)

target_link_directories(upmemvector_wrapper PRIVATE
    ${VECTORDPU_DIR}/lib
    ${UPMEM_HOME}/lib
)

target_link_libraries(upmemvector_wrapper
    JlCxx::cxxwrap_julia
    vectordpu
    dpu
)

# Install the built library next to the source for easy discovery
install(TARGETS upmemvector_wrapper
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build
)
