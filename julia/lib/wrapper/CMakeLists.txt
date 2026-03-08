cmake_minimum_required(VERSION 3.13)
project(upmemvector_wrapper CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find JlCxx (CxxWrap C++ library) ----
find_package(JlCxx REQUIRED)
get_target_property(JLCXX_LOCATION JlCxx::cxxwrap_julia LOCATION)
message(STATUS "Found JlCxx at: ${JLCXX_LOCATION}")

# ---- vectordpu repo path ----
set(VECTORDPU_SRC_DIR "/scratch/david/benchmark-upmem/opt/vectordpu_src")

# ---- Build the wrapper shared library ----
add_library(upmemvector_wrapper SHARED wrapper.cpp)

target_include_directories(upmemvector_wrapper PRIVATE
    ${VECTORDPU_SRC_DIR}/host
    ${VECTORDPU_SRC_DIR}/common
)

target_link_directories(upmemvector_wrapper PRIVATE
    ${VECTORDPU_SRC_DIR}/build/lib
    ${UPMEM_HOME}/lib
    /scratch/david/julia-1.11.3/lib/julia
)

target_link_libraries(upmemvector_wrapper
    JlCxx::cxxwrap_julia
    vectordpu
    dpu
)

set_target_properties(upmemvector_wrapper PROPERTIES RUNPATH "${VECTORDPU_SRC_DIR}/build/lib;/scratch/david/julia-1.11.3/lib/julia;${UPMEM_HOME}/lib")

# Install the built library next to the source for easy discovery
# install(TARGETS upmemvector_wrapper
#     LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build
# )
