# https://github.com/CMU-SAFARI/prim-benchmarks/tree/main 
# leveraged the above repository to create Makefile for DPU and host code compilation

DPU_DIR := dpu
HOST_DIR := host
TEST_DIR := test
BUILDDIR ?= bin
NR_DPUS ?= 32
NR_TASKLETS ?= 16

BACKEND ?= simulator
DEBUG ?= 0
LOGGING ?= 0

# this option enables fencing after dpu-to-host transfers automatically
# you can disable it to manually control fencing in your code with add_fence() calls
ENABLE_AUTO_FENCING ?= 1

# this option enables printing from the DPU to the host stdout
ENABLE_DPU_PRINTING ?= 0

# the default compiler on feta supports up to C++17
# c++20 is needed for some debugging output from std::source_location
CXX_STANDARD ?= c++20

# ----------------- Edit above this line -----------------

ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif

RUNTIME_PATH := $(abspath $(CURDIR)/bin)
RUNTIME := $(RUNTIME_PATH)/runtime.dpu

CONFIG_FLAGS ?= -DDPU_RUNTIME=\"$(RUNTIME)\" \
	-DENABLE_DPU_LOGGING=$(LOGGING) \
	-DBACKEND=\"$(BACKEND)\" \
	-DENABLE_AUTO_FENCING=$(ENABLE_AUTO_FENCING) \
	-DENABLE_DPU_PRINTING=$(ENABLE_DPU_PRINTING)

HOST_TARGET := ${BUILDDIR}/libvectordpu.so
DPU_TARGET := ${BUILDDIR}/runtime.dpu
TEST_TARGET := ${TEST_DIR}/vectordpu_test

COMMON_INCLUDES := common
HOST_INCLUDES := host
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.cc)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)
TEST_SOURCES := $(wildcard ${TEST_DIR}/*.cc)

HOST_HEADERS := $(wildcard ${HOST_DIR}/*.inl) $(wildcard ${HOST_DIR}/*.h)
DPU_HEADERS := $(wildcard ${DPU_DIR}/*.inl) $(wildcard ${DPU_DIR}/*.h)

ifeq ($(DEBUG),1)
  CXXFLAGS += -g -O0 -DDEBUG -fsanitize=address -fno-omit-frame-pointer
  LDFLAGS  +=
  BUILD_TYPE := debug
else
  CXXFLAGS += -O3 -DNDEBUG
  BUILD_TYPE := release
endif

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES}
HOST_FLAGS := ${COMMON_FLAGS} ${CXXFLAGS} `dpu-pkg-config --cflags --libs dpu` \
				-DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS} ${CONFIG_FLAGS}
DPU_FLAGS := ${COMMON_FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS}

all: ${HOST_TARGET} ${DPU_TARGET}
	@echo "Build complete: $(BUILD_TYPE)"

${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES} ${HOST_HEADERS}
	$(CXX) -std=${CXX_STANDARD} -shared -fPIC -o $@ ${HOST_SOURCES} ${HOST_FLAGS} 


${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES} ${DPU_HEADERS}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

$(TEST_TARGET): all
	$(CXX) -std=${CXX_STANDARD} $(CXXFLAGS) $(COMMON_FLAGS) -o $@ $(TEST_SOURCES) -I$(HOST_INCLUDES)  \
		-L$(BUILDDIR) -Wl,-rpath,$(RUNTIME_PATH) -lvectordpu

clean:
	$(RM) -r $(BUILDDIR) $(TEST_TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)
