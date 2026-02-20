# https://github.com/CMU-SAFARI/prim-benchmarks/tree/main 
# leveraged the above repository to create Makefile for DPU and host code compilation
BUILDDIR ?= build
NR_TASKLETS ?= 12

BACKEND ?= simulator
DEBUG ?= 0
LOGGING ?= 0

# this option enables experimental pipeline and fusion features
PIPELINE ?= 0
# this option enables JIT compilation of pipeline kernels
JIT ?= 0
# this option sets the maximum number of unique kernels to fuse into a single JIT binary
MAX_JIT_QUEUE_DEPTH ?= 8

# JIT requires pipeline logic to dispatch events correctly
ifeq ($(JIT),1)
  PIPELINE := 1
endif

# this option enables fencing after dpu-to-host transfers automatically
# you can disable it to manually control fencing in your code with add_fence() calls
ENABLE_AUTO_FENCING ?= 1

# this option enables printing from the DPU to the host stdout
ENABLE_DPU_PRINTING ?= 0

# this option enables tracing with Perfetto
TRACE ?= 0
PERFETTO_HOME ?= /scratch/david/benchmark-upmem/opt/perfetto

# this option prevents the automatic removal of the JIT build directory at shutdown
DEBUG_KEEP_JIT_DIR ?= 0

# the default compiler on feta supports up to C++17
# c++20 is needed for some debugging output from std::source_location
CXX_STANDARD ?= c++17

# ----------------- Edit above this line -----------------

ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif

DPU_DIR := dpu
HOST_DIR := host
TEST_DIR := test

DESTDIR ?= ../vectordpu

CONFIG_STAMP := build.config

HOST_TARGET := ${BUILDDIR}/lib/libvectordpu.so
DPU_TARGET := ${BUILDDIR}/bin/runtime.dpu
TEST_TARGET := ${TEST_DIR}/vectordpu_test

COMMON_DIR := common
HOST_INCLUDES := host
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.cc) $(wildcard ${HOST_DIR}/perfetto/*.cc)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)
TEST_SOURCES := $(wildcard ${TEST_DIR}/*.cc)

HOST_HEADERS := $(wildcard ${HOST_DIR}/*.inl) $(wildcard ${HOST_DIR}/*.h) $(wildcard ${HOST_DIR}/perfetto/*.h)
DPU_HEADERS := $(wildcard ${DPU_DIR}/*.inl) $(wildcard ${DPU_DIR}/*.h)
COMMON_HEADERS := ${COMMON_DIR}/common.h ${COMMON_DIR}/config.h

ifeq ($(DEBUG),1)
  CXXFLAGS += -g -pg -O0 -DDEBUG -fsanitize=address -fno-omit-frame-pointer
  LDFLAGS  +=
  BUILD_TYPE := debug
else
  CXXFLAGS += -O3 -DNDEBUG
  CXXFLAGS += -O3 -DNDEBUG
  BUILD_TYPE := release
endif

# Debian 10 / GCC 8 requirement for filesystem
LDFLAGS += -lstdc++fs

ifeq ($(TRACE),1)
  CXXFLAGS += -pthread -I$(PERFETTO_HOME)/include
  LDFLAGS += -L$(PERFETTO_HOME)/lib -lperfetto -ldl -lpthread
endif

.PHONY: config_check cache_old reconfigure all clean clean-internal test install uninstall print_config make_header

GENERATED_TARGETS := dpu/kernels.h host/opinfo.h host/kernelids.h common/opcodes.h


__dirs := $(shell mkdir -p ${BUILDDIR} && mkdir -p ${BUILDDIR}/bin && mkdir -p ${BUILDDIR}/lib)

COMMON_FLAGS := -Wall -Wextra -I${COMMON_DIR} -I${HOST_DIR}
HOST_FLAGS := ${COMMON_FLAGS} ${CXXFLAGS} `dpu-pkg-config --cflags --libs dpu`
# DPU-specific flags
DPU_FLAGS := ${COMMON_FLAGS} -O3 -DNR_TASKLETS=${NR_TASKLETS}

all: $(GENERATED_TARGETS) config_check print_config ${HOST_TARGET} ${DPU_TARGET}
	@echo "Build complete: $(BUILD_TYPE) \n"


$(GENERATED_TARGETS): tools/generate.py
	@echo "Generating kernel headers..."
	python3 tools/generate.py


# Explicit rule for config.h
common/config.h: tools/generate_config.py $(CONFIG_STAMP)
	@echo "Generating config header..."
	python3 tools/generate_config.py

$(CONFIG_STAMP):
	@$(MAKE) reconfigure

make_header: common/config.h

reconfigure:
	@echo "NR_TASKLETS=$(NR_TASKLETS)" > $(CONFIG_STAMP)
	@echo "BACKEND=$(BACKEND)" >> $(CONFIG_STAMP)
	@echo "DEBUG=$(DEBUG)" >> $(CONFIG_STAMP)
	@echo "ENABLE_DPU_LOGGING=$(LOGGING)" >> $(CONFIG_STAMP)
	@echo "ENABLE_AUTO_FENCING=$(ENABLE_AUTO_FENCING)" >> $(CONFIG_STAMP)
	@echo "ENABLE_DPU_PRINTING=$(ENABLE_DPU_PRINTING)" >> $(CONFIG_STAMP)
	@echo "CXX_STANDARD=$(CXX_STANDARD)" >> $(CONFIG_STAMP)
	@echo "PIPELINE=$(PIPELINE)" >> $(CONFIG_STAMP)
	@echo "JIT=$(JIT)" >> $(CONFIG_STAMP)
	@echo "MAX_JIT_QUEUE_DEPTH=$(MAX_JIT_QUEUE_DEPTH)" >> $(CONFIG_STAMP)
	@echo "TRACE=$(TRACE)" >> $(CONFIG_STAMP)
	@echo "PERFETTO_HOME=$(PERFETTO_HOME)" >> $(CONFIG_STAMP)
	@echo "DEBUG_KEEP_JIT_DIR=$(DEBUG_KEEP_JIT_DIR)" >> $(CONFIG_STAMP)

cache_old:
	@if [ -f "$(CONFIG_STAMP)" ]; then \
	    rm -f $(CONFIG_STAMP).old; \
		cp -f $(CONFIG_STAMP) $(CONFIG_STAMP).old; \
	fi

config_check: cache_old reconfigure make_header
	@if [ -f "$(CONFIG_STAMP)" ]; then \
	    cmp -s $(CONFIG_STAMP) $(CONFIG_STAMP).old 2>/dev/null || { \
	        echo "Configuration changed, cleaning build..."; \
	        $(MAKE) clean-internal; \
			mkdir -p $(BUILDDIR) && mkdir -p $(BUILDDIR)/bin && mkdir -p $(BUILDDIR)/lib; \
	    }; \
		rm -f $(CONFIG_STAMP).old; \
	fi

${HOST_TARGET}: ${HOST_SOURCES} ${HOST_HEADERS} ${COMMON_HEADERS} $(GENERATED_TARGETS)
	$(CXX) -std=${CXX_STANDARD} -shared -fPIC -o $@ ${HOST_SOURCES} ${HOST_FLAGS} $(LDFLAGS)

${DPU_TARGET}: ${DPU_SOURCES} ${DPU_HEADERS} ${COMMON_HEADERS} $(GENERATED_TARGETS)
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

$(TEST_TARGET): ${TEST_SOURCES} ${HOST_TARGET} ${DPU_TARGET}
	@echo "Building test target: $@"
	$(CXX) -std=${CXX_STANDARD} $(CXXFLAGS) $(COMMON_FLAGS) -o $@ $(TEST_SOURCES) -I$(HOST_INCLUDES)  \
		-L$(BUILDDIR)/lib -Wl,-rpath,$(BUILDDIR)/lib -lvectordpu

clean-internal:
	$(RM) -r $(BUILDDIR) $(TEST_TARGET)

clean: clean-internal
	$(RM) -r $(CONFIG_STAMP) $(GENERATED_TARGETS) common/config.h

# ANSI color codes
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
CYAN   := \033[0;36m
NC     := \033[0m  # No color

print_config: reconfigure
	@echo "\n$(CYAN)Current build configuration:$(NC)"
	@cat $(CONFIG_STAMP) | while read line; do \
	    key=$${line%%=*}; \
	    value=$${line#*=}; \
	    echo "  $(YELLOW)$${key}=$(GREEN)$${value}$(NC)"; \
	done
	@echo "\n"

test: all $(TEST_TARGET) 
	@printf "\n$(CYAN)Running tests...$(NC)\n\n"
	./$(TEST_TARGET)

bindir := $(DESTDIR)/bin
libdir := $(DESTDIR)/lib
includedir := $(DESTDIR)/include/vectordpu

install: all
	@echo "Installing to $(DESTDIR)..."
	install -d $(bindir) $(libdir) $(includedir)
	install -m 644 $(DPU_TARGET) $(bindir)
	install -m 644 $(HOST_TARGET) $(libdir)
	# Install base host headers
	install -m 644 $(wildcard ${HOST_DIR}/*.inl) $(wildcard ${HOST_DIR}/*.h) $(includedir)
	# Install perfetto headers
	install -d $(includedir)/perfetto
	install -m 644 $(wildcard ${HOST_DIR}/perfetto/*.h) $(includedir)/perfetto
	# Install common and generated headers
	install -m 644 $(COMMON_HEADERS) $(includedir)
	install -m 644 $(GENERATED_TARGETS) $(includedir)

uninstall:
	@echo "Removing from $(prefix)..."
	rm -f $(bindir)/$(notdir $(DPU_TARGET))
	rm -f $(libdir)/$(notdir $(HOST_TARGET))
	rm -rf $(includedir)
