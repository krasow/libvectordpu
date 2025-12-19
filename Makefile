# https://github.com/CMU-SAFARI/prim-benchmarks/tree/main 
# leveraged the above repository to create Makefile for DPU and host code compilation
BUILDDIR ?= build
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

DPU_DIR := dpu
HOST_DIR := host
TEST_DIR := test

DESTDIR ?= ./install

CONFIG_STAMP := build.config

CONFIG_FLAGS ?= \
	-DENABLE_DPU_LOGGING=$(LOGGING) \
	-DBACKEND=\"$(BACKEND)\" \
	-DENABLE_AUTO_FENCING=$(ENABLE_AUTO_FENCING) \
	-DENABLE_DPU_PRINTING=$(ENABLE_DPU_PRINTING)

HOST_TARGET := ${BUILDDIR}/lib/libvectordpu.so
DPU_TARGET := ${BUILDDIR}/bin/runtime.dpu
TEST_TARGET := ${TEST_DIR}/vectordpu_test

COMMON_DIR := common
HOST_INCLUDES := host
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.cc)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)
TEST_SOURCES := $(wildcard ${TEST_DIR}/*.cc)

HOST_HEADERS := $(wildcard ${HOST_DIR}/*.inl) $(wildcard ${HOST_DIR}/*.h) $(wildcard ${COMMON_DIR}/*.h)
DPU_HEADERS := $(wildcard ${DPU_DIR}/*.inl) $(wildcard ${DPU_DIR}/*.h) $(wildcard ${COMMON_DIR}/*.h)

ifeq ($(DEBUG),1)
  CXXFLAGS += -g -O0 -DDEBUG -fsanitize=address -fno-omit-frame-pointer
  LDFLAGS  +=
  BUILD_TYPE := debug
else
  CXXFLAGS += -O3 -DNDEBUG
  BUILD_TYPE := release
endif

.PHONY: config_check cache_old reconfigure all clean clean-internal test install uninstall print_config

GENERATED_TARGETS := dpu/kernels.h host/opinfo.h host/kernelids.h


__dirs := $(shell mkdir -p ${BUILDDIR} && mkdir -p ${BUILDDIR}/bin && mkdir -p ${BUILDDIR}/lib)

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_DIR}
HOST_FLAGS := ${COMMON_FLAGS} ${CXXFLAGS} `dpu-pkg-config --cflags --libs dpu` \
				-DNR_TASKLETS=${NR_TASKLETS} ${CONFIG_FLAGS}
DPU_FLAGS := ${COMMON_FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS}

all: $(GENERATED_TARGETS) config_check print_config ${HOST_TARGET} ${DPU_TARGET}
	@echo "Build complete: $(BUILD_TYPE) \n"


$(GENERATED_TARGETS): tools/generate.py
	@echo "Generating kernel headers..."
	python3 tools/generate.py

reconfigure:
	@echo "NR_TASKLETS=$(NR_TASKLETS)" > $(CONFIG_STAMP)
	@echo "BACKEND=$(BACKEND)" >> $(CONFIG_STAMP)
	@echo "DEBUG=$(DEBUG)" >> $(CONFIG_STAMP)
	@echo "LOGGING=$(LOGGING)" >> $(CONFIG_STAMP)
	@echo "ENABLE_AUTO_FENCING=$(ENABLE_AUTO_FENCING)" >> $(CONFIG_STAMP)
	@echo "ENABLE_DPU_PRINTING=$(ENABLE_DPU_PRINTING)" >> $(CONFIG_STAMP)
	@echo "CXX_STANDARD=$(CXX_STANDARD)" >> $(CONFIG_STAMP)

cache_old:
	@if [ -f "$(CONFIG_STAMP)" ]; then \
	    rm -f $(CONFIG_STAMP).old; \
		cp -f $(CONFIG_STAMP) $(CONFIG_STAMP).old; \
	fi

config_check: cache_old reconfigure
	@if [ -f "$(CONFIG_STAMP)" ]; then \
	    cmp -s $(CONFIG_STAMP) $(CONFIG_STAMP).old 2>/dev/null || { \
	        echo "Configuration changed, cleaning build..."; \
	        $(MAKE) clean-internal; \
			mkdir -p $(BUILDDIR) && mkdir -p $(BUILDDIR)/bin && mkdir -p $(BUILDDIR)/lib; \
	    }; \
		rm -f $(CONFIG_STAMP).old; \
	fi

${HOST_TARGET}: ${HOST_SOURCES} ${HOST_HEADERS}
	$(CXX) -std=${CXX_STANDARD} -shared -fPIC -o $@ ${HOST_SOURCES} ${HOST_FLAGS} 

${DPU_TARGET}: ${DPU_SOURCES} ${DPU_HEADERS}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES} ${CONFIG_FLAGS}

$(TEST_TARGET): ${TEST_SOURCES} ${HOST_TARGET} ${DPU_TARGET}
	@echo "Building test target: $@"
	$(CXX) -std=${CXX_STANDARD} $(CXXFLAGS) $(COMMON_FLAGS) ${CONFIG_FLAGS} -o $@ $(TEST_SOURCES) -I$(HOST_INCLUDES)  \
		-L$(BUILDDIR)/lib -Wl,-rpath,$(BUILDDIR)/lib -lvectordpu

clean-internal:
	$(RM) -r $(BUILDDIR) $(TEST_TARGET)

clean: clean-internal
	$(RM) -r $(CONFIG_STAMP) $(GENERATED_TARGETS)

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

install: ${DPU_TARGET} ${HOST_TARGET}
	@echo "Installing to $(DESTDIR)..."
	install -d $(bindir) $(libdir) $(includedir)
	install -m 644 $(DPU_TARGET) $(bindir)
	install -m 644 $(HOST_TARGET) $(libdir)
	install -m 644 $(HOST_HEADERS) $(includedir)
	install -m 644 $(DPU_HEADERS) $(includedir)

uninstall:
	@echo "Removing from $(prefix)..."
	rm -f $(bindir)/$(notdir $(DPU_TARGET))
	rm -f $(libdir)/$(notdir $(HOST_TARGET))
	rm -f $(patsubst %,$(includedir)/%,$(notdir $(HOST_HEADERS)))
	rm -f $(patsubst %,$(includedir)/%,$(notdir $(DPU_HEADERS)))