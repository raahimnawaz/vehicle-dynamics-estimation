# Build the C++ edge port.
#
#   make            -> parity + bench binaries (host = x86_64 here)
#   make bench      -> just the benchmark binary
#   make parity     -> just the parity checker
#   make run-bench  -> build and run the benchmark
#   make clean      -> remove build/
#
# To cross-compile for Jetson (aarch64-linux-gnu):
#   make CXX=aarch64-linux-gnu-g++ TARGET_SUFFIX=-aarch64

CXX        ?= g++
CXXFLAGS   ?= -std=c++17 -O3 -ffast-math -funroll-loops -DNDEBUG -Wall -Wextra -Wpedantic
INCLUDES   := -Iinclude
BUILD_DIR  := build
TARGET_SUFFIX ?=

# Windows / MSYS2: ensure g++ has a writeable temp dir (the system one is
# locked-down for the current user). Harmless on Linux/macOS.
export TMP     := $(CURDIR)/$(BUILD_DIR)/tmp
export TMPDIR  := $(CURDIR)/$(BUILD_DIR)/tmp
export TEMP    := $(CURDIR)/$(BUILD_DIR)/tmp

PARITY := $(BUILD_DIR)/parity$(TARGET_SUFFIX)
BENCH  := $(BUILD_DIR)/bench$(TARGET_SUFFIX)

.PHONY: all parity bench clean run-bench size

all: parity bench

parity: $(PARITY)
bench:  $(BENCH)

$(PARITY): src/parity.cc include/vd/ekf.hpp include/vd/matrix.hpp include/vd/pinn.hpp include/vd/pinn_weights.h
	@mkdir -p $(BUILD_DIR)/tmp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ src/parity.cc

$(BENCH): src/bench.cc include/vd/ekf.hpp include/vd/matrix.hpp include/vd/pinn.hpp include/vd/pinn_weights.h
	@mkdir -p $(BUILD_DIR)/tmp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ src/bench.cc

run-bench: $(BENCH)
	@$(BENCH)

size: $(PARITY) $(BENCH)
	@echo "Binary sizes (stripped):"
	@for f in $(PARITY) $(BENCH); do \
		cp $$f $$f.stripped && strip $$f.stripped && \
		ls -l $$f.stripped | awk '{ printf "  %-40s %s bytes\n", $$NF, $$5 }'; \
	done

clean:
	rm -rf $(BUILD_DIR)
