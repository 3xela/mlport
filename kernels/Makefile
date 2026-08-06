CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc

ARCH ?= sm_90a
CXXFLAGS := -O3 -lineinfo -arch=$(ARCH) --ptxas-options=-v
INCLUDES := -Iinclude -Isrc

BIN_DIR := bin
BUILD_DIR := build
RESULTS_DIR := results

TARGET := $(BIN_DIR)/main

SRCS := src/main.cu \
        src/harness/axpy.cu \
        src/harness/reduction.cu \
		src/harness/matmul.cu \
        src/kernels/axpy.cu \
		src/kernels/reduction.cu \
		src/kernels/matmul.cu

OBJS := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(SRCS))

HDR := $(wildcard include/*.cuh include/kernels/*.cuh)

all: $(TARGET)

$(BUILD_DIR)/%.o: src/%.cu $(HDR)
	mkdir -p $(dir $@) $(RESULTS_DIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 2>> $(RESULTS_DIR)/ptxas.txt

$(TARGET): $(OBJS)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $^ -o $@

clean:
	rm -rf $(BIN_DIR)/* $(BUILD_DIR)/* $(RESULTS_DIR)/*