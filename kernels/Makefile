CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc

ARCH ?= sm_86
CXXFLAGS := -O3 -lineinfo -arch=$(ARCH) --ptxas-options=-v
INCLUDES := -Iinclude -Isrc

BIN_DIR := bin
BUILD_DIR := build
RESULTS_DIR := results

TARGET := $(BIN_DIR)/main
SRC := src/main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR) $(BUILD_DIR) $(RESULTS_DIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $^ -o $@ 2> $(RESULTS_DIR)/ptxas.txt

clean:
	rm -rf $(BIN_DIR)/* $(BUILD_DIR)/* $(RESULTS_DIR)/*
