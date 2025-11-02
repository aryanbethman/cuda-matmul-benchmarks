# Makefile for CUDA Matrix Multiplication Project

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -arch=sm_75
CUBLAS_FLAGS = -lcublas

# Directories
SRC_DIR = src
BUILD_DIR = build
SCRIPTS_DIR = scripts
RESULTS_DIR = results

# Source files
VECTOR_ADD_SRC = $(SRC_DIR)/vector_add_fixed.cu
MATMUL_SRC = $(SRC_DIR)/matmul_global.cu

# Executables
VECTOR_ADD_BIN = $(BUILD_DIR)/vector_add_fixed
MATMUL_BIN = $(BUILD_DIR)/matmul_global

# Default target
.PHONY: all clean build vector-add matmul benchmark visualize

all: build

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build vector addition program
vector-add: $(BUILD_DIR) $(VECTOR_ADD_BIN)

$(VECTOR_ADD_BIN): $(VECTOR_ADD_SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Build matrix multiplication program
matmul: $(BUILD_DIR) $(MATMUL_BIN)

$(MATMUL_BIN): $(MATMUL_SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(CUBLAS_FLAGS)

# Build both programs
build: vector-add matmul

# Run benchmarks
benchmark: matmul
	@chmod +x $(SCRIPTS_DIR)/run_benchmarks.sh
	$(SCRIPTS_DIR)/run_benchmarks.sh

# Visualize results (requires Python dependencies)
visualize: benchmark
	@chmod +x $(SCRIPTS_DIR)/visualize_results.py
	python3 $(SCRIPTS_DIR)/visualize_results.py

# Parse results only
parse:
	@chmod +x $(SCRIPTS_DIR)/parse_results.py
	python3 $(SCRIPTS_DIR)/parse_results.py

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(RESULTS_DIR)/*.txt $(RESULTS_DIR)/*.png

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build all programs (default)"
	@echo "  build      - Build both vector-add and matmul programs"
	@echo "  vector-add - Build vector addition program"
	@echo "  matmul     - Build matrix multiplication program"
	@echo "  benchmark  - Run benchmark suite"
	@echo "  visualize  - Run benchmarks and generate visualization"
	@echo "  parse      - Parse benchmark results"
	@echo "  clean      - Remove build artifacts and results"
	@echo "  help       - Show this help message"
