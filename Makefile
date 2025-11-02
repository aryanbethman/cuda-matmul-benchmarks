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
MATMUL_SRC = $(SRC_DIR)/matmul_global.cu

# Executables
MATMUL_BIN = $(BUILD_DIR)/matmul_global

# Default target
.PHONY: all clean build matmul benchmark visualize parse help

all: build

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build matrix multiplication program
matmul: $(BUILD_DIR) $(MATMUL_BIN)

$(MATMUL_BIN): $(MATMUL_SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(CUBLAS_FLAGS)

# Build target (same as matmul for consistency)
build: matmul

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
	@echo "  build      - Build matrix multiplication program"
	@echo "  matmul     - Build matrix multiplication program"
	@echo "  benchmark  - Run benchmark suite"
	@echo "  visualize  - Run benchmarks and generate visualization"
	@echo "  parse      - Parse benchmark results"
	@echo "  clean      - Remove build artifacts and results"
	@echo "  help       - Show this help message"
