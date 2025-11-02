#!/bin/bash

# Define the matrix sizes (powers of 2 from 2 to 4096)
Ns=(2 4 8 16 32 64 128 256 512 1024 2048 4096)

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Path to the compiled binary
BINARY="$PROJECT_ROOT/build/matmul_global"
RESULTS_FILE="$RESULTS_DIR/benchmark_results.txt"

# Clear previous results
> "$RESULTS_FILE"

# Loop through each matrix size
for N in "${Ns[@]}"; do
  echo "Running benchmark for N=$N" >> "$RESULTS_FILE"
  # Execute the compiled program with N as argument and append output to the file
  "$BINARY" $N >> "$RESULTS_FILE"
  echo "" >> "$RESULTS_FILE" # Add a newline for separation
done

echo "Benchmarking complete. Results saved to $RESULTS_FILE"
