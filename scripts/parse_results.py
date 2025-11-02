#!/usr/bin/env python3
"""
Parse benchmark results from benchmark_results.txt and extract timing data.
"""

import sys
import os
from pathlib import Path


def parse_benchmark_results(results_file):
    """
    Parse benchmark results file and extract timing data.
    
    Args:
        results_file: Path to the benchmark_results.txt file
        
    Returns:
        tuple: (matrix_sizes, gpu_times, cublas_times)
    """
    gpu_times = []
    cublas_times = []
    matrix_sizes = []

    with open(results_file, 'r') as f:
        for line in f:
            if "GPU execution time for N=" in line:
                parts = line.split("N=")[1].split(":")
                n = int(parts[0])
                gpu_time = float(parts[1].split("ms")[0].strip())
                matrix_sizes.append(n)
                gpu_times.append(gpu_time)
            elif "cuBLAS execution time for N=" in line:
                cublas_time = float(line.split("N=")[1].split(":")[1].split("ms")[0].strip())
                cublas_times.append(cublas_time)

    return matrix_sizes, gpu_times, cublas_times


def main():
    # Get project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_file = project_root / "results" / "benchmark_results.txt"

    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Please run the benchmark script first.")
        sys.exit(1)

    matrix_sizes, gpu_times, cublas_times = parse_benchmark_results(results_file)

    print("Matrix Sizes:", matrix_sizes)
    print("GPU Times:", gpu_times)
    print("cuBLAS Times:", cublas_times)

    return matrix_sizes, gpu_times, cublas_times


if __name__ == "__main__":
    main()
