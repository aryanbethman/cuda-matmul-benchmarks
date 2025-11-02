# CUDA Matrix Multiplication Benchmark

A CUDA-based matrix multiplication project comparing custom GPU kernel performance against cuBLAS library implementations. This project benchmarks matrix multiplication across various sizes (from 2×2 to 4096×4096) and visualizes the performance comparison using grouped bar charts.

## Project Structure

```
matmul-benchmarks-cuda/
├── src/                    # CUDA source files
│   └── matmul_global.cu    # Matrix multiplication kernel implementation
├── scripts/                # Utility scripts
│   ├── run_benchmarks.sh   # Automated benchmark runner
│   ├── parse_results.py    # Results parser
│   └── visualize_results.py # Bar chart visualization generator
├── build/                  # Compiled binaries (generated)
├── results/                # Benchmark results and visualizations
│   ├── benchmark_results.txt
│   └── benchmark_visualization.png
├── Makefile                # Build configuration
├── requirements.txt        # Python dependencies
└── README.md
```

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc compiler)
- Python 3.6+ (for visualization scripts)
- cuBLAS library (usually included with CUDA)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/matmul-benchmarks-cuda.git
cd matmul-benchmarks-cuda
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Verify CUDA installation:
```bash
nvcc --version
```

## Building

### Build All Programs
```bash
make all
# or
make build
```

### Build Individual Programs
```bash
# Build matrix multiplication program
make matmul
```

### Manual Compilation

For matrix multiplication:
```bash
nvcc -arch=sm_75 src/matmul_global.cu -o build/matmul_global -lcublas
```

**Note**: Adjust `-arch=sm_75` to match your GPU's compute capability. Common values:
- `sm_50` - Maxwell (GTX 900 series)
- `sm_60` - Pascal (GTX 10 series)
- `sm_75` - Turing (RTX 20 series, GTX 16 series)
- `sm_80` - Ampere (RTX 30 series)
- `sm_86` - Ampere (RTX 30 series laptop)
- `sm_89` - Ada Lovelace (RTX 40 series)

Find your GPU's compute capability: [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)

## Usage

### Matrix Multiplication

Run matrix multiplication for a specific size:
```bash
./build/matmul_global <matrix_size>
```

Example:
```bash
./build/matmul_global 512
```

### Benchmarking

Run the full benchmark suite:
```bash
make benchmark
```

This will:
1. Compile the matrix multiplication program (if not already compiled)
2. Run benchmarks for matrix sizes: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
3. Save results to `results/benchmark_results.txt`

### Visualization

Generate performance comparison bar charts:
```bash
make visualize
```

This will:
1. Run benchmarks (if not already run)
2. Parse results
3. Generate a grouped bar chart visualization saved to `results/benchmark_visualization.png`

The visualization shows side-by-side comparison of:
- **Custom CUDA Kernel** (blue bars)
- **cuBLAS** (orange bars)

Or parse results manually:
```bash
python3 scripts/parse_results.py
```

Or visualize existing results:
```bash
python3 scripts/visualize_results.py
```

## Implementation Details

### Custom CUDA Kernel

The `matmul_global` kernel uses a naive matrix multiplication algorithm:
- Each thread computes one element of the output matrix
- Thread blocks are organized as 32x32 grids
- Global memory access pattern (no shared memory optimization)

### cuBLAS Comparison

The project includes cuBLAS (`cublasSgemm`) implementation for comparison:
- Industry-standard optimized matrix multiplication
- Single-precision floating point operations
- Uses optimized GPU kernels from NVIDIA

## Results

The benchmark compares:
- **Custom CUDA Kernel**: Naive implementation with global memory access
- **cuBLAS**: Highly optimized NVIDIA library implementation

### Key Findings

Based on the benchmark results included in this repository:

- **Small Matrices (N=2 to N=512)**: The custom CUDA kernel significantly outperforms cuBLAS
  - At N=64: Custom kernel ~0.12ms vs cuBLAS ~5.25ms (~43x faster)
  - At N=512: Custom kernel ~1.01ms vs cuBLAS ~5.51ms (~5x faster)

- **Crossover Point (N=1024)**: Performance becomes comparable
  - Custom kernel: ~6.87ms
  - cuBLAS: ~6.54ms

- **Large Matrices (N=2048+)**: cuBLAS outperforms the custom kernel
  - At N=2048: Custom kernel ~53.75ms vs cuBLAS ~11.92ms (~4.5x faster for cuBLAS)
  - At N=4096: Custom kernel ~323.63ms vs cuBLAS ~26.15ms (~12x faster for cuBLAS)

**Analysis**: The custom kernel benefits from lower overhead for small operations, but cuBLAS's highly optimized algorithms and memory management strategies excel for larger matrices where computational complexity dominates.

See `results/benchmark_visualization.png` for a visual comparison!

## Cleaning

Remove build artifacts and results:
```bash
make clean
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE) (or your preferred license).

## Acknowledgments

- NVIDIA CUDA Toolkit
- cuBLAS library for optimized matrix operations

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [NVIDIA GPU Compute Capabilities](https://developer.nvidia.com/cuda-gpus)
