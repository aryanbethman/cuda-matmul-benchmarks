#!/usr/bin/env python3
"""
Visualize benchmark results comparing custom CUDA kernel vs cuBLAS performance.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Import parse function from parse_results module
from parse_results import parse_benchmark_results


def visualize_benchmark_results(results_file, output_file=None):
    """
    Create visualization of benchmark results using grouped bar charts.
    
    Args:
        results_file: Path to the benchmark_results.txt file
        output_file: Optional path to save the plot. If None, displays the plot.
    """
    matrix_sizes, gpu_times, cublas_times = parse_benchmark_results(results_file)

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Number of matrix sizes
    n = len(matrix_sizes)
    
    # Set up positions for bars
    x = np.arange(n)
    width = 0.35  # Width of the bars

    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, gpu_times, width, label="Custom CUDA Kernel", color='#1f77b4')
    bars2 = ax.bar(x + width/2, cublas_times, width, label="cuBLAS", color='#ff7f0e')

    # Set the x-axis label
    ax.set_xlabel("Matrix Size (N)", fontsize=12)

    # Set the y-axis label
    ax.set_ylabel("Execution Time (ms)", fontsize=12)

    # Set the title of the plot
    ax.set_title("Execution Time vs. Matrix Size for Matrix Multiplication", fontsize=14, fontweight='bold')

    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in matrix_sizes], fontsize=10)

    # Add a legend to the plot
    ax.legend(fontsize=11)

    # Add a grid to the plot (behind bars)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Use log scale for y-axis to better visualize the wide range
    ax.set_yscale('log')

    # Add value labels on bars (optional, but helpful for small values)
    # Only add labels if the bar is tall enough
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > ax.get_ylim()[0] * 10:  # Only label if visible enough
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=7, rotation=90)

    # Uncomment to show values on bars (might be cluttered for many data points)
    # autolabel(bars1)
    # autolabel(bars2)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Display or save the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()  # Close the figure to free memory


def main():
    # Get project root directory (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_file = project_root / "results" / "benchmark_results.txt"

    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Please run the benchmark script first.")
        sys.exit(1)

    # Save plot to results directory
    output_file = project_root / "results" / "benchmark_visualization.png"
    visualize_benchmark_results(results_file, output_file)


if __name__ == "__main__":
    main()
