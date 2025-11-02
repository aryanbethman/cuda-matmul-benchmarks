#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <chrono> // For CPU timing
#include <cublas_v2.h> // Include cuBLAS header

__global__ void matmul_global(float* a, float* b, float* c, int N){

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row<N && col<N){
    float val = 0.0f;
    for(int k=0; k<N;k++){
      val += a[row*N+k]*b[k*N+col];
    }
    c[row*N+col] = val;
  }

}

// CPU matrix multiplication function
void matmul_cpu(float* a, float* b, float* c, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float val = 0.0f;
            for (int k = 0; k < N; k++) {
                val += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = val;
        }
    }
}


void init_matrix(float *mat, int N) {
    // Initialize random seed
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        mat[i] = static_cast<float>(rand() % 10); // Generate random numbers between 0 and 9 as floats
    }
}

int main(int argc, char **argv){ // Modified main signature
  if (argc != 2) { // Check for correct number of arguments
    printf("Usage: %s <matrix_size_N>\n", argv[0]);
    return 1;
  }

  const int N = atoi(argv[1]); // Convert argument to integer

  float* h_a = (float*)malloc(N*N*sizeof(float));
  float* h_b = (float*)malloc(N*N*sizeof(float));
  // float* h_c_cpu = (float*)malloc(N*N*sizeof(float)); // Separate buffer for CPU result
  float* h_c_gpu = (float*)malloc(N*N*sizeof(float)); // Separate buffer for GPU result
  float* h_c_cublas = (float*)malloc(N*N*sizeof(float)); // Buffer for cuBLAS result


  init_matrix(h_a,N);
  init_matrix(h_b,N);

  // CPU Timing
  // auto start_cpu = std::chrono::high_resolution_clock::now();
  // matmul_cpu(h_a, h_b, h_c_cpu, N);
  // auto end_cpu = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;


  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  cudaMalloc((void**)&d_a, N*N*sizeof(float));
  cudaMalloc((void**)&d_b, N*N*sizeof(float));
  cudaMalloc((void**)&d_c, N*N*sizeof(float));

  cudaMemcpy(d_a, h_a, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N*N*sizeof(float), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 threadsPerBlock(32, 32);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

  auto start_gpu = std::chrono::high_resolution_clock::now();
  matmul_global<<<blocksPerGrid, threadsPerBlock>>>(d_a,d_b,d_c,N);
  cudaDeviceSynchronize();
  auto end_gpu = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed_gpu = end_gpu-start_gpu;

  cudaError_t err = cudaGetLastError();
  if(err!=cudaSuccess){
    printf("Kernel err: %s\n",cudaGetErrorString(err));
    return 1;
  }

  cudaMemcpy(h_c_gpu, d_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  // cuBLAS
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
      printf("cuBLAS initialization failed\n");
      return 1;
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  auto start_cublas = std::chrono::high_resolution_clock::now();
  // Perform matrix multiplication using cublasSgemm (single precision float)
  // C = alpha * A * B + beta * C
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_a, N, d_b, N, &beta, d_c, N);
  cudaDeviceSynchronize(); // Synchronize to measure cuBLAS time accurately
  auto end_cublas = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_cublas = end_cublas - start_cublas;

  if (status != CUBLAS_STATUS_SUCCESS) {
      printf("cuBLAS Sgemm failed: %d\n", status);
      return 1;
  }

  cudaMemcpy(h_c_cublas, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  cublasDestroy(handle);


  // printf("CPU execution time: %f ms\n", elapsed_cpu.count()*1000);
  printf("GPU execution time for N=%d: %f ms\n", N, elapsed_gpu.count()*1000);
  printf("cuBLAS execution time for N=%d: %f ms\n", N, elapsed_cublas.count()*1000);


  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Free host memory
  free(h_a);
  free(h_b);
  // free(h_c_cpu);
  free(h_c_gpu);
  free(h_c_cublas);


  return 0;
}
