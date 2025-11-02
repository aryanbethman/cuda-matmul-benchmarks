#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addVectors(const int *a, const int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    const int N = 8;
    int h_a[N], h_b[N], h_c[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 10;
    }

    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch enough threads and blocks explicitly
    int threadsPerBlock = 8;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
