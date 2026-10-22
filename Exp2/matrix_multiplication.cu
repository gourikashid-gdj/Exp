#include <stdio.h>1
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
void matrixMultiplyCPU(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] *
                       B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}
Each output element is calculated sequentially using a row of matrix A and a column of matrix B.

__global__ void matrixMultiplyGPU(
    float *A,
    float *B,
    float *C,
    int N)
{
    int col =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int row =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] *
                   B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
int main()
{
    int N;

    printf("Enter matrix size N: ");
    scanf("%d", &N);

    size_t size = N * N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_CPU = (float *)malloc(size);
    float *h_C_GPU = (float *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // -------------------------
    // CPU Execution
    // -------------------------

    auto cpu_start =
        std::chrono::high_resolution_clock::now();

    matrixMultiplyCPU(
        h_A,
        h_B,
        h_C_CPU,
        N);

    auto cpu_end =
        std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double>(
            cpu_end - cpu_start).count();

    // -------------------------
    // GPU Memory Allocation
    // -------------------------

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(
        d_A,
        h_A,
        size,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_B,
        h_B,
        size,
        cudaMemcpyHostToDevice);

    // -------------------------
    // GPU Configuration
    // -------------------------

    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) /
            threadsPerBlock.x,

        (N + threadsPerBlock.y - 1) /
            threadsPerBlock.y);

    // -------------------------
    // GPU Timing
    // -------------------------

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMultiplyGPU<<<
        blocksPerGrid,
        threadsPerBlock>>>(
            d_A,
            d_B,
            d_C,
            N);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float gpu_time_ms;

    cudaEventElapsedTime(
        &gpu_time_ms,
        start,
        stop);

    double gpu_time =
        gpu_time_ms / 1000.0;

    // Copy GPU result to CPU

    cudaMemcpy(
        h_C_GPU,
        d_C,
        size,
        cudaMemcpyDeviceToHost);

    // -------------------------
    // GFLOP/s Calculation
    // -------------------------

    double operations =
        2.0 * N * N * N;

    double cpu_gflops =
        operations /
        (cpu_time * 1e9);

    double gpu_gflops =
        operations /
        (gpu_time * 1e9);

    double speedup =
        cpu_time / gpu_time;

    printf("\nCPU Execution Time : %f seconds\n",
           cpu_time);

    printf("GPU Execution Time : %f seconds\n",
           gpu_time);

    printf("\nCPU Performance : %f GFLOP/s\n",
           cpu_gflops);

    printf("GPU Performance : %f GFLOP/s\n",
           gpu_gflops);

    printf("\nGPU Speedup : %f times\n",
           speedup);

    // -------------------------
    // Verify Results
    // -------------------------

    int correct = 1;

    for (int i = 0; i < N * N; i++)
    {
        if (fabs(h_C_CPU[i] -
                 h_C_GPU[i]) > 0.01)
        {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("\nCPU and GPU results match.\n");
    else
        printf("\nResults do not match.\n");

    // -------------------------
    // Free Memory
    // -------------------------

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}


