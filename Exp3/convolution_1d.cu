#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define MASK_WIDTH 5
#define RADIUS (MASK_WIDTH / 2)
__global__ void convolutionGlobal(
    float *input,
    float *mask,
    float *output,
    int N)
{
    int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i < N)
    {
        float sum = 0.0f;

        for (int j = 0; j < MASK_WIDTH; j++)
        {
            int index = i + j - RADIUS;

            if (index >= 0 && index < N)
            {
                sum += input[index] * mask[j];
            }
        }

        output[i] = sum;
    }
}
__global__ void convolutionShared(
    float *input,
    float *mask,
    float *output,
    int N)
{
    extern __shared__ float shared[];

    int tx = threadIdx.x;

    int i =
        blockIdx.x * blockDim.x +
        tx;

    int sharedIndex = tx + RADIUS;

    // Load main element
    if (i < N)
        shared[sharedIndex] = input[i];
    else
        shared[sharedIndex] = 0.0f;

    // Load left halo
    if (tx < RADIUS)
    {
        int left = i - RADIUS;

        if (left >= 0)
            shared[tx] = input[left];
        else
            shared[tx] = 0.0f;
    }

    // Load right halo
    if (tx < RADIUS)
    {
        int right = i + blockDim.x;

        if (right < N)
            shared[
                sharedIndex + blockDim.x
            ] = input[right];
        else
            shared[
                sharedIndex + blockDim.x
            ] = 0.0f;
    }

    __syncthreads();

    if (i < N)
    {
        float sum = 0.0f;

        for (int j = 0;
             j < MASK_WIDTH;
             j++)
        {
            sum +=
                shared[tx + j] *
                mask[j];
        }

        output[i] = sum;
    }
}
int main()
{
    int N;

    printf("Enter input size N: ");
    scanf("%d", &N);

    size_t size =
        N * sizeof(float);

    // Host memory
    float *h_input =
        (float *)malloc(size);

    float *h_output_global =
        (float *)malloc(size);

    float *h_output_shared =
        (float *)malloc(size);

    float h_mask[MASK_WIDTH] =
    {
        0.2f,
        0.2f,
        0.2f,
        0.2f,
        0.2f
    };

    // Initialize input
    for (int i = 0; i < N; i++)
    {
        h_input[i] =
            (float)(i % 100);
    }

    // Device pointers
    float *d_input;
    float *d_mask;
    float *d_output_global;
    float *d_output_shared;

    cudaMalloc(
        (void **)&d_input,
        size);

    cudaMalloc(
        (void **)&d_mask,
        MASK_WIDTH * sizeof(float));

    cudaMalloc(
        (void **)&d_output_global,
        size);

    cudaMalloc(
        (void **)&d_output_shared,
        size);

    // Copy data to GPU
    cudaMemcpy(
        d_input,
        h_input,
        size,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_mask,
        h_mask,
        MASK_WIDTH * sizeof(float),
        cudaMemcpyHostToDevice);

    // Kernel configuration
    int threadsPerBlock = 256;

    int blocksPerGrid =
        (N + threadsPerBlock - 1) /
        threadsPerBlock;

    // CUDA events
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---------------------------------
    // Global Memory Implementation
    // ---------------------------------

    cudaEventRecord(start);

    convolutionGlobal<<<
        blocksPerGrid,
        threadsPerBlock>>>(
            d_input,
            d_mask,
            d_output_global,
            N);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float globalTime;

    cudaEventElapsedTime(
        &globalTime,
        start,
        stop);

    // ---------------------------------
    // Shared Memory Implementation
    // ---------------------------------

    int sharedMemorySize =
        (threadsPerBlock +
         2 * RADIUS) *
        sizeof(float);

    cudaEventRecord(start);

    convolutionShared<<<
        blocksPerGrid,
        threadsPerBlock,
        sharedMemorySize>>>(
            d_input,
            d_mask,
            d_output_shared,
            N);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float sharedTime;

    cudaEventElapsedTime(
        &sharedTime,
        start,
        stop);

    // Copy results to CPU
    cudaMemcpy(
        h_output_global,
        d_output_global,
        size,
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        h_output_shared,
        d_output_shared,
        size,
        cudaMemcpyDeviceToHost);

    // Verify results
    int correct = 1;

    for (int i = 0; i < N; i++)
    {
        if (fabs(
            h_output_global[i] -
            h_output_shared[i])
            > 0.001)
        {
            correct = 0;
            break;
        }
    }

    printf(
        "\nGlobal Memory Time : %f ms\n",
        globalTime);

    printf(
        "Shared Memory Time : %f ms\n",
        sharedTime);

    printf(
        "Speedup : %f times\n",
        globalTime / sharedTime);

    if (correct)
        printf(
            "Results Match Successfully.\n");
    else
        printf(
            "Results Do Not Match.\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output_global);
    cudaFree(d_output_shared);

    // Free host memory
    free(h_input);
    free(h_output_global);
    free(h_output_shared);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}