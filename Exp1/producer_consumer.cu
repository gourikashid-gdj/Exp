#include <cuda_runtime.h>
#include <stdio.h>

#define N 10

__global__ void producer(
    int *buffer,
    int *count)
{
    int id =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (id < N)
    {
        int position =
            atomicAdd(count, 1);

        if (position < N)
        {
            buffer[position] =
                (id + 1) * 10;
        }
    }
}

__global__ void consumer(
    int *buffer,
    int *output)
{
    int id =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (id < N)
    {
        output[id] = buffer[id];
    }
}

int main()
{
    int h_buffer[N];
    int h_output[N];

    int *d_buffer;
    int *d_output;
    int *d_count;

    cudaMalloc(
        (void **)&d_buffer,
        N * sizeof(int));

    cudaMalloc(
        (void **)&d_output,
        N * sizeof(int));

    cudaMalloc(
        (void **)&d_count,
        sizeof(int));

    cudaMemset(d_count, 0, sizeof(int));

    // Producer kernel
    producer<<<1, N>>>(
        d_buffer,
        d_count);

    cudaDeviceSynchronize();

    // Consumer kernel
    consumer<<<1, N>>>(
        d_buffer,
        d_output);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_buffer,
        d_buffer,
        N * sizeof(int),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(
        h_output,
        d_output,
        N * sizeof(int),
        cudaMemcpyDeviceToHost);

    printf("Produced Data:\n");

    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_buffer[i]);
    }

    printf("\n\nConsumed Data:\n");

    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_output[i]);
    }

    printf("\n");

    cudaFree(d_buffer);
    cudaFree(d_output);
    cudaFree(d_count);

    return 0;
}
