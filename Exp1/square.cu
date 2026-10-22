#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>

// Prints the numeric code, name and message of a CUDA error, then returns 1.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            const char* name_ = cudaGetErrorName(err_);                       \
            const char* msg_  = cudaGetErrorString(err_);                     \
            fprintf(stderr, "CUDA error at %s:%d -> code %d (%s): %s\n",      \
                    __FILE__, __LINE__, (int)err_,                            \
                    name_ ? name_ : "unknown", msg_ ? msg_ : "no message");   \
            return 1;                                                         \
        }                                                                     \
    } while (0)

__global__ void square(const long long* input, long long* output, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        output[i] = input[i] * input[i];
    }
}

// Returns true if at least one usable CUDA device exists, and prints diagnostics.
static bool checkCudaDevice()
{
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA driver supports: %d.%d | CUDA runtime: %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0)
    {
        const char* name = cudaGetErrorName(e);
        printf("No usable CUDA device (code %d, %s).\n", (int)e, name ? name : "unknown");
        if (driverVersion == 0)
            printf("The NVIDIA driver is missing, or this PC has no NVIDIA GPU.\n");
        else if (driverVersion < runtimeVersion)
            printf("The NVIDIA driver is older than the CUDA Toolkit. Update the driver.\n");
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
    return true;
}

int main()
{
    bool gpuOk = checkCudaDevice();

    long long N;
    printf("\nEnter the value of N: ");
    if (scanf("%lld", &N) != 1 || N <= 0 || N > 100000000LL)
    {
        fprintf(stderr, "Invalid N. Enter a number from 1 to 100000000.\n");
        return 1;
    }

    size_t n = (size_t)N;
    size_t bytes = n * sizeof(long long);

    std::vector<long long> h_input(n), h_output(n);
    for (size_t i = 0; i < n; i++)
    {
        h_input[i] = (long long)i + 1;
    }

    if (gpuOk)
    {
        long long *d_input = nullptr, *d_output = nullptr;

        CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_output, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (int)((n + threadsPerBlock - 1) / threadsPerBlock);

        square<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());       // catches launch errors
        CUDA_CHECK(cudaDeviceSynchronize());  // catches errors during execution

        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
        printf("\nComputed on the GPU.\n");
    }
    else
    {
        // CPU fallback so you still get correct output without a working GPU.
        for (size_t i = 0; i < n; i++)
        {
            h_output[i] = h_input[i] * h_input[i];
        }
        printf("\nComputed on the CPU (GPU not available).\n");
    }

    printf("\nNumber\tSquare\n");
    size_t shown = (n < 50) ? n : 50;
    for (size_t i = 0; i < shown; i++)
    {
        printf("%lld\t%lld\n", h_input[i], h_output[i]);
    }
    if (n > shown)
    {
        printf("... (%zu more values not shown)\n", n - shown);
    }

    return 0;
}
