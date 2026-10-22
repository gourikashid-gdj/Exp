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
