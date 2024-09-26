#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N_BLOCS 512
#define N_THREADS_PER_BLOC 256

__global__ void ind_1D()
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
}
