#include <stdio.h>
#include <stdlib.h>
#include "helper_cuda.h"

#define THREADS 1024
#define TAB_SIZE 100000

__global__ void kernel(int *a, int *b, int *c) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < TAB_SIZE) c[tid] = a[tid] + b[tid];
}

__global__ void init(int *a, int value) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < TAB_SIZE) a[tid] = value;
}

// check error type
void checkError(const char *msg, int shift) 
{
    switch (shift)
    {
        case 0:
        {
            cudaError_t errPeek = cudaPeekAtLastError();
            if (errPeek != cudaSuccess) 
            {
                fprintf(stderr, "CUDA ERROR: %s: %s\n", msg, cudaGetErrorString(errPeek));
            }
            break;
        }
        
        case 1:
        {
            cudaError_t errGet = cudaGetLastError();
            if (errGet != cudaSuccess) 
            {
                fprintf(stderr, "CUDA ERROR: %s: %s RESET\n", msg, cudaGetErrorString(errGet));
                //exit(EXIT_FAILURE);
            }
        }
    }
}

int main(int argc, char **argv)
{
    int sz_in_bytes = sizeof(int) * TAB_SIZE;
    int *h_c;
    int res = 0;
    int *d_a, *d_b, *d_c;

    // Allocation on host (malloc)
    h_c = (int *)malloc(sz_in_bytes);

    // Kernel configuration
    dim3 dimBlock(THREADS, 1, 1);
    dim3 dimGrid((TAB_SIZE + THREADS - 1) / THREADS, 1, 1);

    // Allocation on device (cudaMalloc)
    checkCudaErrors(cudaMalloc((void **)&d_a, sz_in_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_b, sz_in_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_c, sz_in_bytes));

    init<<<dimGrid, dimBlock>>>(d_a, 1);
    checkError("d_a initialization", 0);

    init<<<dimGrid, dimBlock>>>(d_b, 2);
    checkError("d_b initialization", 0);

    // Kernel launch
    kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
    checkError("sum kernel launch", 0);

    // Synchronisation pour s'assurer que le kernel est terminé
    checkCudaErrors(cudaDeviceSynchronize());

    // Retrieving data from device (cudaMemcpy)
    checkCudaErrors(cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost));

    // Freeing on device (cudaFree)
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    // computing sum of tab element
    for (int i = 0; i < TAB_SIZE; i++) 
    {
        res += h_c[i];
    }

    // Verifying if
    if (res == 3 * TAB_SIZE) 
    {
        fprintf(stderr, "TEST PASSED !\n");
    }
    else
    {
        fprintf(stderr, "TEST FAILED !\n");
    }
    free(h_c);
    return 0;
}
