#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TSIZE 1024

__global__ void addVector(int *tab_a, int *tab_b, int *tab_c) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < TSIZE) 
    {
        tab_c[index] = tab_a[index] + tab_b[index];
    }
}

int verif_tab(int *tab, int size) 
{
    int ret = 0;
    for (int i = 0; i < size; i++) 
    {
        if (tab[i] != size) 
        {
            ret = 1;
            printf("Error at index %d: tab[%d] = %d\n", i, i, tab[i]);
        }
    }
    return ret;
}

int main(int argc, char* argv[]) 
{
    CUdevice device;
    CUcontext context;
    CUresult result;

    // Initialize the CUDA driver API
    result = cuInit(0);
    if (result != CUDA_SUCCESS) 
    {
        fprintf(stderr, "Error initializing CUDA\n");
        return EXIT_FAILURE;
    }

    // Get the device 0
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) 
    {
        fprintf(stderr, "Error getting device 0\n");
        return EXIT_FAILURE;
    }

    // Create a context for the device
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) 
    {
        fprintf(stderr, "Error creating context\n");
        return EXIT_FAILURE;
    }

    // Allocate host memory
    int *tab_a = (int *)malloc(sizeof(int) * TSIZE);
    int *tab_b = (int *)malloc(sizeof(int) * TSIZE);
    int *tab_c = (int *)malloc(sizeof(int) * TSIZE);
    int *d_a, *d_b, *d_c;

    struct timeval cuda_start, cuda_stop;

    // Initialize data
    for (int i = 0; i < TSIZE; i++) 
    {
        tab_a[i] = i;
        tab_b[i] = TSIZE - i;
        tab_c[i] = 0;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(int) * TSIZE);
    cudaMalloc((void**)&d_b, sizeof(int) * TSIZE);
    cudaMalloc((void**)&d_c, sizeof(int) * TSIZE);

    // Measure GPU time
    gettimeofday(&cuda_start, NULL);
    cudaMemcpy(d_a, tab_a, sizeof(int) * TSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, tab_b, sizeof(int) * TSIZE, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    addVector<<<(TSIZE + 255) / 256, 256>>>(d_a, d_b, d_c);

    // Copy results back to host
    cudaMemcpy(tab_c, d_c, sizeof(int) * TSIZE, cudaMemcpyDeviceToHost);
    gettimeofday(&cuda_stop, NULL);

    // Verify results
    int res = verif_tab(tab_c, TSIZE);
    if (res == 0) 
    {
        printf("GPU Compute is OK; time is %f s\n",
               (float)(cuda_stop.tv_sec - cuda_start.tv_sec) +
               ((float)(cuda_stop.tv_usec - cuda_start.tv_usec)) / 1000000);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(tab_a);
    free(tab_b);
    free(tab_c);

    // Destroy the context
    cuCtxDestroy(context);

    return 0;
}
