#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 1024
#define N_BLOCKS 1024

__global__ void multiplyVector(float *tab_a, float *tab_b, float *tab_c) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    tab_c[index] = tab_a[index] * tab_b[index];
}

void reduceVector(float *tab, int size) 
{
    float sum = 0;
    for (int i = 0; i < size; i ++) 
    {
        sum += tab[i];
    }
}

int main(int argc, char** argv)
{
    if (argc != 2) 
    {
        fprintf(stderr, "Usage: %s <size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int size = atoi(argv[1]);
    if (size > N_BLOCKS * THREADS_PER_BLOCK) 
    {
        fprintf(stderr, "Size must be less than %d\n", BLOCK_SIZE * THREADS_PER_BLOCK);
        return EXIT_FAILURE;
    }

    float *tab_a, *tab_b, *tab_c;
    float *d_a, *d_b, *d_c;

    // Allocate host memory
    tab_a = (float *)malloc(sizeof(float) * size);
    tab_b = (float *)malloc(sizeof(float) * size);
    tab_c = (float *)malloc(sizeof(float) * size);

    // Initialize data
    for (int i = 0; i < size; i ++) 
    {
        srand(i);
        tab_a[i] = rand() % 100;
        tab_b[i] = rand() % 100;
        tab_c[i] = 0.0;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * size);
    cudaMalloc((void**)&d_b, sizeof(float) * size);
    cudaMalloc((void**)&d_c, sizeof(float) * size);

    // Copy data to device
    cudaMemcpy(d_a, tab_a, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, tab_b, sizeof(float) * size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(N_BLOCKS, 1, 1);
    dim3 dimGrid((THREADS_PER_BLOCK + dimBlock.x - 1) / dimBlock.x, 1, 1);
    multiplyVector<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(tab_c, d_c, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // Reduce vector
    reduceVector(tab_c, size);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(tab_a);
    free(tab_b);
    free(tab_c);

    return 0;
}