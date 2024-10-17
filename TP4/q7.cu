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
    for (int i = 0; i < size; i++) 
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
        fprintf(stderr, "Size must be less than %d\n", N_BLOCKS * THREADS_PER_BLOCK);
        return EXIT_FAILURE;
    }

    float *tab_a, *tab_b, *tab_c;

    // Allocate unified memory
    cudaMallocManaged(&tab_a, sizeof(float) * size);
    cudaMallocManaged(&tab_b, sizeof(float) * size);
    cudaMallocManaged(&tab_c, sizeof(float) * size);

    // Initialize data
    for (int i = 0; i < size; i++) 
    {
        srand(i);
        tab_a[i] = rand() % 100; // Utilisation de rand() % 100 pour éviter de très grands nombres
        tab_b[i] = rand() % 100; // Idem
        tab_c[i] = 0.0;
    }

    // Launch kernel
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x, 1, 1); // Ajustement pour le nombre total d'éléments
    multiplyVector<<<dimGrid, dimBlock>>>(tab_a, tab_b, tab_c);

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();

    // Reduce vector
    reduceVector(tab_c, size);

    // Free unified memory
    cudaFree(tab_a);
    cudaFree(tab_b);
    cudaFree(tab_c);

    return 0;
}
