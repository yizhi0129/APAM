#include <omp.h>
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
    int size, i;

    // Determine the number of threads (OpenMP)
    #pragma omp parallel
    {
        size = omp_get_num_threads(); // Get number of threads
    }

    // Allocate host memory
    int *tab_a = (int *)malloc(sizeof(int) * TSIZE * size);
    int *tab_b = (int *)malloc(sizeof(int) * TSIZE * size);
    int *tab_c = (int *)malloc(sizeof(int) * TSIZE * size);
    int *tab_d = (int *)malloc(sizeof(int) * TSIZE * size);

    struct timeval cpu_start, cpu_stop;
    struct timeval cuda_start, cuda_stop;

    // Initialize data
    for (i = 0; i < TSIZE * size; i++) 
    {
        tab_a[i] = i;
        tab_b[i] = TSIZE * size - i;
        tab_c[i] = 0;
    }

    // Measure CPU time with OpenMP
    gettimeofday(&cpu_start, NULL);
    
    #pragma omp parallel for
    for (i = 0; i < TSIZE * size; i++) 
    {
        tab_c[i] = tab_a[i] + tab_b[i]; // CPU computation using OpenMP
    }

    gettimeofday(&cpu_stop, NULL);

    // Verify CPU results
    int res = verif_tab(tab_c, TSIZE * size);
    if (res == 0) 
    {
        printf("CPU Compute is OK; time is %f s\n",
               (float)(cpu_stop.tv_sec - cpu_start.tv_sec) +
               ((float)(cpu_stop.tv_usec - cpu_start.tv_usec)) / 1000000);
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(int) * TSIZE * size);
    cudaMalloc((void**)&d_b, sizeof(int) * TSIZE * size);
    cudaMalloc((void**)&d_c, sizeof(int) * TSIZE * size);

    // Measure GPU time
    gettimeofday(&cuda_start, NULL);
    cudaMemcpy(d_a, tab_a, sizeof(int) * TSIZE * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, tab_b, sizeof(int) * TSIZE * size, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    addVector<<<(TSIZE * size + 255) / 256, 256>>>(d_a, d_b, d_c);

    // Copy results back to host
    cudaMemcpy(tab_d, d_c, sizeof(int) * TSIZE * size, cudaMemcpyDeviceToHost);
    gettimeofday(&cuda_stop, NULL);

    // Verify GPU results
    res = verif_tab(tab_d, TSIZE * size);
    if (res == 0) 
    {
        printf("GPU Compute is OK; time is %f s\n",
               (float)(cuda_stop.tv_sec - cuda_start.tv_sec) +
               ((float)(cuda_stop.tv_usec - cuda_start.tv_usec)) / 1000000);
    }

    // Free allocated memory
    free(tab_a);
    free(tab_b);
    free(tab_c);
    free(tab_d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
