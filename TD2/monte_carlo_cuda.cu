#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include <curand_kernel.h>

#define N_BLOCKS 512
#define N_THREADS_PER_BLOCK 256
#define TRIALS_PER_THREAD 10E10

__global__ void monte_carlo(double* pi_d)
{
    // Variables
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int n_success = 0;
    float x, y;
    unsigned long long seed = 1234;
    curandState_t state;
    curand_init(seed, id, 0, &state);

    // Loop over trials
    if (id < n) 
    {   
        for (int i = 0; i < TRIALS_PER_THREAD; i ++)
        {
            x = curand_uniform(&state);
            y = curand_uniform(&state);
            if (x * x + y * y < 1.0)
                n_success ++;
        }
        // Store pi values
        pi_d[id] = (double)n_success * 4.0 / n_test;
    }
}

int main(int argv, char** argc)
{
    // Variables
    double* pi, *pi_d;
    double avg_pi = 0.0;

    int i;
    double start, stop;
    cudaEvent_t start_event, stop_event;
    
    // Allocate memory
    pi = (double*)malloc(N_THREADS_PER_BLOCK * sizeof(double));
    cudaMalloc(&pi_d, N_THREADS_PER_BLOCK * sizeof(double));
    
    // Create events
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    // Start timer
    cudaEventRecord(start_event, 0);
    
    // Launch kernel
    dim3  dimBlock(N_BLOCKS, 1, 1);
    dim3  dimGrid((N_THREADS_PER_BLOCK + dimBlock.x - 1)/dimBlock.x, 1, 1);
    monte_carlo<<<dimGrid, dimBlock>>>(pi_d);
    
    // Copy success to host
    cudaMemcpy(pi, pi_d, N_THREADS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Stop timer
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&start, start_event);
    cudaEventElapsedTime(&stop, stop_event);
    
    // Sequential reduction: need to be parallelized
    for (int i = 0; i < N_THREADS_PER_BLOCK * N_BLOCKS; i ++)
        avg_pi += pi[i];

    avg_pi /= N_THREADS_PER_BLOCK * N_BLOCKS;

    // Free memory
    free(pi);
    cudaFree(pi_d);

    // Print results
    printf("Pi: %f\n", avg_pi);
    printf("Time: %f ms\n", stop - start);
}
