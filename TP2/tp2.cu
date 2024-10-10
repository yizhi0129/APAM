#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define N_STREAMS 4
#define N_THREADS_PER_BLOCK 256

__global__ void kernel(float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        out[idx] = exp(- abs(out[idx]));
    }
}

void verify(float* out, int size)
{
    float err = 0.0;
    for (int i = 0; i < size; i++)
    {
        err += abs(out[i] - exp( - abs(sin(i * 1.0)) ));
    }
    if (err / size < 1.e-4)
    {
	    fprintf(stdout, "TEST PASSED (error %3.f < 1.e-4)\n", err / size);
    }
    else
    {
	    fprintf(stderr, "TEST FAILED (error %3.f > 1.e-4)\n", err / size);
    }
}

int main(int argc, char** argv)
{
    int size = 1024;
    if (argc == 2)
    {
	    size = atoi(argv[1]);
    }
    size *= N_STREAMS;

    float *tab = (float*)malloc(sizeof(float) * size);
    if (tab == NULL)
    {
        fprintf(stderr, "Bad allocation\n");
        return -1;
    }

    for (int i = 0; i < size; ++i)
    {
        tab[i] = sin(i * 1.0f);
    }

    float *d_tab;
    cudaMalloc((void**)&d_tab, sizeof(float) * size);

    cudaStream_t streams[N_STREAMS];

    for (int i = 0; i < N_STREAMS; i ++)
    {   
        cudaStreamCreate(&streams[i]);
    }

    int chunk_size = size / N_STREAMS;
    dim3 dimBlock(N_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid((chunk_size + dimBlock.x - 1) / dimBlock.x, 1, 1);

    for (int i =0; i < N_STREAMS; i ++)
    {
        int offset = i * chunk_size;
        cudaMemcpyAsync(d_tab + offset, tab + offset, sizeof(float) * chunk_size, cudaMemcpyHostToDevice, streams[i]);
        kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(d_tab + offset, chunk_size);
        cudaMemcpyAsync(tab + offset, d_tab + offset, sizeof(float) * chunk_size, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < N_STREAMS; i ++)
    {
        cudaStreamSynchronize(streams[i]);
    }
    
    verify(tab, size);

    for (int i = 0; i < N_STREAMS; i ++)
    {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_tab); 
    free(tab);
    
    return 0;
}