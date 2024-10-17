#include <mpi.h>
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

int main(int argc, char** argv) 
{
    int rank, size;
    int *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cudaSetDevice(rank);  // Set the CUDA device for this rank

    h_a = (int *)malloc(sizeof(int) * TSIZE);
    h_b = (int *)malloc(sizeof(int) * TSIZE);
    h_c = (int *)malloc(sizeof(int) * TSIZE);

    if (rank == 1) 
    {
        // Initialize data for rank 1
        for (int i = 0; i < TSIZE; i++) 
        {
            h_a[i] = i;            // Example data
            h_b[i] = TSIZE - i;   // Example data
        }

        // Allocate device memory and copy data
        cudaMalloc((void**)&d_a, sizeof(int) * TSIZE);
        cudaMalloc((void**)&d_b, sizeof(int) * TSIZE);
        cudaMalloc((void**)&d_c, sizeof(int) * TSIZE);

        cudaMemcpy(d_a, h_a, sizeof(int) * TSIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(int) * TSIZE, cudaMemcpyHostToDevice);

        // Launch CUDA kernel
        addVector<<<(TSIZE + 255) / 256, 256>>>(d_a, d_b, d_c);
        cudaMemcpy(h_c, d_c, sizeof(int) * TSIZE, cudaMemcpyDeviceToHost);

        // Send results to rank 0
        MPI_Send(h_c, TSIZE, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        // Clean up CUDA
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    } 
    else if (rank == 0) 
    {
        // Rank 0 receives data from rank 1
        MPI_Recv(h_c, TSIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Verification
        int res = verif_tab(h_c, TSIZE);
        if (res == 0) 
        {
            printf("Data received from rank 1 is correct!\n");
        } 
        else 
        {
            printf("Data received from rank 1 has errors!\n");
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    MPI_Finalize();
    return 0;
}
