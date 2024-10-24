all: mpi openmp

CC=nvcc

openmp: openmp_cuda_addvector.pgr

openmp_%.pgr: openmp_%.cu
	$(CC) $< -o $@ 

mpi: mpi_cuda_addvector.pgr

mpi_%.pgr: mpi_%.cu
	$(CC) $< -o $@

clean:
	rm -rf *.pgr
