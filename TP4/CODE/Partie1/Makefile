all: mpi openmp

CC=nvcc
OMP_COMPILER=gcc

MPILIB=-L /lib/x86_64-linux-gnu/openmpi/lib -lmpi
MPIINC=-I /lib/x86_64-linux-gnu/openmpi/include

OMPLIB=-L /usr/lib/gcc/x86_64-linux-gnu/8/ -lgomp
OMPINC=-I /usr/lib/gcc/x86_64-linux-gnu/8/include


openmp: openmp_cuda_addvector.pgr

openmp_%.pgr: openmp_%.cu
	$(CC) $< -o $@ $(OMPLIB) $(OMPINC) --compiler-options "-fopenmp" --compiler-options "-ccbin=$(OMP_COMPILER)"

mpi: mpi_cuda_addvector.pgr

mpi_%.pgr: mpi_%.cu
	$(CC) $< -o $@ $(MPILIB) $(MPIINC) --compiler-options "-ccbin=$(OMP_COMPILER)"


clean:
	rm -rf *.pgr
