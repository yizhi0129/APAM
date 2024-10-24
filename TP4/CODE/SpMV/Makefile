CUDA_ARCH = 61
OPENMP_FLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_$(CUDA_ARCH)

CC = clang $(OPENMP_FLAGS)
CFLAGS = -O3 -Wall
LDFLAGS = -lm -lrt

EXEC_CSR = spmv_csr.exe
EXEC_CSR_SEQ = spmv_csr_seq.exe
EXEC_ELL = spmv_ell.exe
EXEC_ELL_SEQ = spmv_ell_seq.exe

TARGETS = $(EXEC_CSR) $(EXEC_CSR_SEQ) $(EXEC_ELL) $(EXEC_ELL_SEQ)

all : $(TARGETS)

$(EXEC_CSR) : main.o CSRMatrix.o EllMatrix.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(EXEC_CSR_SEQ) : main_seq.o CSRMatrix.o EllMatrix.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(EXEC_ELL) : main_ell.o CSRMatrix.o EllMatrix.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(EXEC_ELL_SEQ) : main_ell_seq.o CSRMatrix.o EllMatrix.o
	$(CC) -o $@ $^ $(LDFLAGS)

main_seq.o : main.c
	$(CC) -D SEQ $(CFLAGS) -o $@ -c $<

main_ell_seq.o : main.c
	$(CC) -D SEQ -D ELL $(CFLAGS) -o $@ -c $<

main_ell.o : main.c
	$(CC) -D ELL $(CFLAGS) -o $@ -c $<

%.o : %.c
	$(CC) $(CFLAGS) -o $@ -c $<

proper :
	rm -f *.o

clean : proper
	rm -f $(TARGETS)
