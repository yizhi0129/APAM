CC=gcc
FLAGS=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60
EXE=ongpu.exe

all : $(EXE)

$(EXE) : ongpu.o
	$(CC) $(FLAGS) -o $@ $<

ongpu.o : ongpu.c
	$(CC) $(FLAGS) -c -o $@ $<

clean :
	rm -f *.o $(EXE)

.PHONY: all clean

# version clonflict on Googl Colab