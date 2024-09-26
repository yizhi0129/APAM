CC = nvcc
CFLAGS= -g -G
INCLUDES = 

# the build target executable:
EXE = saxpy.exe 

all: $(EXE)

%.exe: %.o
	$(CC) $(CFLAGS) $^ -o $@ 

%.o: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) $^ -c -o $@

clean:
	$(RM) $(EXE) *.o
	
