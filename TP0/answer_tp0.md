# caractéristique des GPUs disponibles (device_query/prog.cu)

La commande `nvidia-smi` donne des informations des GPUs, par exemple :
```txt
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 3080    Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   55C    P2    80W / 320W |   500MiB / 10000MiB  |     25%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      4567      C   ./my_program                   450MiB    |
|    0   N/A  N/A      7891      G   /usr/bin/X                        50MiB   |
+-----------------------------------------------------------------------------+
```

Sur Google Colab, cela donne :
```txt
Thu Oct  3 13:00:33 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
| N/A   43C    P8               9W /  70W |      0MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

Après l'exécution de `prog`, on obtient les résultats suivants:
```
==========  cudaDeviceGetProperties ============  

Device 0: "Tesla T4"
  GPU Clock Rate (MHz):                          1590
  Memory Clock Rate (MHz):                       5001
  Memory Bus Width (bits):                       256
  CUDA Cores/MP:                                 64
  CUDA Cores:                                    2560
  Total amount of global memory:                 15 GB
  Total amount of shared memory per block:       48 kB
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Peak FP32 operations per second in GFLOPS:    363264592


==========  cudaDeviceGetAttribute ============  

Device 0: "Tesla T4"
  Max number of threads per block:              1024
  Max block dimension X:                        1024
  Max block dimension Y:                        1024
  Max block dimension Z:                        64
  Max grid dimension X:                         2147483647
  Max grid dimension Y:                         65535
  Max grid dimension Z:                         65535
  Max shared memory per block:                  49152
  Warp size:                                    32
  Peak clock frequency in kilohertz:            1590000
  Peak memory clock frequency in kilohertz:     5001000
  Global memory bus width in bits:              256
  Size of L2 cache in bytes:                    4194304
  Maximum resident threads per SM:              1024
  Major compute capability version number:      7
  Minor compute capability version number:      5
  Max shared memory per SM in bytes:            65536
  Max number of 32-bit registers per SM:        65536
  Max per block shared mem size on the device:  65536
  Max thread blocks that can reside on a SM:    16
  Peak memory bandwidth in GB/s:                320.06
```

# vérification des erreurs

## `error_checking/exemple1.cu`

### configuration de bloc

On compile et lance le programme `example1.exe`, cela nous retourne une erreur de configuration :
```txt
GPUassert: invalid configuration argument exemple1.cu 40
GPUassert: invalid configuration argument exemple1.cu 44
TEST FAILED !
```
Le problème est dû aux lignes suivantes :
```c
#define THREADS 2048
dim3 dimGrid(TAB_SIZE / THREADS + 1, 1, 1);
init<<<dimGrid, dimBlock>>>(d_a, 1);
kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
```
En effet, le nombre de threads qu'on a défini (2048) dépasse la valeur maximale (1024) du nombre de threads par bloc, et le calcul de `dimGrid` n'est pas correctement défini. On les corrige :
```c
#define THREADS 1024
dim3 dimGrid((TAB_SIZE + THREADS - 1) / THREADS, 1, 1);
```

### erreur synchrone / asynchrone

On rejoute la fonction suivante afin de vérifier si l'erreur apparue est synchrone ou asynchrone. 
```c
void checkError(const char *msg) 
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "CUDA ERROR : %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
```

```c
init<<<dimGrid, dimBlock>>>(d_a, 1);
checkError("Error: d_a initialization");
init<<<dimGrid, dimBlock>>>(d_b, 2);
checkError("Error: d_b initialization");
kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
checkError("Error: sum kernel launch");
checkCudaErrors(cudaDeviceSynchronize());
```

```
CUDA ERROR : Error: d_a initialization : invalid configuration argument
```
## `error_checking/exemple2.cu`

## `error_checking/exemple3.cu`


# compute-sanitizer et CUDA-gdb