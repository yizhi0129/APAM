# Indice global

## Inidce global d'un bloc dans une grille 2D

```c
__global__ void cal_ind_global_bloc_2d()
{
    int ind_2d = blockIdx.x + blockIdx.y * blockDim.x;
}
```

## Indice global d'un bloc dans une grille 3D

```c
__global__ void cal_ind_global_bloc_3d()
{
    int ind_3d = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * blockDim.x * blockDim.y;
}
```

## Taille d'un bloc

```c
__global__ void cal_taille_bloc()
{
    int n_threads = blockDim.x * blockDim.y * blockDim.z;
}
```

## Indice d'un thread dans un bloc 3D
```c
__global__ void cal_ind_thread_par_bloc()
{
    int ind_thread_bloc = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x blockDim.y;
}
```

## Indice global d'un thread dans une grille 3D des blocs 3D

```c 
__global__ void cal_ind_thread_global()
{
    int ind_thread_global = ind_thread_bloc + n_threads * ind_3d;
}
```
Il faut juste remplacer `ind_thread_bloc`, `n_threads` et `ind_3d` par leur calculs précédemment correspondants.

# Analyse de td1.cu

## Quelle partie du programme doit s’exécuter sur l’hôte ? Quelle partie sur le device ?

### sur le device : calcul `kernel`
```c
__global__ void kernel(double *a, double *b, double *c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
    	c[i] = a[i] + b[i];
    }
}
```

### sur l'hôte : allocation de mémoire (pour `h_a`, `h_b`et `h_c`) et initialisation (pour `h_a` et `h_b`)
```c
h_a = (double*)malloc(sz_in_bytes);
h_b = (double*)malloc(sz_in_bytes);
h_c = (double*)malloc(sz_in_bytes);
for(int i = 0 ; i < N ; i++)
{
    h_a[i] = 1./(1.+i);
    h_b[i] = (i-1.)/(i+1.);
}
```

### sur le device : allocation de mémoire (pour `d_a`, `d_b` et `d_c`)
```c
    cudaMalloc((void**)&d_a, sz_in_bytes);
    cudaMalloc((void**)&d_b, sz_in_bytes);
    cudaMalloc((void**)&d_c, sz_in_bytes);
```

### sur l'hôte et sur le device : copier les deux vecteurs de CPU vers GPU et le vecteur résultat de GPU vers CPU
```c
cudaMemcpy(d_a, h_a, sz_in_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, sz_in_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost);
```

### sur l'hôte et sur le device : libérer la mémoire (pour `d_a`, `d_b`, `d_c`, `h_a`, `h_b` et `h_c`)
```c
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
free(h_a);
free(h_b);
free(h_c);
```
## Que calcule ce programme ?

Ce programme calcule une addition des deux vecteurs.

## Combien y a-t-il de blocs au total ? Combien de threads par blocs ? Combien de threads au total ?
```c
int N = 1000;
dim3  dimBlock(64, 1, 1);
dim3  dimGrid((N + dimBlock.x - 1)/dimBlock.x, 1, 1);
```
Ici, on définit 64 threads par bloc, et on calcule le nombre de blocs au total par la fonction `dimGrid`, donc on obtient 16. Il y a donc 1024 threads au total (dont seulement 1000 pour le calcul de l'addition vectorielle).  