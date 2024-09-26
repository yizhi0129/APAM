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

## Que calcule ce programme ?

## Combien y a-t-il de blocs au total ? Combien de threads par blocs ? Combien de threads au total ?
```c
dim3  dimBlock(64, 1, 1);
dim3  dimGrid((N + dimBlock.x - 1)/dimBlock.x, 1, 1);
```
Ici, on définit 64 threads par bloc, 