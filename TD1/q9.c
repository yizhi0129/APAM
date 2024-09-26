#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Fonction kernel simulant l'exécution sur CPU
void kernel(double *a, double *b, double *c, int N, int blockSize, int gridSize)
{
    for (int blockIdx = 0; blockIdx < gridSize; blockIdx++) 
    {
        for (int threadIdx = 0; threadIdx < blockSize; threadIdx++) 
        {
            // Calcul de l'indice global comme en CUDA
            int i = blockIdx * blockSize + threadIdx;

            if (i < N) 
            {
                c[i] = a[i] + b[i];  // Opération simple d'addition
            }
        }
    }
}

int main(int argc, char **argv)
{
    int N = 1000;
    int sz_in_bytes = N * sizeof(double);

    // Allocation des tableaux sur CPU
    double *h_a, *h_b, *h_c;
    h_a = (double*)malloc(sz_in_bytes);
    h_b = (double*)malloc(sz_in_bytes);
    h_c = (double*)malloc(sz_in_bytes);

    // Initialisation des valeurs dans h_a et h_b
    for (int i = 0; i < N; i ++) 
    {
        h_a[i] = 1.0 / (1.0 + i);
        h_b[i] = (i - 1.0) / (i + 1.0);
    }

    // Emuler les grilles et les blocs (64 threads par bloc ici)
    int blockSize = 64;  // Taille du bloc (équivalent à blockDim.x)
    int gridSize = (N + blockSize - 1) / blockSize;  // Taille de la grille (équivalent à dimGrid.x)

    // Appel de la fonction kernel (simulation de la copie des données vers le GPU)
    kernel(h_a, h_b, h_c, N, blockSize, gridSize);

    // Affichage de quelques résultats pour vérifier
    for (int i = 0; i < 10; i ++) 
    {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    // Libération de la mémoire
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
