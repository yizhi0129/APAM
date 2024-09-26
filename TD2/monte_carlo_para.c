#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <TRIALS_PER_THREAD> <num_threads>\n", argv[0]);
        return 1;
    }       
    int TRIALS_PER_THREAD = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    
    // Tableau pour stocker les résultats partiels de chaque thread
    long double *partial_counts = malloc(num_threads * sizeof(long double));
    
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        unsigned int seed = thread_id;  // Utilisation du thread_id comme graine pour rand_r
        long double local_count = 0.0;

        // Lancer un nombre défini de fléchettes par thread
        for (int i = 0; i < TRIALS_PER_THREAD; i ++)
        {
            long double x = (long double)rand_r(&seed) / RAND_MAX;
            long double y = (long double)rand_r(&seed) / RAND_MAX;
            if (x * x + y * y <= 1)
            {
                local_count += 1.0;  // Compte les fléchettes à l'intérieur du cercle
            }
        }
        
        // Stocker le résultat local dans le tableau partagé
        partial_counts[thread_id] = local_count;
    }

    // Calculer la somme des résultats partiels
    long double total_count = 0.0;
    for (int i = 0; i < num_threads; i++)
    {
        total_count += partial_counts[i];
    }

    // Estimer π
    long double pi = (long double) (4 * total_count / (num_threads * TRIALS_PER_THREAD));
    
    printf("Pi = %.17Lf\n", pi);

    // Libération de la mémoire allouée
    free(partial_counts);
    
    return 0;
}
