#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }   
    int N;
    long double count = 0.0;
    N = atoi(argv[1]);

    for (int i = 0; i < N; i ++)
    {
        long double x = (long double) (rand() / RAND_MAX);
        long double y = (long double) (rand() / RAND_MAX);
        if (x * x + y * y <= 1)
        {
            count += 1.0;
        }
    }
    long double pi = 4 * (long double) (count / N);
    
    printf("Pi = %.17Lf\n", pi);
    return 0;
}