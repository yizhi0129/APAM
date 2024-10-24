#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <time.h>

typedef struct timespec struct_time;
#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9 * (double)t.tv_nsec)
/** return time in second
 */
double get_elapsedtime(void)
{
  struct_time st;
  int err = gettime(&st);
  if (err != 0)
    return 0;
  return (double)st.tv_sec + get_sub_seconde(st);
}

void axpy(float *y, float *x, float alpha, int n)
{
  // TODO: axpy operation Y = alpha * X + Y
}

int main(int argc, char **argv)
{
  float alpha = 2;
  float *X = NULL, *Y = NULL;
  int N = 1000;
  if (argc > 1)
    N = atoi(argv[1]);

  X = (float *)malloc(sizeof(float) * N);
  Y = (float *)malloc(sizeof(float) * N);

  for (int i = 0; i < N; ++i)
  {
    X[i] = i;
    Y[i] = X[i] + i;
  }

  double t0 = 0., t1 = 0., duration = 0.;
  t0 = get_elapsedtime();
  axpy(Y, X, alpha, N);
  t1 = get_elapsedtime();

  duration = (t1 - t0);

  fprintf(stdout, "    Elapsed Time : %f\n", duration);

  int stop = (N > 5) ? 5 : N;
  for (int i = 0; i < stop; ++i)
    fprintf(stdout, "Y[%d] = %f\n", i, Y[i]);

  return 0;
}
