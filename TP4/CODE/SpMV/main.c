#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

#include "CSRMatrix.h"

#ifdef ELL
#include "EllMatrix.h"
#endif

// #define DEBUG

int main(int argc, char **argv)
{
  double *x = NULL, *y = NULL;
  int nx = 100, ny = 100;
  int nb_test = 1, i = 0, j = 0;
  int check = 1;
  struct timeval t0, t1;
  double duration = 0.;
  CSRMatrix_t *csr_matrix = NULL;

  if (argc > 1)
    nx = atoi(argv[1]);
  if (argc > 2)
    ny = atoi(argv[2]);
  if (argc > 3)
    nb_test = atoi(argv[3]);
  if (argc > 4)
    check = atoi(argv[4]);
  int nrows = nx * ny;

  fprintf(stdout, "NX: %d\tNY: %d\tNTest: %d\n", nx, ny, nb_test);
  csr_matrix = (CSRMatrix_t *)malloc(sizeof(CSRMatrix_t));
  x = (double *)malloc(nrows * sizeof(double));
  y = (double *)malloc(nrows * sizeof(double));
  for (i = 0; i < nrows; ++i)
  {
    x[i] = 1. * i;
    y[i] = 0.;
  }

  buildLaplacian(csr_matrix, nx, ny);
#ifdef DEBUG
  print_CSR(csr_matrix);
#endif

#ifdef ELL
  EllMatrix_t *ell_matrix = (EllMatrix_t *)malloc(sizeof(EllMatrix_t));
  convert_from_CSR(csr_matrix, ell_matrix);
#endif

  for (i = 0; i < nb_test; ++i)
  {
    gettimeofday(&t0, NULL);
    ;

#ifdef SEQ
#ifndef ELL
    mult_CSR(csr_matrix, x, y);
#else
    mult_Ell(ell_matrix, x, y);
#endif
#else
#ifndef ELL
    mult_CSR_gpu(csr_matrix, x, y);
#else
    mult_Ell_gpu(ell_matrix, x, y);
#endif
#endif

    gettimeofday(&t1, NULL);
    duration += get_elapsedtime(t0, t1);

    double norme = 0.;

    for (j = 0; j < nrows; ++j)
      norme += y[j] * y[j];
    norme = sqrt(norme);
    for (j = 0; j < nrows; ++j)
      x[j] = y[j] / norme;
  }

  if (check)
  {
    double norme = 0.;
    for (i = 0; i < nrows; ++i)
      norme += y[i] * y[i];
    norme = sqrt(norme);
    fprintf(stdout, "NORME Y = %.2f\n", norme);
  }

  fprintf(stdout, " Time : %f\n", duration);
  uint64_t flop_csr = (unsigned long long)(csr_matrix->m_nnz) * 2;
  fprintf(stdout, " MFlops : %.2f\n", flop_csr / (duration / nb_test) * 1E-6);
  fprintf(stdout, "AvgTime : %f\n", duration / nb_test);

  free(x);
  free(y);
  destruct_CSR(csr_matrix);

#ifdef ELL
  destruct_Ell(ell_matrix);
#endif

  return 0;
}
