#ifndef CSRMATRIX_H
#define CSRMATRIX_H
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

float get_elapsedtime(struct timeval start, struct timeval stop);

#define max(a, b) \
  ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a, b) \
  ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

typedef struct
{
  // Number of rows
  int m_nrows;
  // Number of non-zeros elements
  int m_nnz;
  // Non-zeros elements
  double *m_values;
  // Store columns indexes
  int *m_cols;
  // Store first index for m_cols and m_values of the rows fist non-zero element
  int *m_rows;
} CSRMatrix_t;

void init_CSR(CSRMatrix_t *A, int nrows, int nnz);
void destruct_CSR(CSRMatrix_t *A);
void mult_CSR(CSRMatrix_t *A, double const *x, double *y);
void mult_CSR_gpu(CSRMatrix_t *A, double const *x, double *y);
void print_CSR(CSRMatrix_t *A);
void buildLaplacian(CSRMatrix_t *matrix, int nx, int ny);

#endif
