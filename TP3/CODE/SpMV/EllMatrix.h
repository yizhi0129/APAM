#ifndef ELLMATRIX_H
#define ELLMATRIX_H
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "CSRMatrix.h"

typedef struct
{
  // Number of rows
  int m_nrows;
  // Number of non-zeros elements
  int m_nnz;
  // Width of the m_values and m_cols rows
  int m_row_width;
  // Non-zeros elements
  double *m_values;
  // Store columns indexes
  int *m_cols;
} EllMatrix_t;

void init_Ell(EllMatrix_t *A, int nrows, int row_width);
void convert_from_CSR(CSRMatrix_t *csr_matrix, EllMatrix_t *ell_matrix);
void mult_Ell(EllMatrix_t *A, double const *x, double *y);
void destruct_Ell(EllMatrix_t *A);

#endif
