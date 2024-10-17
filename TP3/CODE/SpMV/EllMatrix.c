#include "EllMatrix.h"

#define CHECK_IF_PARAM_IS_NULL(param)                                                         \
  if (!param)                                                                                 \
  {                                                                                           \
    fprintf(stderr, "ERROR : %s, parameter is NULL (%s:%d)\n", __func__, __FILE__, __LINE__); \
    fflush(stderr);                                                                           \
    exit(1);                                                                                  \
  }

void init_Ell(EllMatrix_t *A, int nrows, int row_width)
{
  CHECK_IF_PARAM_IS_NULL(A);
  A->m_values = (double *)calloc(nrows * row_width, sizeof(double));
  A->m_cols = (int *)calloc(nrows * row_width, sizeof(int));
  A->m_nrows = nrows;
  A->m_nnz = nrows * row_width;
  A->m_row_width = row_width;
}

void destruct_Ell(EllMatrix_t *A)
{
  CHECK_IF_PARAM_IS_NULL(A);
  CHECK_IF_PARAM_IS_NULL(A->m_cols);
  CHECK_IF_PARAM_IS_NULL(A->m_values);
  free(A->m_values);
  free(A->m_cols);
}

void convert_from_CSR(CSRMatrix_t *csr_matrix, EllMatrix_t *ell_matrix)
{
  CHECK_IF_PARAM_IS_NULL(csr_matrix);
  CHECK_IF_PARAM_IS_NULL(csr_matrix->m_cols);
  CHECK_IF_PARAM_IS_NULL(csr_matrix->m_rows);
  CHECK_IF_PARAM_IS_NULL(csr_matrix->m_values);
  CHECK_IF_PARAM_IS_NULL(ell_matrix);
  ell_matrix->m_nrows = csr_matrix->m_nrows;
  ell_matrix->m_nnz = csr_matrix->m_nnz;
  
  // TODO
  // ell_matrix->m_row_width value
  // ell_matrix->m_values allocation + values
  // ell_matrix->m_cols allocation + values
  ell_matrix->m_row_width = -1;
  ell_matrix->m_values = NULL;
  ell_matrix->m_cols = NULL;

}

void mult_Ell(EllMatrix_t *A, double const *x, double *y)
{
  CHECK_IF_PARAM_IS_NULL(A);
  CHECK_IF_PARAM_IS_NULL(A->m_cols);
  CHECK_IF_PARAM_IS_NULL(A->m_values);
  
  const int N = A->m_nrows;
  const int row_width = A->m_row_width;

  // TODO
}
