#pragma once

#include <iostream>
#include <set>

#include "asgard_kronmult_common.hpp"

namespace asgard::kronmult
{
/*!
 * \internal
 * \brief (internal use only) Indicates how to interpret the alpha/beta scalars.
 *
 * Matrix operations include scalar parameters, e.g., \b beta \b y.
 * Flops can be saved in special cases and those are in turn
 * handled with template parameters and if-constexpr clauses.
 * \endinternal
 */
enum class scalar_case
{
  //! \brief Overwrite the existing output
  zero,
  //! \brief Ignore \b beta and just add to the existing output
  one,
  //! \brief Ignore \b beta and subtract from the existing output
  neg_one,
  //! \brief Scale by \b beta and add the values
  other
};

/*!
 * \brief Performs a batch of kronmult operations using a dense CPU matrix.
 *
 * This is the CPU implementation of the dense case.
 *
 * Takes a matrix where each entry is a Kronecker product and multiplies
 * it by a vector.
 * The matrix has size num_rows by num_rows times num_terms,
 * each row outputs in a tensor represented by a contiguous block within
 * y with size n^d, similarly x is comprised by linearized tensor blocks
 * with size n^d and each consecutive num_terms entries operate on the same
 * block in x.
 *
 * The short notation is that:
 * y[i * n^d ... (i+1) * n^d - 1] = beta * y[i * n^d ... (i+1) * n^d - 1]
 *      + alpha * sum_j sum_k
 *          kron(vA[t][n * n * (elem[j * dimensions] * num_1d_blocks
 *                              + elem[i * dimensions])]
 *               ...
 *              kron(vA[t][(dims - 1) * num_1d_blocks^2 * n * n +
 *                    n * n * (elem[j * dimensions + (dims - 1)] * num_1d_blocks
 *                                  + elem[i * dimensions + (dims - 1)])]
 *          * x[j * n^d ... (j+1) * n^d - 1]
 *
 * i indexes the tensors in y, j the tensors in x,
 * both go from 0 to num_rows - 1
 * t indexes the operator terms (0 to num_terms - 1)
 * vA[t] is the list of coefficients for this term, i.e., the n by n matrices
 * all such matrices are stored in column-major format and stacked by
 * rows inside vA (i.e., there is one row of matrices)
 *
 * \tparam P is float or double
 *
 * \param dimensions must be between 1D and 6D (included)
 * \param n is the size of the problem, e.g., for linear basis n=2
 *        and cubic basis n=4
 *
 * \param num_rows is the number of rows of the matrix
 * \param num_cols is the number of rows of the matrix
 * \param num_terms is the number of operator terms
 * \param elem is the list multi-indexes
 * \param row_offset is the offset inside elem of the first row multi-index
 * \param col_offset is the offset inside elem of the first row multi-index
 * \param vA is an array of arrays that holds all coefficients
 * \param num_1d_blocks is the number of cells in one-dimension
 */
template<typename T>
void cpu_dense(int const dimensions, int const n, int const num_rows,
               int const num_cols, int const num_terms, int const elem[],
               int const row_offset, int const col_offset, T const *const vA[],
               int const num_1d_blocks, T const alpha, T const x[],
               T const beta, T y[]);

/*!
 * \brief Sparse variant for the CPU.
 *
 * The inputs are the same with the exception of the pntr and indx
 * that describe a standard sparse matrix in row-compressed format.
 * The indexes cover the tensor, i.e., for the pair i, indx[pntr[i]]
 * the Y offset is i * tensor-size and the X one is indx[pntr[i]] * tensor-size
 * The length of pntr is num_rows+1 and indx is pntr[num_rows]
 */
template<typename T>
void cpu_sparse(int const dimensions, int const n, int const num_rows,
                int const pntr[], int const indx[], int const num_terms,
                int const iA[], T const vA[], T const alpha, T const x[],
                T const beta, T y[]);

#ifdef ASGARD_USE_CUDA
/*!
 * \brief Performs a batch of kronmult operations using a dense GPU matrix.
 *
 * The arrays iA, vA, x and y are stored on the GPU device.
 * The indexes and scalars alpha and beta are stored on the CPU.
 *
 * \b output_size is the total size of y, i.e., num_rows * n^dimensions
 *
 * \b num_batch is the product num_cols times num_rows
 */
template<typename P>
void gpu_dense(int const dimensions, int const n, int const output_size,
               int64_t const num_batch, int const num_cols, int const num_terms,
               int const elem[], int const row_offset, int const col_offset,
               P const *const vA[], int const num_1d_blocks, P const alpha,
               P const x[], P const beta, P y[]);

/*!
 * \brief Sparse variant for the GPU.
 *
 * The inputs are the same with the exception of the ix and iy that hold the
 * offsets of the tensors for each product in the batch.
 * The tensors for the i-th product are at ix[i] and iy[i] and there no need
 * for multiplication by the tensor-size, also the length of ix[] and iy[]
 * matches and equals num_batch.
 */
template<typename T>
void gpu_sparse(int const dimensions, int const n, int const output_size,
                int const num_batch, int const ix[], int const iy[],
                int const num_terms, int const iA[], T const vA[],
                T const alpha, T const x[], T const beta, T y[]);
#endif

} // namespace asgard::kronmult
