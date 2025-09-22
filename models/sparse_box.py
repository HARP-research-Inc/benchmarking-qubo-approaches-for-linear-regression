# sparse_box.py

import numpy as np
from numpy.random import PCG64, default_rng
from scipy.sparse import csr_matrix, coo_matrix


def build_spd_csr(d: int, density: float, rng, diag_pad: float = 1.0) -> csr_matrix:
    """
    Return symmetric positive definite A (d x d) in CSR with approx off-diagonal density.
    We place random values on upper triangle by probability 'density', mirror to lower,
    then make strictly diagonally dominant (=> SPD).
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0, 1]")
    A = np.zeros((d, d), dtype=float)
    mask = rng.random((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if mask[i, j] < density:
                val = rng.normal(0.0, 1.0)
                A[i, j] = val
                A[j, i] = val
    row_abs_sum = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, row_abs_sum + diag_pad)
    return csr_matrix(A)


def cache_upper_triangle_coo(A_csr: csr_matrix):
    """
    Cache the upper-triangular nonzeros (i <= j) of A in COO arrays.
    Returns arrays I, J, V for fast loops without rebuilding formats each iter.
    """
    coo = A_csr.tocoo()
    I = []
    J = []
    V = []
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if j < i:
            continue
        I.append(i); J.append(j); V.append(v)
    return np.asarray(I, dtype=np.int32), np.asarray(J, dtype=np.int32), np.asarray(V, dtype=float)

