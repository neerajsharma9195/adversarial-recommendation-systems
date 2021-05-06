import numpy as np
from numba import njit, prange

@njit(parallel=True)
def getitem_by_row_col(interaction, row_idx, col_idx):
    res = np.empty(row_idx.shape[0])
    for i in prange(row_idx.shape[0]):
        res[i] = interaction[row_idx[i]][col_idx[i]]
    return res
