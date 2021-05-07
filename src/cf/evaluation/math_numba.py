import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def mean_squared_error(x: np.ndarray, y: np.ndarray) -> float:
    squared_error = (x - y) ** 2
    mse = np.mean(squared_error)

    return mse


@njit(fastmath=True)
def mean_absolute_error(x: np.ndarray, y: np.ndarray) -> float:
    abs_error = np.abs(x - y)
    mae = np.mean(abs_error)

    return mae


# @njit(fastmath=True)
# def r2_score(x: np.ndarray, y: np.ndarray) -> float:
#     correlation_matrix = np.corrcoef(x, y)
#     correlation_xy = correlation_matrix[0,1]
#     r_squared = correlation_xy**2

#     return r_squared


@njit(fastmath=True)
def explained_variance_score(x: np.ndarray, y: np.ndarray) -> float:
    evs = 1 - np.cov(x-y) / np.cov(x)

    return evs



# @njit(fastmath=True)
# def log_loss(x: np.ndarray, y: np.ndarray, eps:float) -> float:
#     underflow_idx = (y < eps)
#     y[underflow_idx] = eps
#     log_loss = -np.sum(x * np.log(y))

#     return log_loss


if __name__ == '__main__':
    import sklearn.metrics as metrics

    a = np.array([2,2,3])
    b = np.array([0,2,6])

    print(f"RMSE",
          f"numba: {np.sqrt(mean_squared_error(a,b)):.6f}", 
          f"sklearn: {np.sqrt(metrics.mean_squared_error(a,b)):.6f}", sep='\n')
    print()

    print(f"MAE",
          f"numba: {mean_absolute_error(a,b):.6f}", 
          f"sklearn: {metrics.mean_absolute_error(a,b):.6f}", sep='\n')
    print()

    # print(f"r2_score",
    #       f"numba: {r2_score(a,b):.6f}", 
    #       f"sklearn: {metrics.r2_score(a,b):.6f}", sep='\n')
    # print()

    print(f"explained_variance_score",
          f"numba: {explained_variance_score(a,b):.6f}", 
          f"sklearn: {metrics.explained_variance_score(a,b):.6f}", sep='\n')
    print()

    # print(f"log_loss",
    #       f"numba: {log_loss(a, b, 1e-15):.6f}", 
    #       f"sklearn: {metrics.log_loss(a,b):.6f}", sep='\n')
    # print()
