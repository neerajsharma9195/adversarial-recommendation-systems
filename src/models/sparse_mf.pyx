import time
import progressbar
import numpy as np
from scipy import sparse

def sort_coo(m):
    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: x[2], reverse=True)

def csr_allclose(rows, cols, R, pred_R, tol=1e-5):
    max_diff = 0
    for idx in range(len(rows)):
        i, j = rows[idx], cols[idx]
        diff = R[i, j] - pred_R[i, j]
        if diff > max_diff:
            max_diff = diff
            if max_diff > tol:
                return False
    return True

def predict_with_surprise(unmasked_R_csr, mask_coo, algo):
    result = mask_coo.copy().tocsr()
    for i, j in zip(mask_coo.row, mask_coo.col):
        output = algo.predict(str(i), str(j), r_ui=unmasked_R_csr[i,j])
        result[i,j] = output[3]
    return result

def matrix_factorization(R, P, Q, K, steps=20, alpha=0.0002, beta=0.02):
    Q = Q.T
    R_csr = R.tocsr()
    for step in progressbar.progressbar(range(steps)):
        for i,j,rating in zip(R.row, R.col, R.data):
            eij = rating - np.dot(P[i,:],Q[:,j])
            for k in range(K):
                P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        pred_R = sparse.coo_matrix(np.dot(P,Q))
        pred_R_csr = pred_R.tocsr()
        e = 0
        for i,j in zip(R.row, R.col):
            e += (R_csr[i,j] - pred_R_csr[i,j])**2
            for k in range(K):
                e += (beta/2) * (P[i][k]**2 + Q[k][j]**2)
        if e < 0.005:
            break
        rows, cols = R_csr.nonzero()
        if csr_allclose(rows, cols, R_csr, pred_R_csr, tol=.07):
            print('All non-zero elements were close enough after {} steps. Returned.'.format(step))
            return P, Q.T
    return P, Q.T


def getPandR(ks, predictions, predictions_csr, ground_truth_csr, mask_csr):
    sorted_predictions = sort_coo(predictions)
    precisions, recalls = [], []
    for k in ks:
        k_count = 0
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
        for i, j, v in sorted_predictions:
            if mask_csr[i,j] != 0:
                if k_count >= k:
                    break
                k_count += 1
                if ground_truth_csr[i,j] >= 3.5:
                    if predictions_csr[i,j] >= 3.5:
                        true_pos += 1
                    if predictions_csr[i,j] < 3.5:
                        false_neg += 1
                if ground_truth_csr[i,j] < 3.5:
                    if predictions_csr[i,j] >= 3.5:
                        false_pos += 1
        precision = true_pos / (true_pos + false_pos + .00000001)
        recall = true_pos / (true_pos + false_neg + .00000001)
        precisions.append(round(precision, 8))
        recalls.append(round(recall, 8))
    return precisions, recalls


def MAE_and_RMSE(predictions_csr, ground_truth_csr, mask_coo):
    mae, rmse = 0, 0
    total = mask_coo.nnz
    for i, j in zip(mask_coo.row, mask_coo.col):
        mae += abs(predictions_csr[i,j] - ground_truth_csr[i,j])
        rmse += (predictions_csr[i,j] - ground_truth_csr[i,j])**2
    mae /= total
    rmse /= total
    return mae, np.sqrt(rmse)