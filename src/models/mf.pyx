import time
# import progressbar
import numpy as np
from libc.stdio cimport printf

def matrix_factorization(R, P, Q, K, steps=20, alpha=0.0002, beta=0.02):
    mask = R > 0
    Q = Q.T
    # for step in progressbar.progressbar(range(steps)):
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + R[i][j] - np.dot(P[i,:],Q[:,j])**2
                    for k in range(K):
                        e = e + (beta/2) * (P[i][k]**2 + Q[k][j]**2)
        if e < 0.001:
            break
        if np.allclose(eR*mask, R, rtol=.05):
            print('All non-zero elements were close enough after {} steps. Returned.'.format(step))
            return P, Q.T
    return P, Q.T


def getPandR(ks, predictions, ground_truth):
    sorted_pred_idxs = np.dstack(np.unravel_index(np.argsort(predictions.ravel()), predictions.shape))[0][::-1]
    precisions, recalls = [], []
    for k in ks:
        k_count = 0
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
        for i, j in sorted_pred_idxs:
            if k_count >= k:
                break
            else:
                if ground_truth[i,j] != 0:
                    k_count += 1
                    if ground_truth[i,j] >= 3.5:
                        if predictions[i,j] >= 3.5:
                            true_pos += 1
                        if predictions[i,j] < 3.5:
                            false_neg += 1
                    if ground_truth[i,j] < 3.5:
                        if predictions[i,j] >= 3.5:
                            false_pos += 1
        precision = true_pos / (true_pos + false_pos + .00001)
        recall = true_pos / (true_pos + false_neg + .00001)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls


def RMSE(predictions, ground_truth):
    rmse = 0
    total = 0
    num_users, num_items = ground_truth.shape
    for i in range(num_users):
        for j in range(num_items):
            if ground_truth[i,j] != 0:
                rmse += (predictions[i,j] - ground_truth[i,j])**2
                total += 1
    rmse /= total
    return np.sqrt(rmse)


def MAE(predictions, ground_truth):
    mae = 0
    total = 0
    num_users, num_items = ground_truth.shape
    for i in range(num_users):
        for j in range(num_items):
            if ground_truth[i,j] != 0:
                mae += abs(predictions[i,j] - ground_truth[i,j])
                total += 1
    mae /= total
    alt = np.sum(np.abs(predictions - ground_truth))
    return mae, alt