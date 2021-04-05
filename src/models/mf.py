from pprint import pprint

import numpy
import numpy as np
import pandas as pd


def matrix_factorization(R, P, Q, K, steps=50000, alpha=0.0002, beta=0.02):
    mask = R > 0
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
        if np.allclose(eR*mask, R, rtol=.01):
            print('All non-zero elements were close enough after {} steps. Returned.'.format(step))
            return P, Q.T

    return P, Q.T


def run_MF(R):
    R = numpy.array(R)
    N = len(R)
    M = len(R[0])
    K = 2  # hidden dim

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)
    nR = numpy.dot(nP, nQ.T)
    return nR, nP, nQ


############## Evaluation ###################
def eval(k, masked_R, predicted_R, unmasked_R):
    num_users, num_items = unmasked_R.shape
    unseen_mask = (masked_R == 0)
    ground_truth = unmasked_R * unseen_mask
    ground_truth_mask = (ground_truth > 0)
    predictions = predicted_R * ground_truth_mask
    sorted_pred_idxs = np.dstack(np.unravel_index(np.argsort(predictions.ravel()), predictions.shape))[0][::-1]
    print(predictions)
    for idx in range(k):
        i, j = sorted_pred_idxs[idx]
        print(predictions[i, j])
    true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
    # for i in range(num_users):
    #     for j in range(num_items):
    #         pass
    return 0, 0
                    

    print(true_pos, false_pos, true_neg)


def RMSE(masked_R, predicted_R, unmasked_R):
    return 0


def MAE(masked_R, predicted_R, unmasked_R):
    return 0

##############################################


if __name__ == "__main__":
    masked_R = np.array([
     [5.,3.,0.,1.],
     [0.,0.,0.,1.],
     [1.,1.,0.,5.],
     [1.,0.,0.,4.],
     [0.,0.,5.,4.],
    ])

    unmasked_R = np.array([
     [5.,3.,0.,1.],
     [4.,0.,0.,1.],
     [1.,1.,0.,5.],
     [1.,0.,0.,4.],
     [0.,1.,5.,4.],
    ])

    predicted_R, nP, nQ = run_MF(masked_R)
    k = 3
    precision, recall = eval(k, masked_R, predicted_R, unmasked_R)
    print('P@{} = {}'.format(k, precision))
    print('R@{} = {}'.format(k, recall))
    print('RMSE = ', RMSE(masked_R, predicted_R, unmasked_R))
    print('MAP = ', MAE(masked_R, predicted_R, unmasked_R))