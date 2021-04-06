import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import time
import progressbar


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    mask = R > 0
    Q = Q.T
    for step in progressbar.progressbar(range(steps)):
    # for step in range(steps):
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
        if np.allclose(eR*mask, R, rtol=.05):
            print('All non-zero elements were close enough after {} steps. Returned.'.format(step))
            return P, Q.T
    return P, Q.T


def run_MF(R):
    R = numpy.array(R)
    N = len(R)
    M = len(R[0])
    K = 2  # hidden dim
    # random initialization of P and Q
    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)
    # factorize R into nP and nQ
    nP, nQ = matrix_factorization(R, P, Q, K)
    nR = numpy.dot(nP, nQ.T)
    return nR, nP, nQ


def evalMF(masked_R, unmasked_R, ks):
    predicted_R, nP, nQ = run_MF(masked_R)
    MFprecisions, MFrecalls, MFmae, MFrmse = CF_metrics(ks, masked_R, predicted_R, unmasked_R)
    popular_precisions, popular_recalls, popular_mae, popular_rmse = popularity_metrics(ks, masked_R, unmasked_R)
    random_precisions, random_recalls, random_mae, random_rmse = random_metrics(ks, masked_R, unmasked_R)
    MAPs = [MFprecisions, popular_precisions, random_precisions]
    MARs = [MFrecalls, popular_recalls, random_recalls]
    models = ['Random Recommender', 'Popularity Recommender', 'Collaborative Filter']
    plot_MAP(MAPs, models, ks)
    plot_MAR(MARs, models, ks)
    error_labels = ['MAE', 'RMSE']
    errors = [[random_mae, random_rmse], [popular_mae, popular_rmse], [MFmae, MFrmse]]
    tab_data = []
    for i in range(len(models)):
        tab_data.append([models[i]] + errors[i])
    print_table(tab_data, error_labels)


############## Evaluation ###################

def popularity_metrics(ks, masked_R, unmasked_R):
    num_users, num_items = masked_R.shape
    # sum over all users
    most_popular_items = np.sum(masked_R, axis=0)
    max_value = np.max(most_popular_items)
    # normalize most popular items to be in the range (1,5)
    most_popular_items = most_popular_items / max_value * 5
    # predict the (same) normalized vetor for all users
    most_popular = np.tile(most_popular_items,(num_users,1))
    # create masked predictions
    unseen_mask = (masked_R == 0)
    ground_truth = unmasked_R * unseen_mask
    ground_truth_mask = (ground_truth > 0)
    predictions = most_popular * ground_truth_mask
    # get precisions and recalls for all k
    precisions, recalls = getPandR(ks, predictions, ground_truth)
    error = MAE(predictions, ground_truth)
    rmse = RMSE(predictions, ground_truth)
    return precisions, recalls, error, rmse


def random_metrics(ks, masked_R, unmasked_R):
    unseen_mask = (masked_R == 0)
    ground_truth = unmasked_R * unseen_mask
    ground_truth_mask = (ground_truth > 0)
    predictions = np.random.rand(*masked_R.shape) * 5 * ground_truth_mask
    precisions, recalls = getPandR(ks, predictions, ground_truth)
    error = MAE(predictions, ground_truth)
    rmse = RMSE(predictions, ground_truth)
    return precisions, recalls, error, rmse


def CF_metrics(ks, masked_R, predicted_R, unmasked_R):
    unseen_mask = (masked_R == 0)
    ground_truth = unmasked_R * unseen_mask
    ground_truth_mask = (ground_truth > 0)
    predictions = predicted_R * ground_truth_mask
    precisions, recalls = getPandR(ks, predictions, ground_truth)
    error = MAE(predictions, ground_truth)
    rmse = RMSE(predictions, ground_truth)
    return precisions, recalls, error, rmse

def getPandR(ks, predictions, ground_truth):
    sorted_pred_idxs = np.dstack(np.unravel_index(np.argsort(predictions.ravel()), predictions.shape))[0][::-1]
    precisions, recalls = [], []
    for k in ks:
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
        for idx in range(k):
            i, j = sorted_pred_idxs[idx]
            true_pos += 1
            false_pos += 2
            false_neg += 3
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls


def RMSE(predictions, ground_truth):
    rmse = 0
    total = 0
    num_users, num_items = ground_truth.shape
    for i in range(num_users):
        for j in range(num_items):
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
            mae += abs(predictions[i,j] - ground_truth[i,j])
            total += 1
    mae /= total
    return mae


def plot_MAP(MAPs, labels, ks):
    for i in range(len(MAPs)):
        plt.plot(ks, MAPs[i], label=labels[i])
    plt.title('Mean Average Precision at k (MAP@k)')
    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.legend(loc='upper left')
    # plt.savefig('./results/MAP@k')
    plt.show()

def plot_MAR(MARs, labels, ks):
    for i in range(len(MARs)):
        plt.plot(ks, MARs[i], label=labels[i])
    plt.title('Mean Average Recall at k (MAR@k)')
    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.legend(loc='upper left')
    # plt.savefig('./results/MAR@k')
    plt.show()

def print_table(tab_data, labels):
    table = tabulate(tab_data, headers=labels, tablefmt="fancy_grid")
    print(table)


##############################################


if __name__ == "__main__":
    masked_R = np.array([
     [5.,1.,5.,0.],
     [4.,0.,0.,4.],
     [0.,1.,4.,5.],
     [0.,0.,0.,4.],
     [0.,1.,5.,4.],
    ])

    unmasked_R = np.array([
     [5.,1.,5.,5.],
     [4.,1.,4.,4.],
     [4.,1.,4.,5.],
     [4.,1.,4.,4.],
     [5.,1.,5.,4.],
    ])

    # dims = (20,40)
    # mask = np.random.rand(*dims) > .5
    # unmasked_R = np.random.rand(*dims) * 5
    # masked_R = unmasked_R * mask

    ks = [3, 5, 10, 15]
    evalMF(masked_R, unmasked_R, ks)