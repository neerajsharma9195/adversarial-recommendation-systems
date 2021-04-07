import os
import time
import progressbar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from src.preprocessing.dataloader import UserDataset
from src.models.mf import matrix_factorization, getPandR, MAE, RMSE


def run_MF(R):
    R = np.array(R)
    N = len(R)
    M = len(R[0])
    K = 500  # hidden dim
    # random initialization of P and Q
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    # factorize R into nP and nQ
    print('factorizing...\r')
    start = time.time()
    nP, nQ = matrix_factorization(R, P, Q, K, steps=50000)
    end = time.time()
    print('done')
    print('finished running in ', round(end-start), ' seconds')
    nR = np.dot(nP, nQ.T)
    return nR, nP, nQ


def evalMF(masked_R, unmasked_R, ks):
    predicted_R, nP, nQ = run_MF(masked_R)
    MFprecisions, MFrecalls, MFmae, MFrmse = CF_metrics(ks, masked_R, predicted_R, unmasked_R)
    popular_precisions, popular_recalls, popular_mae, popular_rmse = popularity_metrics(ks, masked_R, unmasked_R)
    random_precisions, random_recalls, random_mae, random_rmse = random_metrics(ks, masked_R, unmasked_R)
    models = ['Random Recommender', 'Popularity Recommender', 'Collaborative Filter']
    MAPs = [random_precisions, popular_precisions, MFprecisions]
    MARs = [random_recalls, popular_recalls, MFrecalls]
    errors = [[random_mae, random_rmse], [popular_mae, popular_rmse], [MFmae, MFrmse]]
    os.makedirs('results', exist_ok=True)
    plot_MAP(MAPs, models, ks)
    plot_MAR(MARs, models, ks)
    error_labels = ['MAE', 'RMSE']
    tab_data = [[models[i]] + errors[i] for i in range(len(models))]
    print_table(tab_data, error_labels)


############## Evaluation ###################

def popularity_metrics(ks, masked_R, unmasked_R):
    print('calculating popularity metrics...', end='')
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
    error, alt = MAE(predictions, ground_truth)
    print(error, alt)
    rmse = RMSE(predictions, ground_truth)
    print('done')
    return precisions, recalls, error, rmse


def random_metrics(ks, masked_R, unmasked_R):
    print('calculating random metrics...', end='')
    unseen_mask = (masked_R == 0)
    ground_truth = unmasked_R * unseen_mask
    ground_truth_mask = (ground_truth > 0)
    predictions = np.random.rand(*masked_R.shape) * 5 * ground_truth_mask
    precisions, recalls = getPandR(ks, predictions, ground_truth)
    error, alt = MAE(predictions, ground_truth)
    print(error, alt)
    rmse = RMSE(predictions, ground_truth)
    print('done')
    return precisions, recalls, error, rmse


def CF_metrics(ks, masked_R, predicted_R, unmasked_R):
    print('calculating CF metrics...', end='')
    unseen_mask = (masked_R == 0)
    ground_truth = unmasked_R * unseen_mask
    ground_truth_mask = (ground_truth > 0)
    predictions = predicted_R * ground_truth_mask
    precisions, recalls = getPandR(ks, predictions, ground_truth)
    error, alt = MAE(predictions, ground_truth)
    print(error, alt)
    rmse = RMSE(predictions, ground_truth)
    print('done')
    return precisions, recalls, error, rmse


def plot_MAP(MAPs, labels, ks):
    plt.figure(0)
    for i in range(len(MAPs)):
        plt.plot(ks, MAPs[i], label=labels[i])
    plt.title('Mean Average Precision at k (MAP@k)')
    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.legend(loc='upper left')
    plt.savefig('./results/MAPk')
    plt.show()

def plot_MAR(MARs, labels, ks):
    plt.figure(1)
    for i in range(len(MARs)):
        plt.plot(ks, MARs[i], label=labels[i])
    plt.title('Mean Average Recall at k (MAR@k)')
    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.legend(loc='upper left')
    plt.savefig('./results/MARk')
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
     [1.,4.,4.,4.],
     [1.,1.,5.,4.],
    ])

    # print('loading the data...', end='')
    # start = time.time()
    # train_dataset = UserDataset(data_name='food', load_full=True, subset_only=True, masked='full')
    # val_dataset = UserDataset(data_name='food', load_full=True, subset_only=True, masked='partial')
    # masked_R = train_dataset.get_interactions(style="numpy")
    # unmasked_R = val_dataset.get_interactions(style="numpy")
    # end = time.time()
    # print('done')
    # print('downloaded in ', round(end-start), ' seconds')
    
    ks = [3, 5, 10, 20, 30, 40, 50, 75, 100]
    evalMF(masked_R, unmasked_R, ks)