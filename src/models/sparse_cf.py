import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import sparse
from src.preprocessing.dataloader import UserDataset
from src.models.sparse_mf import matrix_factorization, getPandR, MAE_and_RMSE


def run_MF(R):
    N, M = R.shape
    K = 5  # hidden dim

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    print('factorizing...')
    start = time.time()
    nP, nQ = matrix_factorization(R, P, Q, K, steps=5000)
    end = time.time()
    print('done')
    print('finished running in ', round(end-start), ' seconds')
    return nP, nQ


def evalMF(masked_R, unmasked_R, ks, mask_coo):
    P, Q = run_MF(masked_R)
    nR = np.dot(P, Q.T)
    predicted_R = sparse.coo_matrix(nR)
    mask_csr = mask_coo.tocsr()
    
    MFprecisions, MFrecalls, MFmae, MFrmse = CF_metrics(ks, predicted_R, unmasked_R, mask_csr, mask_coo)
    popular_precisions, popular_recalls, popular_mae, popular_rmse = popularity_metrics(ks, masked_R, unmasked_R, mask_csr, mask_coo)
    random_precisions, random_recalls, random_mae, random_rmse = random_metrics(ks, masked_R, unmasked_R, mask_csr, mask_coo)

    models = ['Random Recommender', 'Popularity Recommender', 'Collaborative Filter']
    MAPs = [random_precisions, popular_precisions, MFprecisions]
    MARs = [random_recalls, popular_recalls, MFrecalls]
    errors = [[random_mae, random_rmse], [popular_mae, popular_rmse], [MFmae, MFrmse]]
    os.makedirs('./results', exist_ok=True)
    plot_MAP(MAPs, models, ks)
    plot_MAR(MARs, models, ks)
    error_labels = ['MAE', 'RMSE']
    tab_data = [[models[i]] + errors[i] for i in range(len(models))]
    print_table(tab_data, error_labels)


############## Evaluation ###################

def popularity_metrics(ks, masked_R, unmasked_R, mask_csr, mask_coo):
    print('calculating popularity metrics...')
    num_users, num_items = masked_R.shape
    masked_R_csr = masked_R.tocsr()
    # sum over all users
    most_popular_items = masked_R_csr.sum(axis=0)
    max_value = most_popular_items.max()
    # normalize most popular items to be in the range (1,5)
    most_popular_items = most_popular_items / max_value * 5
    # predict the (same) normalized vetor for all users
    pred_most_popular = np.tile(most_popular_items,(num_users,1))
    # create masked predictions and ground truth
    predictions = sparse.coo_matrix(np.multiply(pred_most_popular, mask_coo.toarray()))
    predictions_csr, ground_truth_csr = predictions.tocsr(), unmasked_R.tocsr()

    precisions, recalls = getPandR(ks, predictions, predictions_csr, ground_truth_csr, mask_csr)
    mae, rmse = MAE_and_RMSE(predictions_csr, ground_truth_csr, mask_coo)
    print('done')
    return precisions, recalls, mae, rmse

def random_metrics(ks, masked_R, unmasked_R, mask_csr, mask_coo):
    print('calculating random metrics...')
    predictions = sparse.coo_matrix(np.random.rand(*masked_R.shape) * 5 * mask_coo.toarray())
    predictions_csr, ground_truth_csr = predictions.tocsr(), unmasked_R.tocsr()

    precisions, recalls = getPandR(ks, predictions, predictions_csr, ground_truth_csr, mask_csr)
    mae, rmse = MAE_and_RMSE(predictions_csr, ground_truth_csr, mask_coo)
    print('done')
    return precisions, recalls, mae, rmse


def CF_metrics(ks, predicted_R, unmasked_R, mask_csr, mask_coo):
    print('calculating CF metrics...')
    predictions = sparse.coo_matrix(predicted_R.toarray() * mask_csr.toarray())
    predictions_csr, ground_truth_csr = predictions.tocsr(), unmasked_R.tocsr()

    precisions, recalls = getPandR(ks, predictions, predictions_csr, ground_truth_csr, mask_csr)
    mae, rmse = MAE_and_RMSE(predictions_csr, ground_truth_csr, mask_coo)
    print('done')
    return precisions, recalls, mae, rmse


def plot_MAP(MAPs, labels, ks):
    plt.figure(0)
    for i in range(len(MAPs)):
        plt.plot(ks, MAPs[i], label=labels[i])
    plt.title('Mean Average Precision at k (MAP@k)')
    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.savefig('./results/MAPk')
    plt.show()

def plot_MAR(MARs, labels, ks):
    plt.figure(1)
    for i in range(len(MARs)):
        plt.plot(ks, MARs[i], label=labels[i])
    plt.title('Mean Average Recall at k (MAR@k)')
    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.savefig('./results/MARk')
    plt.show()

def print_table(tab_data, labels):
    table = tabulate(tab_data, headers=labels, tablefmt="fancy_grid")
    print(table)
    filename = './results/MAE_and_RMSE.txt'
    with open(filename, 'w') as f:
        print(table, file=f)


##############################################


if __name__ == "__main__":
    masked_R = np.array([
     [5.,1.,5.,0.],
     [0.,0.,0.,4.],
     [0.,1.,4.,5.],
     [0.,0.,0.,4.],
     [0.,1.,5.,4.],
     [0.,1.,0.,0.],
     [0.,0.,4.,0.],
     [0.,0.,0.,5.],
     [3.,0.,0.,0.],
     [0.,2.,0.,0.],
    ])

    unmasked_R = np.array([
     [5.,1.,5.,5.],
     [4.,0.,4.,4.],
     [4.,1.,4.,5.],
     [4.,4.,0.,4.],
     [0.,1.,5.,4.],
     [0.,1.,0.,0.],
     [0.,0.,4.,0.],
     [0.,0.,0.,5.],
     [3.,0.,0.,0.],
     [0.,2.,0.,0.],
    ])


    print('loading the data...')
    start = time.time()
    train_dataset = UserDataset(data_name='food', load_full=True, subset_only=True, masked='full')
    val_dataset = UserDataset(data_name='food', load_full=True, subset_only=True, masked='partial')
    masked_R = train_dataset.get_interactions(style="numpy")
    unmasked_R = val_dataset.get_interactions(style="numpy")
    end = time.time()
    print('done')
    print('downloaded in ', round(end-start), ' seconds')

    mask_coo = sparse.coo_matrix(np.logical_xor(unmasked_R, masked_R))
    masked_R = sparse.coo_matrix(masked_R)
    unmasked_R = sparse.coo_matrix(unmasked_R)

    # ks = [3, 5, 10]
    ks = [3, 5, 10, 20, 30, 40, 50, 75, 100]
    evalMF(masked_R, unmasked_R, ks, mask_coo)
