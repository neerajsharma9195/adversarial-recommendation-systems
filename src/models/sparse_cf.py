import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import sparse
from surprise import SVD, Dataset, accuracy, Reader, Trainset
from src.preprocessing.dataloader import UserDataset
from src.models.sparse_mf import matrix_factorization, getPandR, MAE, RMSE, predict_with_surprise


def run_MF(R):
    N, M = R.shape
    K = 500  # hidden dim

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    print('factorizing...')
    start = time.time()
    nP, nQ = matrix_factorization(R, P, Q, K, steps=50000)
    end = time.time()
    print('done')
    print('finished running in ', round(end-start), ' seconds')
    return nP, nQ


def evalMF(masked_R, unmasked_R, ks, mask):
    P, Q = run_MF(masked_R)
    nR = np.dot(P, Q.T)
    predicted_R = sparse.coo_matrix(nR)
    
    MFprecisions, MFrecalls, MFmae, MFrmse = CF_metrics(ks, masked_R, predicted_R, unmasked_R, mask)
    popular_precisions, popular_recalls, popular_mae, popular_rmse = popularity_metrics(ks, masked_R, unmasked_R, mask)
    random_precisions, random_recalls, random_mae, random_rmse = random_metrics(ks, masked_R, unmasked_R, mask)

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


def eval_with_surprise(masked_df, unmasked_R, mask_coo, ks):
    # train
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(masked_df, reader)
    trainset = train_data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    predictions_csr = predict_with_surprise(unmasked_R.tocsr(), mask_coo, algo)
    predicted_R = sparse.coo_matrix(predictions_csr)
    # eval
    mask = mask_coo.toarray()
    MFprecisions, MFrecalls, MFmae, MFrmse = CF_metrics(ks, masked_R, predicted_R, unmasked_R, mask)
    popular_precisions, popular_recalls, popular_mae, popular_rmse = popularity_metrics(ks, masked_R, unmasked_R, mask)
    random_precisions, random_recalls, random_mae, random_rmse = random_metrics(ks, masked_R, unmasked_R, mask)

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

def popularity_metrics(ks, masked_R, unmasked_R, mask):
    print('calculating popularity metrics...')
    num_users, num_items = masked_R.shape
    masked_csr = masked_R.tocsr()
    # sum over all users
    most_popular_items = masked_csr.sum(axis=0)
    max_value = most_popular_items.max()
    # normalize most popular items to be in the range (1,5)
    most_popular_items = most_popular_items / max_value * 5
    # predict the (same) normalized vetor for all users
    pred_most_popular = np.tile(most_popular_items,(num_users,1))
    # create masked predictions and ground truth
    predictions = sparse.coo_matrix(np.multiply(pred_most_popular, mask))
    ground_truth = sparse.coo_matrix(unmasked_R.toarray() * mask)
    predictions_csr, ground_truth_csr = predictions.tocsr(), ground_truth.tocsr()
    # get precisions and recalls for all k
    precisions, recalls = getPandR(ks, predictions, ground_truth, predictions_csr, ground_truth_csr)
    error = MAE(predictions_csr, ground_truth_csr)
    rmse = RMSE(predictions_csr, ground_truth_csr)
    print('done')
    return precisions, recalls, error, rmse

def random_metrics(ks, masked_R, unmasked_R, mask):
    print('calculating random metrics...')
    ground_truth = sparse.coo_matrix(unmasked_R.toarray() * mask)
    predictions = sparse.coo_matrix(np.random.rand(*masked_R.shape) * 5 * mask)
    predictions_csr, ground_truth_csr = predictions.tocsr(), ground_truth.tocsr()

    precisions, recalls = getPandR(ks, predictions, ground_truth, predictions_csr, ground_truth_csr)
    error = MAE(predictions_csr, ground_truth_csr)
    rmse = RMSE(predictions_csr, ground_truth_csr)
    print('done')
    return precisions, recalls, error, rmse


def CF_metrics(ks, masked_R, predicted_R, unmasked_R, mask):
    print('calculating CF metrics...')
    ground_truth = sparse.coo_matrix(unmasked_R.toarray() * mask)
    predictions = sparse.coo_matrix(predicted_R.toarray() * mask)
    predictions_csr, ground_truth_csr = predictions.tocsr(), ground_truth.tocsr()

    precisions, recalls = getPandR(ks, predictions, ground_truth, predictions_csr, ground_truth_csr)
    error = MAE(predictions_csr, ground_truth_csr)
    rmse = RMSE(predictions_csr, ground_truth_csr)
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

    mask = np.logical_xor(unmasked_R, masked_R)
    masked_R = sparse.coo_matrix(masked_R)
    unmasked_R = sparse.coo_matrix(unmasked_R)

    masked_df = pd.DataFrame(data={'userID': masked_R.row, 'itemID': masked_R.col, 'rating': masked_R.data})
    # unmasked_df = pd.DataFrame(data={'userID': unmasked_R.row, 'itemID': unmasked_R.col, 'rating': unmasked_R.data})

    ks = [3, 5, 10]
    # ks = [3, 5, 10, 20, 30, 40, 50, 75, 100]
    # evalMF(masked_R, unmasked_R, ks, mask)

    mask_coo = sparse.coo_matrix(mask)
    eval_with_surprise(masked_df, unmasked_R, mask_coo, ks)