import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import sparse
from surprise import SVD, Dataset, accuracy, Reader, Trainset
from src.preprocessing.dataloader import UserDataset
from src.models.sparse_mf import getPandR, MAE_and_RMSE, predict_with_surprise, get_popularity_preds


def train(masked_df):
    print('factorizing...')
    start = time.time()
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(masked_df, reader)
    trainset = train_data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    end = time.time()
    print('done')
    print('finished running in ', round(end-start), ' seconds')
    return algo

def predict(unmasked_R_coo, mask_coo, algo):
    print('predicting...')
    start = time.time()
    predicted_R_csr = predict_with_surprise(unmasked_R_coo.tocsr(), mask_coo, algo)
    end = time.time()
    print('done')
    print('finished running in ', round(end-start), ' seconds')
    return predicted_R_csr

def eval_with_surprise(masked_df, unmasked_R_coo, mask_coo, mask_csr, ks):
    # train
    algo = train(masked_df)

    # predict
    predicted_R_csr = predict(unmasked_R_coo, mask_coo, algo)
    predicted_R_coo = sparse.coo_matrix(predicted_R_csr)

    # evaluate
    print('predicting...')
    start = time.time()
    MFprecisions, MFrecalls, MFmae, MFrmse = CF_metrics(ks, predicted_R_coo, unmasked_R_coo, mask_csr, mask_coo)
    popular_precisions, popular_recalls, popular_mae, popular_rmse = popularity_metrics(ks, masked_R, unmasked_R_coo, mask_csr, mask_coo)
    random_precisions, random_recalls, random_mae, random_rmse = random_metrics(ks, masked_R, unmasked_R_coo, mask_csr, mask_coo)
    end = time.time()
    print('done')
    print('finished running in ', round(end-start), ' seconds')

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

def popularity_metrics(ks, masked_R, unmasked_R_coo, mask_csr, mask_coo):
    print('calculating popularity metrics...')
    num_users, num_items = masked_R.shape
    masked_R_csr = masked_R.tocsr()
    
    predictions = get_popularity_preds(masked_R_csr, mask_coo)
    predictions_csr = predictions_coo.tocsr()
    ground_truth_csr = unmasked_R_coo.tocsr()

    precisions, recalls = getPandR(ks, predictions, predictions_csr, ground_truth_csr, mask_csr)
    mae, rmse = MAE_and_RMSE(predictions_csr, ground_truth_csr, mask_coo)
    print('done')
    return precisions, recalls, mae, rmse

def random_metrics(ks, masked_R, unmasked_R_coo, mask_csr, mask_coo):
    print('calculating random metrics...')
    predictions_coo = sparse.coo_matrix(np.random.rand(*masked_R.shape) * 5 * mask_coo.toarray())
    predictions_csr = predictions_coo.tocsr()
    ground_truth_csr = unmasked_R_coo.tocsr()

    precisions, recalls = getPandR(ks, predictions_coo, predictions_csr, ground_truth_csr, mask_csr)
    mae, rmse = MAE_and_RMSE(predictions_csr, ground_truth_csr, mask_coo)
    print('done')
    return precisions, recalls, mae, rmse


def CF_metrics(ks, predicted_R, unmasked_R_coo, mask_csr, mask_coo):
    print('calculating CF metrics...')
    predictions_coo = sparse.coo_matrix(predicted_R.toarray() * mask_csr.toarray())
    predictions_csr, ground_truth_csr = predictions_coo.tocsr(), unmasked_R_coo.tocsr()

    precisions, recalls = getPandR(ks, predictions_coo, predictions_csr, ground_truth_csr, mask_csr)
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

    mask = np.logical_xor(unmasked_R, masked_R)
    mask_coo = sparse.coo_matrix(mask)
    mask_csr = mask_coo.tocsr()
    
    masked_R = sparse.coo_matrix(masked_R)
    unmasked_R_coo = sparse.coo_matrix(unmasked_R)

    masked_df = pd.DataFrame(data={'userID': masked_R_coo.row, 'itemID': masked_R_coo.col, 'rating': masked_R_coo.data})
    # unmasked_df = pd.DataFrame(data={'userID': unmasked_R.row, 'itemID': unmasked_R.col, 'rating': unmasked_R.data})

    ks = [3, 5, 10, 20, 30, 40, 50, 75, 100]

    eval_with_surprise(masked_df, unmasked_R_coo, mask_coo, mask_csr, ks)
