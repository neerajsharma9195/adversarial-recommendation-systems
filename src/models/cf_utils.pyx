# cython: language_level=3
import os
import time
import progressbar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from surprise import Dataset, accuracy, Reader, Trainset
from collections import defaultdict
from tabulate import tabulate
from src.preprocessing.dataloader import UserDataset


#################################################################
#                           Evaluation                          #
#################################################################

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def precision_recall_at_k(predictions, k=10, avg=True, threshold=3.5):
    """Return precision and recall at k metrics for each user (or average over all users)"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    P, R = 0, 0
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        P += n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        
        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        R += recalls[uid]

    P /= len(user_est_true.items())
    R /= len(user_est_true.items())
    if avg == False:
        return precisions, recalls 
    else:
        return P, R


#################################################################
#                   Printing and Plotting                       #
#################################################################

def show_and_save(models, aug):
    ks = models[0].ks
    labels = [model.name for model in models]
    errors = [[model.mae, model.rmse] for model in models]
    MAPs = [model.MAPs for model in models]
    MARs = [model.MARs for model in models]

    cold_errors = [[model.cold_mae, model.cold_rmse] for model in models]
    cold_MAPs = [model.cold_MAPs for model in models]
    cold_MARs = [model.cold_MARs for model in models]

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/cold_start', exist_ok=True)
    os.makedirs('results/all_users', exist_ok=True)

    plot_MAP(MAPs, labels, ks, aug)
    plot_MAR(MARs, labels, ks, aug)
    error_labels = ['all_users'] + ['MAE', 'RMSE']
    tab_data = [[labels[i]] + errors[i] for i in range(len(labels))]
    print_table(tab_data, error_labels, aug)

    plot_MAP(cold_MAPs, labels, ks, aug, cold_start=True)
    plot_MAR(cold_MARs, labels, ks, aug, cold_start=True)
    error_labels = ['cold_users'] + ['MAE', 'RMSE']
    tab_data = [[labels[i]] + cold_errors[i] for i in range(len(labels))]
    print_table(tab_data, error_labels, aug, cold_start=True)

def plot_MAP(MAPs, labels, ks, aug, cold_start=False):
    for i in range(len(MAPs)):
        plt.plot(ks, MAPs[i], label=labels[i])
    if cold_start:
        plt.title('Mean Average Precision at k (MAP@k) for Cold Start Users')
        user_type = 'cold_start'
    else:
        plt.title('Mean Average Precision at k (MAP@k)')
        user_type = 'all_users'
    aug_type = 'aug_' if aug is True else ''
    file_loc = f'./results/{user_type}/{aug_type}MAPk'
    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.savefig(file_loc)
    plt.close()

def plot_MAR(MARs, labels, ks, aug, cold_start=False):
    for i in range(len(MARs)):
        plt.plot(ks, MARs[i], label=labels[i])
    if cold_start:
        plt.title('Mean Average Recall at k (MAR@k) for Cold Start Users')
        user_type = 'cold_start'
    else:
        plt.title('Mean Average Recall at k (MAR@k)')
        user_type = 'all_users'
    aug_type = 'aug_' if aug is True else ''
    file_loc = f'./results/{user_type}/{aug_type}MARk'
    plt.xlabel('k')
    plt.ylabel('Recall')
    plt.legend(loc='lower right')
    plt.savefig(file_loc)
    plt.close()

def print_table(tab_data, labels, aug, cold_start=False):
    table = tabulate(tab_data, headers=labels, tablefmt="fancy_grid")
    print(table)
    user_type = 'cold_start' if cold_start is True else 'all_users'
    aug_type = 'aug_' if aug is True else 'base_'
    if aug:
        filename = f'./results/{user_type}/{aug_type}errors.txt'
    with open(filename, 'w') as f:
        f.write(table)

#################################################################
#                        Data Wrangling                         #
#################################################################

def logical_xor(a, b):
    return (a>b)+(b>a)

def only_cold_start(masked_R_coo, unmasked_vals_coo):
    nnzs = masked_R_coo.getnnz(axis=1)
    warm_users = nnzs > 2
    print('num users total = ', len(nnzs))
    print('num cold start users = ', len(nnzs) - len(np.where(warm_users)[0]))
    diagonal = sparse.eye(unmasked_vals_coo.shape[0]).tocsr()
    for i in warm_users:
        diagonal[i, i] = 0
    unmasked_cold_vals = diagonal.dot(unmasked_vals_coo)
    return  sparse.coo_matrix(unmasked_cold_vals)

def setup(masked_R_coo, unmasked_vals_coo, unmasked_cold_coo):
    print('make train and test sets...', end='')
    start = time.time()
    masked_df = pd.DataFrame(data={'userID': masked_R_coo.row, 'itemID': masked_R_coo.col, 'rating': masked_R_coo.data})
    unmasked_df = pd.DataFrame(data={'userID': unmasked_vals_coo.row, 'itemID': unmasked_vals_coo.col, 'rating': unmasked_vals_coo.data})
    unmasked_cold_df = pd.DataFrame(data={'userID': unmasked_cold_coo.row, 'itemID': unmasked_cold_coo.col, 'rating': unmasked_cold_coo.data})
    start = time.time()
    trainset, testset, cold_testset = get_train_and_test_sets(masked_df, unmasked_df, unmasked_cold_df)
    end = time.time()
    print('done in {} seconds'.format(round(end-start)))
    return trainset, testset, cold_testset

def get_train_and_test_sets(masked_df, unmasked_df, unmasked_cold_df):
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(masked_df, reader)
    test_data = Dataset.load_from_df(unmasked_df, reader)
    cold_test_data = Dataset.load_from_df(unmasked_cold_df, reader)
    trainset = train_data.build_full_trainset()
    testset = train_data.construct_testset(test_data.raw_ratings)
    cold_testset = train_data.construct_testset(cold_test_data.raw_ratings)
    return trainset, testset, cold_testset

def get_data_from_dataloader():
    print('loading the data...', end='')
    start = time.time()
    training_dataset = UserDataset(
        data_name='food',
        mode='train'
    )
    validation_dataset = UserDataset(
        data_name='food',
        mode='val'
    )
    masked_R = training_dataset.get_interactions(style="numpy")
    unmasked_R = validation_dataset.get_interactions(style="numpy")
    end = time.time()
    print('downloaded in {} seconds'.format(round(end-start)))

    return masked_R, unmasked_R

def toy_example():
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
     [0.,1.,5.,0.],
     [0.,0.,4.,0.],
     [0.,0.,0.,5.],
     [3.,2.,0.,0.],
     [0.,2.,0.,0.],
    ])

    return sparse.coo_matrix(masked_R), sparse.coo_matrix(unmasked_R)