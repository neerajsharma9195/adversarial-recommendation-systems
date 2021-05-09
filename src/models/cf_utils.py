# cython: language_level=3
import os
import time
import math
import progressbar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse, stats
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

def get_full_prediction_matrix(algo, trainset):
    full_prediction_matrix = np.dot(algo.pu, algo.qi.T)
    predictions = np.zeros_like(full_prediction_matrix)
    num_users, num_items = full_prediction_matrix.shape
    for u in range(num_users):
        for i in range(num_items):
            predictions[u,i] = full_prediction_matrix[u,i]+algo.bu[u]+algo.bi[i] + trainset.global_mean
    return predictions

def refine_ratings(users_dataset, items_dataset, predicted_augmented_rating_matrix, neighbor_users,
                   neighbor_items, alpha):
    print('refining...', end='')
    start = time.time()
    num_users, num_items = predicted_augmented_rating_matrix.shape
    og_num_users, og_num_items = neighbor_items[list(neighbor_items.keys())[0]].shape[1], neighbor_users[list(neighbor_users.keys())[0]].shape[1]
    num_generated_users, num_generated_items = num_users - og_num_users, num_items - og_num_items
    
    for key, val in neighbor_users.items():  # key: index of user # val: list of neighbors
        num_neighbors = val.shape[0]
        real_ratings = users_dataset[key]
        real_rating_vector = np.zeros(num_items)
        for (k, v) in real_ratings:
            real_rating_vector[k] = v

        weights = np.zeros((num_neighbors, 1))
        expanded_val = np.zeros((num_neighbors, num_items))
        expanded_val[:, :og_num_items] = val
        for i, neighbor in enumerate(expanded_val):
            weights[i] = stats.pearsonr(real_rating_vector, neighbor)[0]

        refine_rating_vector = alpha * predicted_augmented_rating_matrix[key] + \
                                    (1 - alpha) * np.sum(weights * expanded_val, axis=0)

        """
        weights:                                 num_neighbors x 1
        val:                                     num_neighbors x og_num_items -> num_neighbors x num_items
        np.sum(weights * val, axis=0):           1 x og_num_items
        predicted_augmented_rating_matrix[key]:  1 x num_items
        """

        refine_rating_vector = np.maximum(np.zeros_like(refine_rating_vector), refine_rating_vector)
        # for i in range(len(refine_rating_vector)):
        #     if refine_rating_vector[i] < 0:
        #         refine_rating_vector[i] = 0.0
            # else:
                # refine_rating_vector[i] = math.ceil(refine_rating_vector[i])
        # if (predicted_augmented_rating_matrix[key] != refine_rating_vector).all():
        #     print('refining changed output by at least', np.max(np.abs(predicted_augmented_rating_matrix[key] - refine_rating_vector)))
        predicted_augmented_rating_matrix[key] = refine_rating_vector

    for key, val in neighbor_items.items():  # key: index of user # val: list of neighbors
        num_neighbors = val.shape[0]
        real_ratings = items_dataset[key]  
        real_rating_vector = np.zeros(num_users)
        for (k, v) in real_ratings:
            real_rating_vector[k] = v

        weights = np.zeros((num_neighbors, 1))
        expanded_val = np.zeros((num_neighbors, num_users))
        expanded_val[:, :og_num_users] = val

        for i, neighbor in enumerate(expanded_val):  # calculating weights per neighbor
            weights[i] = stats.pearsonr(real_rating_vector, neighbor)[0]

        n, m = predicted_augmented_rating_matrix.shape
        predicted_item_vector = predicted_augmented_rating_matrix[:,key].T

        refine_rating_vector = alpha * predicted_item_vector + (1 - alpha) * np.sum(weights * expanded_val, axis=0)
        refine_rating_vector = np.maximum(np.zeros_like(refine_rating_vector), refine_rating_vector)
        # for i in range(len(refine_rating_vector)):
        #     if refine_rating_vector[i] < 0:
        #         refine_rating_vector[i] = 0.0
            # else:
                # refine_rating_vector[i] = math.ceil(refine_rating_vector[i])
        for i in range(n):
            predicted_augmented_rating_matrix[i][key] = refine_rating_vector[i]
    end = time.time()
    print(f'done in {round(end-start)} seconds')
    return predicted_augmented_rating_matrix

#################################################################
#                   Printing and Plotting                       #
#################################################################

def show_and_save(models, aug):
    ks = models[0].ks
    labels = [model.name for model in models]
    errors = [[model.mae, model.rmse] for model in models]
    MAPs = [model.MAPs for model in models]
    MARs = [model.MARs for model in models]

    # cold_errors = [[model.cold_mae, model.cold_rmse] for model in models]
    # cold_MAPs = [model.cold_MAPs for model in models]
    # cold_MARs = [model.cold_MARs for model in models]

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/cold_start', exist_ok=True)
    os.makedirs('results/all_users', exist_ok=True)

    plot_MAP(MAPs, labels, ks, aug)
    plot_MAR(MARs, labels, ks, aug)
    error_labels = ['all_users'] + ['MAE', 'RMSE']
    tab_data = [[labels[i]] + errors[i] for i in range(len(labels))]
    print_table(tab_data, error_labels, aug)

    # plot_MAP(cold_MAPs, labels, ks, aug, cold_start=True)
    # plot_MAR(cold_MARs, labels, ks, aug, cold_start=True)
    # error_labels = ['cold_users'] + ['MAE', 'RMSE']
    # tab_data = [[labels[i]] + cold_errors[i] for i in range(len(labels))]
    # print_table(tab_data, error_labels, aug, cold_start=True)

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
    table = tabulate(tab_data, headers=labels, tablefmt="github")
    print(table)
    user_type = 'cold_start' if cold_start is True else 'all_users'
    aug_type = 'aug_' if aug is True else 'base_'
    filename = f'./results/{user_type}/{aug_type}errors.txt'
    with open(filename, 'w') as f:
        f.write(table)

#################################################################
#                        Data Wrangling                         #
#################################################################

def make_aug_data(masked_R_coo, unmasked_R_coo, keep_item_idxs, mask_coo, warm_users, generated_users_file, generated_items_file):
    generated_users = np.load(generated_users_file, allow_pickle=True).item()
    generated_items = np.load(generated_items_file, allow_pickle=True).item()
    for key, value in generated_users.items():
        generated_users[key] = value[:,keep_item_idxs]
    num_user_ids = len(generated_users.keys())
    num_item_ids = len(generated_items.keys())
    user_neighbor_per_id, user_neighbor_dim = generated_users[list(generated_users.keys())[0]].shape
    item_neighbor_per_id, item_neighbor_dim = generated_items[list(generated_items.keys())[0]].shape
    num_generated_users = num_user_ids * user_neighbor_per_id
    num_generated_items = num_item_ids * item_neighbor_per_id

    generated_users_vectors = np.array([v for v in generated_users.values()]).reshape(num_generated_users, user_neighbor_dim)
    generated_users_coo = sparse.coo_matrix(generated_users_vectors)
    false_coo = sparse.coo_matrix(np.zeros_like(generated_users_vectors, dtype=bool))
    aug_masked_R_coo = sparse.vstack([masked_R_coo, generated_users_coo])
    aug_unmasked_R_coo = sparse.vstack([unmasked_R_coo, generated_users_coo])
    aug_mask_coo = sparse.vstack([mask_coo, false_coo])

    generated_items_vectors = np.array([v for v in generated_items.values()]).reshape(num_generated_items, item_neighbor_dim)
    filler = np.zeros((num_generated_items, num_generated_users))
    generated_items_vectors = np.concatenate((generated_items_vectors, filler), axis=1)
    false_coo = sparse.coo_matrix(np.zeros_like(generated_items_vectors.T, dtype=bool))
    generated_items_coo = sparse.coo_matrix(generated_items_vectors.T)

    aug_masked_R_coo = sparse.hstack([aug_masked_R_coo, generated_items_coo])
    aug_unmasked_R_coo = sparse.hstack([aug_unmasked_R_coo, generated_items_coo])
    aug_mask_coo = sparse.hstack([aug_mask_coo, false_coo])

    return aug_masked_R_coo, aug_unmasked_R_coo, aug_mask_coo, generated_users, generated_items

def logical_xor(a, b):
    return (a>b)+(b>a)

def only_cold_start(masked_R_coo, unmasked_vals_coo, warm_users):
    print('num users total = ', masked_R_coo.shape[0])
    print('num cold start users = ', masked_R_coo.shape[0] - len(np.where(warm_users)[0]))
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
    masked_R = masked_R.tocsr()
    unmasked_R = unmasked_R.tocsr()
    keep_item_idxs = masked_R.getnnz(0)>0
    masked_R = masked_R[:,keep_item_idxs]
    unmasked_R = unmasked_R[:,keep_item_idxs]
    end = time.time()
    print('downloaded in {} seconds'.format(round(end-start)))

    return masked_R.tocoo(), unmasked_R.tocoo(), keep_item_idxs

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