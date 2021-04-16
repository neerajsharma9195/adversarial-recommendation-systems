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

def show_and_save(models):
    ks = models[0].ks
    labels = [model.name for model in models]
    errors = [[model.mae, model.rmse] for model in models]
    MAPs = [model.MAPs for model in models]
    MARs = [model.MARs for model in models]

    os.makedirs('results', exist_ok=True)
    plot_MAP(MAPs, labels, ks)
    plot_MAR(MARs, labels, ks)
    error_labels = ['MAE', 'RMSE']
    tab_data = [[labels[i]] + errors[i] for i in range(len(labels))]
    print_table(tab_data, error_labels)

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
    filename = './results/errors.txt'
    with open(filename, 'w') as f:
        f.write(table)

#################################################################
#                        Data Wrangling                         #
#################################################################

def logical_xor(a, b):
    return (a>b)+(a<b)

def setup(masked_R_coo, unmasked_vals_coo):
    print('make df')
    start = time.time()
    masked_df = pd.DataFrame(data={'userID': masked_R_coo.row, 'itemID': masked_R_coo.col, 'rating': masked_R_coo.data})
    unmasked_df = pd.DataFrame(data={'userID': unmasked_vals_coo.row, 'itemID': unmasked_vals_coo.col, 'rating': unmasked_vals_coo.data})
    end = time.time()
    print('done in ', round(end-start), ' seconds')
    print('make train and test sets')
    start = time.time()
    trainset, testset = get_train_and_test_sets(masked_df, unmasked_df)
    end = time.time()
    print('done in ', round(end-start), ' seconds')
    return trainset, testset

def get_train_and_test_sets(masked_df, unmasked_df):
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(masked_df, reader)
    test_data = Dataset.load_from_df(unmasked_df, reader)
    trainset = train_data.build_full_trainset()
    testset = train_data.construct_testset(test_data.raw_ratings)
    return trainset, testset

def get_data_from_dataloader():
    print('loading the data...')
    start = time.time()
    user_dataset = UserDataset(data_name='food', path='/mnt/nfs/scratch1/neerajsharma/amazon_data/new_5_dataset.h5')
    validation_uid, validation_iid, validation_vid = user_dataset.get_mask(drop_ratio=0.3)
    training_uid, training_iid, training_vid = user_dataset.get_mask(
        drop_ratio=0.6, masked_uid=validation_uid, masked_iid=validation_iid
    )
    train_dataset = UserDataset(
        data_name='food',
        path='/mnt/nfs/scratch1/neerajsharma/amazon_data/new_5_dataset.h5',
        masked_uid=training_uid,
        masked_iid=training_iid,
        masked_vid=training_vid
    )
    validation_dataset = UserDataset(
        data_name='food',
        path='/mnt/nfs/scratch1/neerajsharma/amazon_data/new_5_dataset.h5',
        masked_uid=validation_uid,
        masked_iid=validation_iid,
        masked_vid=validation_vid
    )
    masked_R = train_dataset.get_interactions(style="numpy")
    unmasked_R = validation_dataset.get_interactions(style="numpy")
    end = time.time()
    print('done')
    print('downloaded in ', round(end-start), ' seconds')

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
     [0.,1.,0.,0.],
     [0.,0.,4.,0.],
     [0.,0.,0.,5.],
     [3.,0.,0.,0.],
     [0.,2.,0.,0.],
    ])

    return masked_R, unmasked_R