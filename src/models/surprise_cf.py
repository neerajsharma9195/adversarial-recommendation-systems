import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import sparse
from surprise import Dataset, accuracy, Reader, Trainset, SVD, NormalPredictor, KNNBasic, BaselineOnly
from src.preprocessing.dataloader import UserDataset
from src.models.sparse_mf import get_P_and_R, predict

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
    # ground_truth_csr = unmasked_vals_coo.tocsr()
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

class Model():
    def __init__(self, name, algo):
        self.name = name
        self.algo = algo

    def train(self, trainset):
        print('training ', self.name, '...', end='')
        start = time.time()
        self.algo.fit(trainset)
        end = time.time()
        print('done in ', round(end-start), ' seconds')

    def predict(self, testset):
        self.predictions = self.algo.test(testset)

    def evaluate(self):
        self.mae = accuracy.mae(self.predictions)
        self.rmse = accuracy.rmse(self.predictions)


def run(masked_R_coo, unmasked_vals_coo, mask_coo, mask_csr, ks):
    trainset, testset = setup(masked_R_coo, unmasked_vals_coo)
    models = [
        Model(name='random', algo=NormalPredictor()),
        Model(name='SGD', algo=BaselineOnly(bsl_options = {'method': 'sgd','learning_rate': .00005,})),
        Model(name='SVD', algo=SVD()),
        # Model(name='KNN', algo=KNNBasic())
        ]
    
    for model in models:
        model.train(trainset)
        model.predict(testset)
        model.evaluate()

    labels = [model.name for model in models]
    errors = [[model.mae, model.rmse] for model in models]
    MAPs, MARs = [], []
    
    show_and_save(labels, errors, MAPs, MARs, ks)

def show_and_save(models, errors, MAPs, MARs, ks):
    os.makedirs('results', exist_ok=True)
    plot_MAP(MAPs, models, ks)
    plot_MAR(MARs, models, ks)
    error_labels = ['MAE', 'RMSE']
    tab_data = [[models[i]] + errors[i] for i in range(len(models))]
    print_table(tab_data, error_labels)


############## Evaluation ###################

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

def logical_xor(a, b):
    return (a>b)+(a<b)


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

    masked_R_coo = sparse.coo_matrix(masked_R)
    unmasked_R_coo = sparse.coo_matrix(unmasked_R)

    mask_csr = logical_xor(unmasked_R_coo.astype('bool'), masked_R_coo.astype('bool'))
    mask_coo = sparse.coo_matrix(mask_csr)
    
    unmasked_vals_coo = sparse.coo_matrix(unmasked_R_coo.multiply(mask_coo))

    ks = [3, 5, 10, 20, 30, 40, 50, 75, 100]

    run(masked_R_coo, unmasked_vals_coo, mask_coo, mask_csr, ks)
