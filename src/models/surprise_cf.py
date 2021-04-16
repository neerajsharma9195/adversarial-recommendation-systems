import time
import numpy as np
from scipy import sparse
from multiprocessing import Pool
from surprise import accuracy, SVD, NormalPredictor, KNNBasic, BaselineOnly
from src.models.cf_utils import *

class Model():
    def __init__(self, name, algo, ks):
        self.name = name
        self.algo = algo
        self.ks = ks

    def train(self, trainset):
        print('training ', self.name, '... ', end='')
        start = time.time()
        self.algo.fit(trainset)
        end = time.time()
        print('done in ', round(end-start), 'seconds')
    
    def predict(self, testset):
        self.predictions = self.algo.test(testset)

    def evaluate(self):
        print('evaluating ', self.name, '... ', end='')
        start = time.time()
        self.mae = accuracy.mae(self.predictions, verbose=False)
        self.rmse = accuracy.rmse(self.predictions, verbose=False)
        precisions_and_recalls = [precision_recall_at_k(self.predictions, k) for k in self.ks]
        self.MAPs, self.MARs = zip(*precisions_and_recalls)
        end = time.time()
        print('done in ', round(end-start), 'seconds')

def run_model(model, trainset, testset):
    model.train(trainset)
    model.predict(testset)
    model.evaluate()
    return model

def run(masked_R_coo, unmasked_vals_coo, mask_coo, mask_csr, ks):
    trainset, testset = setup(masked_R_coo, unmasked_vals_coo)
    models = [
        Model(name='random', algo=NormalPredictor(), ks=ks),
        Model(name='SGD', algo=BaselineOnly(verbose=False, bsl_options = {'method': 'sgd','learning_rate': .00005,}), ks=ks),
        # Model(name='KNN', algo=KNNBasic(verbose=False), ks=ks),
        Model(name='SVD', algo=SVD(verbose=False), ks=ks),
        ]

    args = [(model, trainset, testset) for model in models]
    with Pool() as pool:
        models = pool.starmap(run_model, args)
    
    show_and_save(models)


if __name__ == "__main__":

    # masked_R, unmasked_R = toy_example()
    masked_R, unmasked_R = get_data_from_dataloader()

    masked_R_coo = sparse.coo_matrix(masked_R)
    unmasked_R_coo = sparse.coo_matrix(unmasked_R)

    mask_csr = logical_xor(unmasked_R_coo.astype('bool'), masked_R_coo.astype('bool'))
    mask_coo = sparse.coo_matrix(mask_csr)
    
    unmasked_vals_coo = sparse.coo_matrix(unmasked_R_coo.multiply(mask_coo))
    
    ks = [3, 5, 10, 20, 30, 40, 50, 75, 100]

    run(masked_R_coo, unmasked_vals_coo, mask_coo, mask_csr, ks)
