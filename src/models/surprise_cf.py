import time
import numpy as np
from scipy import sparse
import argparse

parser = argparse.ArgumentParser()
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
    
    def predict(self, testset, cold_testset):
        self.predictions = self.algo.test(testset)
        self.cold_predictions = self.algo.test(cold_testset)

    def evaluate_all_users(self):
        print('evaluating all users', self.name, '... ', end='')
        start = time.time()
        self.mae = accuracy.mae(self.predictions, verbose=False)
        self.rmse = accuracy.rmse(self.predictions, verbose=False)
        precisions_and_recalls = [precision_recall_at_k(self.predictions, k) for k in self.ks]
        self.MAPs, self.MARs = zip(*precisions_and_recalls)
        print('evaluating cold users', self.name, '... ', end='')

    def evaluate_cold_users(self):
        print('evaluating cold users', self.name, '... ', end='')
        start = time.time()
        self.cold_mae = accuracy.mae(self.cold_predictions, verbose=False)
        self.cold_rmse = accuracy.rmse(self.cold_predictions, verbose=False)
        precisions_and_recalls = [precision_recall_at_k(self.cold_predictions, k) for k in self.ks]
        self.cold_MAPs, self.cold_MARs = zip(*precisions_and_recalls)
        end = time.time()
        print('done in ', round(end-start), 'seconds')

    # def evaluate_all_users_refined(refined_predictions):
    #     print('evaluating refined users', self.name, '... ', end='')
    #     start = time.time()
    #     self.mae = accuracy.mae(refined_predictions, verbose=False)
    #     self.rmse = accuracy.rmse(refined_predictions, verbose=False)
    #     precisions_and_recalls = [precision_recall_at_k(refined_predictions, k) for k in self.ks]
    #     self.MAPs, self.MARs = zip(*precisions_and_recalls)
    #     end = time.time()
    #     print('done in ', round(end-start), 'seconds')
        
    # def evaluate_cold_users_refined(refined_predictions):
    #     print('evaluating refined users', self.name, '... ', end='')
    #     start = time.time()
    #     self.mae = accuracy.mae(refined_predictions, verbose=False)
    #     self.rmse = accuracy.rmse(refined_predictions, verbose=False)
    #     precisions_and_recalls = [precision_recall_at_k(self.predictions, k) for k in self.ks]
    #     self.MAPs, self.MARs = zip(*precisions_and_recalls)
    #     end = time.time()
    #     print('done in ', round(end-start), 'seconds')

def run_model(model, trainset, testset, cold_testset, aug, generated_users, generated_items):
    model.train(trainset)
    model.predict(testset, cold_testset)
    if model.name == 'SVD' and aug:
    # if model.name == 'SVD':
        print('U and I shape = ', model.algo.pu.shape, model.algo.qi.T.shape)
        full_prediction_matrix = np.dot(model.algo.pu, model.algo.qi.T)
        print('refining')
        refined_predictions = refine_ratings(trainset.ur, trainset.ir, full_prediction_matrix, generated_users,
                   generated_items, .5)
        print('done!')
        # trainset = train_data.build_full_trainset()
        # model.train(trainset)
        # model.predict(testset, cold_testset)
        # model.evaluate_all_users_refined(refined_predictions)
        # model.evaluate_cold_users_refined(refined_predictions)
    model.evaluate_all_users()
    model.evaluate_cold_users()
    return model

def run(masked_R_coo, unmasked_vals_coo, unmasked_cold_coo, mask_coo, mask_csr, ks, aug, generated_users, generated_items):
    print(masked_R_coo.shape, unmasked_vals_coo.shape)
    trainset, testset, cold_testset = setup(masked_R_coo, unmasked_vals_coo, unmasked_cold_coo)
    # for u, i, r in trainset.all_ratings():
    #     print(u, i, r)
    #     print(trainset.to_raw_uid(u), trainset.to_raw_iid(i))
    models = [
        # Model(name='random', algo=NormalPredictor(), ks=ks),
        # Model(name='bias only', algo=BaselineOnly(verbose=False, bsl_options = {'method': 'sgd','learning_rate': .00005,}), ks=ks),
        Model(name='SVD', algo=SVD(verbose=False), ks=ks),
        # Model(name='KNN', algo=KNNBasic(verbose=False), ks=ks),
        ]

    for i, model in enumerate(models):
        models[i] = run_model(model, trainset, testset, cold_testset, aug, generated_users, generated_items)
    
    show_and_save(models, aug)

if __name__ == "__main__":

    parser.add_argument("--augmented_users_file_path", default='/mnt/nfs/scratch1/neerajsharma/model_params/generated_1000_user_neighbors_without_reviews_more_sparse.npy',
                        type=str, required=False,
                        help="Generated user data file path")
    parser.add_argument("--augmented_items_file_path", default='/mnt/nfs/scratch1/neerajsharma/model_params/generated_1000_item_neighbors_without_reviews_more_sparse.npy',
                        type=str, required=False,
                        help="Generated items data file path")
    parser.add_argument("--use_augmentation", default='no',
                        type=str, required=False,
                        help="whether to use augmentation `yes` otherwise `no`")

    args, unknown = parser.parse_known_args()
    generated_users_file = args.augmented_users_file_path
    generated_items_file = args.augmented_items_file_path
    aug = args.use_augmentation

    print("augmentation: {}".format(aug))
    print("file path for augmented users {}".format(generated_users_file))
    print("file path for augmented items {}".format(generated_items_file))
    
    masked_R_coo, unmasked_R_coo, keep_item_idxs = get_data_from_dataloader()
    nnzs = masked_R_coo.getnnz(axis=1)
    warm_users = nnzs > 2
    if aug == 'yes':
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
        masked_R_coo = sparse.vstack([masked_R_coo, generated_users_coo])
        unmasked_R_coo = sparse.vstack([unmasked_R_coo, generated_users_coo])

        generated_items_vectors = np.array([v for v in generated_items.values()]).reshape(num_generated_items, item_neighbor_dim)
        filler = np.zeros((num_generated_items, num_generated_users))
        generated_items_vectors = np.concatenate((generated_items_vectors, filler), axis=1)
        generated_items_coo = sparse.coo_matrix(generated_items_vectors.T)
        
        masked_R_coo = sparse.hstack([masked_R_coo, generated_items_coo])
        unmasked_R_coo = sparse.hstack([unmasked_R_coo, generated_items_coo])
        aug = True
    else:
        aug = False

    mask_coo = sparse.coo_matrix(logical_xor(masked_R_coo, unmasked_R_coo))
    mask_csr = mask_coo.tocsr()

    unmasked_vals_csr = unmasked_R_coo.multiply(mask_coo)
    unmasked_vals_coo = sparse.coo_matrix(unmasked_vals_csr)
    unmasked_cold_coo = only_cold_start(masked_R_coo, unmasked_vals_coo, warm_users)
    
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]

    run(masked_R_coo, unmasked_vals_coo, unmasked_cold_coo, mask_coo, mask_csr, ks, aug, generated_users, generated_items)
