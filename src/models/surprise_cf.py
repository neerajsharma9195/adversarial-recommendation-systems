import time
import numpy as np
from scipy import stats
import argparse
import math
import time

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
        print('done in ', round(end - start), 'seconds')

    def predict(self, testset, cold_testset):
        self.predictions = self.algo.test(testset)
        self.cold_predictions = self.algo.test(testset)

    def evaluate_all_users(self):
        print('evaluating all users', self.name, '... ', end='')
        start = time.time()
        self.mae = accuracy.mae(self.predictions, verbose=False)
        self.rmse = accuracy.rmse(self.predictions, verbose=False)
        precisions_and_recalls = [precision_recall_at_k(self.predictions, k) for k in self.ks]
        self.MAPs, self.MARs = zip(*precisions_and_recalls)
        end = time.time()
        print('done in ', round(end - start), 'seconds')

    def evaluate_cold_users(self):
        print('evaluating cold users', self.name, '... ', end='')
        start = time.time()
        self.cold_mae = accuracy.mae(self.cold_predictions, verbose=False)
        self.cold_rmse = accuracy.rmse(self.cold_predictions, verbose=False)
        precisions_and_recalls = [precision_recall_at_k(self.cold_predictions, k) for k in self.ks]
        self.cold_MAPs, self.cold_MARs = zip(*precisions_and_recalls)
        end = time.time()
        print('done in ', round(end - start), 'seconds')


def run_model(model, trainset, testset, cold_testset):
    model.train(trainset)
    model.predict(testset, cold_testset)
    model.evaluate_all_users()
    model.evaluate_cold_users()
    return model


def run(masked_R_coo, unmasked_vals_coo, unmasked_cold_coo, mask_coo, mask_csr, ks, aug):
    trainset, testset, cold_testset = setup(masked_R_coo, unmasked_vals_coo, unmasked_cold_coo)
    models = [
        Model(name='random', algo=NormalPredictor(), ks=ks),
        Model(name='bias only',
              algo=BaselineOnly(verbose=False, bsl_options={'method': 'sgd', 'learning_rate': .00005, }), ks=ks),
        Model(name='SVD', algo=SVD(verbose=False), ks=ks),
        # Model(name='KNN', algo=KNNBasic(verbose=False), ks=ks),
    ]

    args = [(model, trainset, testset, cold_testset) for model in models]
    with Pool() as pool:
        models = pool.starmap(run_model, args)

    show_and_save(models, aug)
    model = models[2]
    start_time = time.time()
    product_mat = np.dot(model.algo.pu, model.algo.qi.T)
    end_time = time.time()
    print("time taken in multiplication {}".format(end_time-start_time))
    print("shape of mat {}".format(product_mat.shape))


def refine_ratings(users_dataset, items_dataset, predicted_augmented_rating_matrix, neighbor_users_path,
                   neighbor_items_path, alpha):
    neighbor_users = np.load(neighbor_users_path, allow_pickle=True)
    neighbor_items = np.load(neighbor_items_path, allow_pickle=True)

    for key, val in neighbor_users.item().items():  # key: index of user # val: list of neighbors
        real_rating_vector = users_dataset[key][1].numpy()  # real rating vector
        weights = []
        num_neighbors = val.shape[0]
        for neighbor in val:  # calculating weights per neighbor
            weights.append(stats.pearsonr(real_rating_vector, neighbor)[0])

        refine_rating_vector = alpha * predicted_augmented_rating_matrix[key] + (1 - alpha) * np.sum(
            np.array(weights).reshape(num_neighbors, 1) * val, axis=0)
        for i in range(len(refine_rating_vector)):
            if refine_rating_vector[i] < 0:
                refine_rating_vector[i] = 0.0
            else:
                refine_rating_vector[i] = math.ceil(refine_rating_vector[i])

        predicted_augmented_rating_matrix[key] = refine_rating_vector

    for key, val in neighbor_items.item().items():  # key: index of user # val: list of neighbors
        real_rating_vector = items_dataset[key][1].numpy()  # real rating vector
        weights = []
        num_neighbors = val.shape[0]
        for neighbor in val:  # calculating weights per neighbor
            weights.append(stats.pearsonr(real_rating_vector, neighbor)[0])

        predicted_item_vector = []
        n, m = predicted_augmented_rating_matrix.shape
        for i in range(n):
            predicted_item_vector.append(predicted_augmented_rating_matrix[i][key])

        refine_rating_vector = alpha * predicted_item_vector + (1 - alpha) * np.sum(
            np.array(weights).reshape(num_neighbors, 1) * val, axis=0)
        for i in range(len(refine_rating_vector)):
            if refine_rating_vector[i] < 0:
                refine_rating_vector[i] = 0.0
            else:
                refine_rating_vector[i] = math.ceil(refine_rating_vector[i])
        for i in range(n):
            predicted_augmented_rating_matrix[i][key] = refine_rating_vector[i]
    return predicted_augmented_rating_matrix


if __name__ == "__main__":

    parser.add_argument("--augmented_file_path",
                        default="'/mnt/nfs/scratch1/rbialik/adversarial-recommendation-systems/model_params/generated_100_user_neighbors.npy'",
                        type=str, required=False,
                        help="Generated data file path")
    parser.add_argument("--use_augmentation", default='no',
                        type=str, required=False,
                        help="whether to use augmentation `yes` otherwise `no`")

    args, unknown = parser.parse_known_args()
    generated_users_file = args.augmented_file_path
    aug = args.use_augmentation

    print("augmentation use or not {}".format(aug))
    print("file path for augmented data {}".format(generated_users_file))
    # masked_R_coo, unmasked_R_coo = toy_example()
    masked_R_coo, unmasked_R_coo = get_data_from_dataloader()
    if aug == 'yes':
        generated_users = np.load(generated_users_file, allow_pickle=True).item()
        num_ids = len(generated_users.keys())
        neighbor_per_id, neighbor_dim = generated_users[list(generated_users.keys())[0]].shape
        generated_users_coo = sparse.coo_matrix(
            np.array([v for v in generated_users.values()]).reshape(num_ids * neighbor_per_id, neighbor_dim))
        masked_R_coo = sparse.vstack([masked_R_coo, generated_users_coo])
        unmasked_R_coo = sparse.vstack([unmasked_R_coo, generated_users_coo])
        aug = True
    else:
        aug = False

    mask_coo = sparse.coo_matrix(logical_xor(masked_R_coo, unmasked_R_coo))
    mask_csr = mask_coo.tocsr()

    unmasked_vals_csr = unmasked_R_coo.multiply(mask_coo)
    unmasked_vals_coo = sparse.coo_matrix(unmasked_vals_csr)
    unmasked_cold_coo = only_cold_start(masked_R_coo, unmasked_vals_coo)

    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]

    run(masked_R_coo, unmasked_vals_coo, unmasked_cold_coo, mask_coo, mask_csr, ks, aug)
