import os
import numpy as np
import pytables as tb
from preprocessing import DATASET_DIR, HDF5_DATASET, DATASET_NAME

REVIEW_TABLE = 'Review'
META_TABLE = 'Meta'

class Reviews_Idx:
    reviewerID = 0
    summary = 1
    reviewText = 2

def get_rating_vector(user_id):
    # todo: Get Rating Vector for user
    pass


def get_missing_vector(user_id):
    # todo: Get Missing Vector for user
    pass


def get_conditional_vector(user_id):
    # todo: Get Conditional Vector for user
    pass

def get_review(dir=DATASET_DIR, data_name=HDF5_DATASET) -> np.ndarray:
    h5f = tb.open_file(os.path.join(dir, data_name), 'r')
    table = h5f.root[data_name]['Review']
    for row in table.iterrows():
        yield row['reviewText']

def get_review_by_id(user_id: int, dir=DATASET_DIR, data_name=HDF5_DATASET) -> np.ndarray:
    with tb.open_file(os.path.join(dir, data_name), 'r') as h5f:
        table = h5f.root[data_name]['Review']
        return table[user_id][Reviews_Idx.reviewText]

def get_all_reviews_of_user(user_id):
    # todo: Get all reviews for user
    pass


def get_noise_vector():
    pass
