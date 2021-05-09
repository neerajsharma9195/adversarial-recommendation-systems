import surprise
import numpy as np
import pandas as pd
from scipy import sparse

from src.cf.utils.timer import Timer
from src.cf.utils.typings import Trainset, Testset
from src.cf.utils.general import invert_dictionary
from src.preprocessing.dataloader import UserDataset

from src.cf.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL
)

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

def logical_xor(a: sparse.coo_matrix, b: sparse.coo_matrix) -> sparse.coo_matrix:
    return (a>b)+(b>a)

def surprise_trainset_to_df(trainset, 
                            col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL) -> pd.DataFrame:
    """Converts a `surprise.Trainset` object to `pd.DataFrame`
    More info: https://surprise.readthedocs.io/en/stable/trainset.html
    Args:
        trainset (obj): A surprise.Trainset object.
        col_user (str): User column name.
        col_item (str): Item column name.
        col_rating (str): Rating column name.
    
    Returns:
        pd.DataFrame: A dataframe. The user and item columns are strings and the rating columns are floats.
    """
    df = pd.DataFrame(trainset.all_ratings(), columns=[col_user, col_item, col_rating])
    map_user = (
        trainset._inner2raw_id_users
        if trainset._inner2raw_id_users is not None
        else invert_dictionary(trainset._raw2inner_id_users)
    )
    map_item = (
        trainset._inner2raw_id_items
        if trainset._inner2raw_id_items is not None
        else invert_dictionary(trainset._raw2inner_id_items)
    )
    df[col_user] = df[col_user].map(map_user)
    df[col_item] = df[col_item].map(map_item)
    return df

def surprise_testset_to_df(testset: Testset,
                                  col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL) -> pd.DataFrame:
    testset_df = pd.DataFrame(testset)
    testset_df = testset_df.rename(
        index=str, columns={0: col_user, 1: col_item, 2: col_rating}
    )
    return testset_df

def get_data_from_dataloader(data_name='food') -> (sparse.coo_matrix, sparse.coo_matrix):
    print('loading the data...', end=' ')

    with Timer() as loading_time:
        training_dataset = UserDataset(
            data_name=data_name,
            mode='train'
        )
        validation_dataset = UserDataset(
            data_name=data_name,
            mode='val'
        )
        masked_R = training_dataset.get_interactions(style="numpy")
        unmasked_R = validation_dataset.get_interactions(style="numpy")
        masked_R = masked_R.tocsr()
        unmasked_R = unmasked_R.tocsr()
        keep_item_idxs = masked_R.getnnz(0)>0
        masked_R = masked_R[:,keep_item_idxs]
        unmasked_R = unmasked_R[:,keep_item_idxs]
    print("took {} seconds for loading the dataset.".format(loading_time.interval))

    return masked_R.tocoo(), unmasked_R.tocoo(), keep_item_idxs

def get_train_and_test_sets(masked_df: pd.DataFrame, unmasked_df: pd.DataFrame, unmasked_cold_df: pd.DataFrame) -> (Trainset, Testset, Testset):
    reader = surprise.Reader(rating_scale=(1, 5))
    train_data = surprise.Dataset.load_from_df(masked_df, reader)
    test_data = surprise.Dataset.load_from_df(unmasked_df, reader)
    cold_test_data = surprise.Dataset.load_from_df(unmasked_cold_df, reader)

    trainset = train_data.build_full_trainset()
    testset = train_data.construct_testset(test_data.raw_ratings)
    cold_testset = train_data.construct_testset(cold_test_data.raw_ratings)
    return trainset, testset, cold_testset

def only_cold_start(masked_R_coo: sparse.coo_matrix, unmasked_vals_coo: sparse.coo_matrix, warm_users: np.ndarray) -> sparse.coo_matrix:
    print('num users total = ', masked_R_coo.shape[0])
    print('num cold start users = ', masked_R_coo.shape[0] - len(np.where(warm_users)[0]))
    diagonal = sparse.eye(unmasked_vals_coo.shape[0]).tocsr()
    for i in warm_users:
        diagonal[i, i] = 0
    unmasked_cold_vals = diagonal.dot(unmasked_vals_coo)
    return  sparse.coo_matrix(unmasked_cold_vals)

def setup(masked_R_coo: sparse.coo_matrix, unmasked_vals_coo: sparse.coo_matrix, unmasked_cold_coo: sparse.coo_matrix):
    print('make train and test sets...', end=' ')

    with Timer() as loading_time:
        masked_df = pd.DataFrame(data={'userID': masked_R_coo.row, 'itemID': masked_R_coo.col, 'rating': masked_R_coo.data})
        unmasked_df = pd.DataFrame(data={'userID': unmasked_vals_coo.row, 'itemID': unmasked_vals_coo.col, 'rating': unmasked_vals_coo.data})
        unmasked_cold_df = pd.DataFrame(data={'userID': unmasked_cold_coo.row, 'itemID': unmasked_cold_coo.col, 'rating': unmasked_cold_coo.data})

        trainset, testset, cold_testset = get_train_and_test_sets(masked_df, unmasked_df, unmasked_cold_df)

    print("`Setup` took {} seconds.".format(loading_time.interval))
    return trainset, testset, cold_testset