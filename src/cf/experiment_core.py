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