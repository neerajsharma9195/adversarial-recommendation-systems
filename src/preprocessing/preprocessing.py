import os
import numpy as np
import tables as tb
import pandas as pd

from src.preprocessing.tiny_review_embeddings import get_embedding
from src.preprocessing.utils import (
    DATASET_DIR, META_PREFIX, REVIEW_PREFIX, HDF5_DATASET, DATASET_NAME,
    getAmazonData, Reviews
)

def review_data_triming(df: pd.DataFrame, max_length: int, min_item_reviews: int, min_user_reviews: int) -> pd.DataFrame:
    # Trim HTML tags
    df['reviewText'].replace(
        to_replace=r"""<(\w|\d|\n|[().=,\-:;@#$%^&*\[\]"'+–/\/®°⁰!?{}|`~]| )+?>""",
        value="",
        regex=True,
        inplace=True
    )
    df['reviewText'].replace(
        to_replace=r"""&nbsp;""",
        value=". ",
        regex=True,
        inplace=True
    )

    df = df[df['reviewText'].str.len() <= max_length]
    df = df[df['reviewText'].notnull() | (df['reviewText'] == "  ") | (df['reviewText'] == " ") ]
    df = df[df['summary'].notnull()]
    df = df[df['asin'].map(df['asin'].value_counts()) >= min_item_reviews]
    df = df[df['reviewerID'].map(df['reviewerID'].value_counts()) >= min_user_reviews]

    return df


def save_review_dataset(data_name: str, dataset=None, agg_func=get_embedding,
                        dir=DATASET_DIR, hdf5_name=HDF5_DATASET, categories=['reviewText'],
                        min_item_reviews=100, min_user_reviews=3, max_length=256) -> None:
    if dataset is not None:
        assert 'asin' in dataset.columns
        assert 'reviewerID' in dataset.columns
        assert 'summary' in dataset.columns
        df = dataset
    elif isinstance(data_name, str):
        df = getAmazonData(data_name)
    else:
        raise TypeError("'dataset' is neither a DataFrame nor a str!")

    df = review_data_triming(df, max_length, min_item_reviews, min_user_reviews)

    grouped_df = df.groupby('reviewerID')

    # file_mode = 'a' if os.path.isfile(os.path.join(dir, hdf5_name)) else 'w'
    file_mode = 'a'
    with tb.open_file(os.path.join(dir, hdf5_name), file_mode) as h5f:
        # Get the HDF5 root group
        root = h5f.root

        # Create the group:
        if data_name not in str(h5f.list_nodes('/')):
            group = h5f.create_group(root, data_name)

        # Now, create and fill the tables in Particles group
        cur_group = root[data_name]

        # Create table
        tablename = "Review"
        table = h5f.create_table(
            f"/{data_name}", tablename, Reviews, "{data_name}: "+tablename
        )

        # Get the record object associated with the table:
        review = table.row

        for i, d in enumerate(grouped_df):
            try:
                idx, val = d
                review['reviewerID'] = idx
                for col in categories:
                    review[col]    = agg_func(list(filter(lambda x: len(x) > 0, val[col]))).cpu().detach().numpy()
            
                # This injects the Record values
                review.append()
            except:
                print(f"Error Found when processing user #{i} {idx}")
                for col in categories:
                    print(f"{col}:    {val[col]}")
                continue
    
        # Flush the table buffers
        table.flush()

def save_user_item_interaction(data_name: str, dataset=None,
                               dir=DATASET_DIR, hdf5_name=HDF5_DATASET,
                               min_item_reviews=100, min_user_reviews=3, max_length=256) -> None:
    if dataset is not None:
        assert 'asin' in dataset.columns
        assert 'reviewerID' in dataset.columns
        assert 'overall' in dataset.columns
        df = dataset
    elif isinstance(data_name, str):
        df = getAmazonData(data_name)
    else:
        raise TypeError("'dataset' is neither a DataFrame nor a str!")

    df = review_data_triming(df, max_length, min_item_reviews, min_user_reviews)

    df = df[['reviewerID', 'asin', 'overall']].groupby(['reviewerID', 'asin'])['overall']\
                                              .mean().unstack().reset_index() \
                                              .fillna(0).set_index('reviewerID')

    file_mode = 'a' if os.path.isfile(os.path.join(dir, hdf5_name)) else 'w'
    with tb.open_file(os.path.join(dir, hdf5_name), file_mode) as h5f:
        # Get the HDF5 root group
        root = h5f.root

        # Create the group:
        if data_name not in str(h5f.list_nodes('/')):
            group = h5f.create_group(root, data_name)

        # Now, create and fill the tables in Particles group
        cur_group = root[data_name]

        # Create table
        filters = tb.Filters(complib='zlib', complevel=5)
        h5f.create_carray(cur_group, 'Interactions', obj=df.to_numpy(), filters=filters)

def get_masked_interactions(interactions: np.ndarray, min_user_rating, mask_ratio) -> (np.ndarray, np.ndarray, np.ndarray):
    num_valid_items = np.count_nonzero(interactions, axis=1)
    uid_mask = num_valid_items > min_user_rating
    uids = np.arange(interactions.shape[0])[uid_mask]
    masks = []

    for i in uids:
        mask = np.ones(num_valid_items[i], dtype=bool)
        valid_items = np.arange(num_valid_items[i])  

        # Randomly mask 50% of the ratings above the `min_user_rating` threshold
        masked_items = np.random.choice(
            valid_items,
            size=np.maximum(1, int((num_valid_items[i] - min_user_rating) * mask_ratio)),
            replace=False
        )
        mask[masked_items] = False    

        # This mask will be used for review embedding 
        masks.append(valid_items[mask])

        # Flip the masked user-item ratings to 0
        row = interactions[i, :]
        interactions[i, row.nonzero()[0][~mask]] = 0
    
    return interactions, uids, masks

def save_masked_review_dataset(data_name: str, dataset=None, agg_func=get_embedding,
                               dir=DATASET_DIR, hdf5_name=HDF5_DATASET, categories=['reviewText'],
                               min_item_reviews=100, min_user_reviews=3, max_length=256,
                               min_user_rating=4, mask_ratio=0.75) -> None:
    if dataset is not None:
        assert 'asin' in dataset.columns
        assert 'reviewerID' in dataset.columns
        assert 'summary' in dataset.columns
        df = dataset
    elif isinstance(data_name, str):
        print("Data Importing now...")
        df = getAmazonData(data_name)
    else:
        raise TypeError("'dataset' is neither a DataFrame nor a str!")

    print("Data Triming now...")
    df = review_data_triming(df, max_length, min_item_reviews, min_user_reviews)

    # Generate user-item interactions
    print("Generating interactions and masks...")
    interactions = df[['reviewerID', 'asin', 'overall']].groupby(['reviewerID', 'asin'])['overall']\
                                                        .mean().unstack().reset_index() \
                                                        .fillna(0).set_index('reviewerID')
    interactions = interactions.to_numpy()
    interactions, uids, masks = get_masked_interactions(interactions, min_user_rating, mask_ratio)
    
    # Generate review embeddings
    grouped_df = df.groupby('reviewerID')
    
    file_mode = 'a' if os.path.isfile(os.path.join(dir, hdf5_name)) else 'w'
    with tb.open_file(os.path.join(dir, hdf5_name), file_mode) as h5f:
        # Get the HDF5 root group
        root = h5f.root

        # Create the group:
        if data_name not in str(h5f.list_nodes('/')):
            group = h5f.create_group(root, data_name)

        for name in ["masked_Review", "masked_Interactions", "uid_mask"]:
            if name in str(h5f.list_nodes(f'/{data_name}')):
                print(f"Deleting group /{data_name}/{name}")
                h5f.remove_node(f'/{data_name}/{name}')

        # Copy the unmodified embeddings
        cur_group = root[data_name]
        tablename = "masked_Review"

        h5f.copy_node(where=f'/{data_name}/Review', newparent=f'/{data_name}', newname=tablename)
        cur_table = cur_group[tablename]

        print("Saving Dataset now...")
        for i, d in enumerate(grouped_df):
            if i % 1000 == 0:
                print(f"current uid: {i}") 
            
            if i in uids:
                idx, val = d
                for col in categories:
                    r = list(filter(lambda x: len(x) > 0, val[col]))
                    try:
                        cur_table[i] = (
                            agg_func([r[j] for j in masks[i]]).cpu().detach().numpy(),
                            cur_table[i]['reviewerID']
                        )
                    except IndexError:
                        uids = np.delete(uids, np.where(uids == i))

        # Store masked userIDs and interactions
        filters = tb.Filters(complib='zlib', complevel=5)
        h5f.create_carray(cur_group, 'masked_Interactions', obj=interactions, filters=filters)
        h5f.create_carray(cur_group, 'uid_mask', obj=uids, filters=filters)

    
def save_meta_dataset(data_name: str, dataset=None, agg_func=lambda x: x, save_path="./") -> None:
    if dataset is not None:
        assert 'asin' in dataset.columns
        df = dataset
    elif isinstance(dataset, str):
        df = getAmazonData(data_name)
    else:
        raise TypeError("'dataset' is neither a DataFrame nor a str!")

    raise NotImplementedError


if __name__ == '__main__':
    df = getAmazonData('food', 'all')
    # save_review_dataset(data_name='food', dataset=df, agg_func=get_embedding)
    # save_user_item_interaction(data_name='food', dataset=df)
    save_masked_review_dataset(
        data_name='food', dataset=df, agg_func=get_embedding, 
        min_user_rating=3, mask_ratio=0.8
    )
