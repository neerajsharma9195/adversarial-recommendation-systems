import os
import gzip
import json
import torch
import tables as tb
import pandas as pd

DATASET_DIR = "/mnt/nfs/scratch1/neerajsharma/amazon_data/"
META_PREFIX = "meta_"
REVIEW_PREFIX = "review_"
HDF5_DATASET = "new_5_dataset.h5"
DATASET_NAME = {
    'phone': 'Cell_Phones_and_Accessories.json.gz',
    'beauty': 'All_Beauty.json.gz',                     # haven't downloaded yet
    'food': 'Grocery_and_Gourmet_Food.json.gz'
}

# Review Dataset Format
class Reviews(tb.IsDescription):
    reviewerID = tb.StringCol(itemsize=20)    
    reviewText = tb.Float64Col(shape=(1, 128))

    # `summary` is ignored for now
    # summary    = tb.Float64Col(shape=(1, 128))


# Meta-data Dataset Format (is not used yet)
class Metadata(tb.IsDescription):
    # Temporary config
    asin      = tb.StringCol(itemsize=20)
    title     = tb.Float64Col(shape=(1, 128))
    category  = tb.Float64Col(shape=(1, 128))
    brand     = tb.Float64Col(shape=(1, 128))
    also_buy  = tb.Float64Col(shape=(1, 128))
    also_view = tb.Float64Col(shape=(1, 128))
    price     = tb.Float64Col(shape=(1, 128))


def parse(path: str):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path: str, num_entry='all') -> pd.DataFrame:
    df = {}
    if num_entry == 'all':
        for i, d in enumerate(parse(path)):
            df[i] = d
    elif isinstance(num_entry, int):
        for i, d in enumerate(parse(path)):
            if i > num_entry - 1:
                break
            df[i] = d
    else:
        raise TypeError("'num_entry' can either be an int or 'all'")

    return pd.DataFrame.from_dict(df, orient='index')

def getAmazonData(data_name: str, num_entry='all') -> pd.DataFrame:
    try:
        path = os.path.join(DATASET_DIR, DATASET_NAME[data_name])
        return getDF(path, num_entry)
    except KeyError:
        print(f"Dataset '{data_name}' is not supported!")

def get_item_mask(og_interactions: torch.Tensor, masked_interactions: torch.Tensor) -> torch.Tensor:
    return torch.logical_xor(og_interactions, masked_interactions)
