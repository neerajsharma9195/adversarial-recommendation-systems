import os
import nltk
import gzip
import json
import numpy as np
import tables as tb
import pandas as pd

from collections import Counter
from review_embeddings import get_embedding
from collections.abc import Iterable


nltk.download('punkt')

DATASET_DIR = "/mnt/nfs/scratch1/neerajsharma/amazon_data/"
META_PREFIX = "meta_"
REVIEW_PREFIX = "review_"
HDF5_DATASET = "dataset.h5"
DATASET_NAME = {
    'phone': 'Cell_Phones_and_Accessories.json.gz',
    'beauty': 'All_Beauty.json.gz',                     # haven't downloaded yet
    'food': 'Grocery_and_Gourmet_Food.json.gz'
}

# Review Dataset Format
class Reviews(tb.IsDescription):
    reviewerID = tb.StringCol(itemsize=20)
    summary    = tb.Float64Col(shape=(1, 768))
    reviewText = tb.Float64Col(shape=(1, 768))
    


# Meta-data Dataset Format (is not used yet)
class Metadata(tb.IsDescription):
    # Temporary config
    asin      = tb.StringCol(itemsize=20)
    title     = tb.Float64Col(shape=(1, 768))
    category  = tb.Float64Col(shape=(1, 768))
    brand     = tb.Float64Col(shape=(1, 768))
    also_buy  = tb.Float64Col(shape=(1, 768))
    also_view = tb.Float64Col(shape=(1, 768))
    price     = tb.Float64Col(shape=(1, 768))


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

def save_review_dataset(data_name: str, dataset=None, agg_func=get_embedding, 
                        dir=DATASET_DIR, hdf5_name=HDF5_DATASET, min_item_review=5) -> None:
    if dataset is not None:
        assert 'asin' in dataset.columns
        assert 'reviewerID' in dataset.columns
        assert 'summary' in dataset.columns
        df = dataset
    elif isinstance(data_name, str):
        df = getAmazonData(data_name)
    else:
        raise TypeError("'dataset' is neither a DataFrame nor a str!")

    df = df[df['asin'].map(df['asin'].value_counts()) >= min_item_review]
    df = df[df['summary'].notnull()]
    df = df[df['reviewText'].notnull()]
    grouped_df = df.groupby('reviewerID')

    file_mode = 'a' if os.path.isfile(os.path.join(DATASET_DIR, HDF5_DATASET)) else 'w'
    with tb.open_file(os.path.join(dir, hdf5_name), 'w') as h5f:
        # Get the HDF5 root group
        root = h5f.root

        # Create the group:
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
                review['summary']    = agg_func(val["summary"]).cpu().detach().numpy()
                review['reviewText'] = agg_func(val["reviewText"]).cpu().detach().numpy()
            
                # This injects the Record values
                review.append()
            except:
                print(f"Error Found when processing user \#{i}")
                print(f"summary:    {val['summary']}")
                print(f"reviewText: {val['reviewText']}")
                continue
    
        # Flush the table buffers
        table.flush()

def save_meta_dataset(data_name: str, dataset=None, agg_func=lambda x: x, save_path="./") -> None:
    if dataset is not None:
        assert 'asin' in dataset.columns
        df = dataset
    elif isinstance(dataset, str):
        df = getAmazonData(data_name)
    else:
        raise TypeError("'dataset' is neither a DataFrame nor a str!")

    raise NotImplementedError

def describe_dataset(data_name: str, dataset=None, is_meta=False) -> None:
    dataset = getAmazonData(data_name) if dataset is None else dataset
    columns = list(dataset.columns)

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("input 'dataset' is not a dataframe")
    print(f"Dataset name: {DATASET_NAME['food']}")
    print(f"Dataset size: {len(dataset)}")
    print("Dataset columns:")
    for col in columns:
        if dataset[col].dtypes in [float, int]:
            print(f"\t'{col}'--Unique: {dataset[col].nunique()}")
            print(" "*(len(col)+2) + f"\t--max: {dataset[col].max()}, mean: {dataset[col].mean()}, min: {dataset[col].min()}")
        else:
            try:
                print(f"\t'{col}'--Unique: {dataset[col].nunique()}")
            except:
                print(f"\t'{col}'--Total: {dataset[col].count()}")

def count_regex_match(dataset: pd.DataFrame, column: str, regex: str):
    return dataset[column].str.contains(regex).sum()

def top_phrases(dataset: pd.DataFrame, column: str, phrase_length: int or Iterable, print_top50=False) -> pd.Series:
    if dataset[column].dtypes != object:
        raise TypeError(f"The targeted column '{column}' doesn't have the correct dtype 'object'!")
    if isinstance(phrase_length, int):
        n = [phrase_length]
    elif isinstance(phrase_length, Iterable):
        n = phrase_length
    else:
        raise TypeError(f"'phrase_length' must be an integer or an iterable of integers!")
    data_split = [y for x in dataset[column] for y in str(x).split()]
    phrase_counter = pd.Series(
        [' '.join(y) for x in n for y in nltk.ngrams(data_split, x)]
    ).value_counts()
    
    if print_top50:
        with pd.option_context('display.max_rows', 50, 'display.max_columns', 2):
            if len(phrase_counter) >= 50:
                print(phrase_counter.iloc[:50])
            else:
                print(phrase_counter)
    return phrase_counter

def sentence_count(dataset: pd.DataFrame, column: str, print_top50=False) -> pd.Series:
    if dataset[column].dtypes != object:
        raise TypeError(f"The targeted column '{column}' doesn't have the correct dtype 'object'!")

    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    sentence_counts = dataset[column].apply(lambda x: len(tokenizer.tokenize(str(x)))).value_counts()

    if print_top50:
        with pd.option_context('display.max_rows', 50, 'display.max_columns', 3):
            if len(sentence_counts) >= 50:
                top_50 = sentence_counts.iloc[:50]
                df = pd.DataFrame({'Sentence Length':top_50.index, 'Counts':top_50.values})
                print(df)
            else:
                df = pd.DataFrame({'Sentence Length':sentence_counts.index, 'Counts':sentence_counts.values})
                print(df)
    return sentence_counts


if __name__ == '__main__':
    df = getAmazonData('food', 'all')
    save_review_dataset(data_name='food', dataset=df, agg_func=get_embedding)
