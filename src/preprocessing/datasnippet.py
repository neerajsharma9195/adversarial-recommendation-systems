import nltk
import pandas as pd

from collections.abc import Iterable
from src.preprocessing.utils import (
    DATASET_DIR, META_PREFIX, REVIEW_PREFIX, HDF5_DATASET, DATASET_NAME,
    getAmazonData
)


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
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
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
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
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
