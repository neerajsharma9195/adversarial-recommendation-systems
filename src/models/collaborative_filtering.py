import surprise
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import os
from surprise.model_selection import train_test_split as surprise_train_test_split
from src.preprocessing.dataloader import UserDataset

# header_list = ["userID", "itemID", "rating", "reviewUnixTime"]

# rating_file_path = os.path.join('/mnt/nfs/scratch1/neerajsharma/amazon_data/', 'ratings_Beauty.csv')

# ratings_df = pd.read_csv(rating_file_path, names=header_list)

# print("Total data ")
# print("Total no of ratings :", ratings_df.shape[0])
# print("Total No of Users   :", len(np.unique(ratings_df['userID'])))
# print("Total No of products  :", len(np.unique(ratings_df['itemID'])))

def collaborative_filter(interaction_matrix):
    # ratings_df.drop(['reviewUnixTime'], axis=1, inplace=True)

    # new_df = ratings_df.groupby("itemID").filter(lambda x: x['rating'].count() >= 50)
    # new_df.reset_index(inplace=True)

    # reader = Reader(rating_scale=(1, 5))
    # data = Dataset.load_from_df(new_df[['userID', 'itemID', 'rating']], reader)
    # # Use user_based true/false to switch between user-based or item-based collaborative filtering
    # trainset, testset = surprise_train_test_split(data, test_size=0.3, random_state=10)

    algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})

    # train model
    
    algo.fit(trainset)

    # on test data:

    # run the trained model against the testset
    test_pred = algo.test(testset)
    print("Item-based Model : Test Set")
    accuracy.rmse(test_pred, verbose=True)
    accuracy.mae(test_pred, verbose=True)

    # using svd based method for matrix factorization

    new_df1 = new_df.head(100000)
    ratings_matrix = new_df1.pivot_table(values='rating', index='userID', columns='itemID', fill_value=0)
    ratings_matrix.head()

    print(ratings_matrix.shape)

    X = ratings_matrix.T

    # Decomposing the Matrix
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    print(decomposed_matrix.shape)

    # Correlation Matrix
    correlation_matrix = np.corrcoef(decomposed_matrix)
    print(correlation_matrix.shape)

    sample_item_index = 0

    sample_item_id = X.index[sample_item_index]

    product_names = list(X.index)
    product_ID = product_names.index(sample_item_id)

    print("product id {}".format(product_ID))

    correlation_product_ID = correlation_matrix[product_ID]

    print(correlation_product_ID.shape)

    recommended = list(X.index[correlation_product_ID > 0.65])

    # Removes the item already bought by the customer
    recommended.remove(sample_item_id)

    print(recommended)
    return recommended

