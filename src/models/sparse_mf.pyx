import time
import progressbar
import numpy as np
from scipy import sparse
from surprise import accuracy
from collections import defaultdict

def predict(unmasked_R_coo, unmasked_R_csr, algo):
    predictions = defaultdict(defaultdict())
    for i, j, val in zip(unmasked_R_coo.row, unmasked_R_coo.col, unmasked_R_coo.data):
        prediction = algo.predict(iid=i, uid=j, r_ui=val)
        predictions[i][j] = prediction

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def get_P_and_R(ks, predictions):
    for k in ks:
        precision_recall_at_k(predictions, k=10, threshold=3.5)
    # return precisions, recalls

def mae_and_rmse(predictions):
    pass

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


# def sort_coo(m):
#     tuples = zip(m.row, m.col, m.data)
#     return sorted(tuples, key=lambda x: x[2], reverse=True)

# def sort_csr(m):
#     tuples = zip(m.row, m.col, m.data)
#     return sorted(tuples, key=lambda x: x[2], reverse=True)

# def get_popularity_preds(masked_R_coo, mask_coo):
#     avg_item_rating = masked_R_coo.mean(axis=0).T
#     avg_user_rating = masked_R_coo.mean(axis=1)
#     result = sparse.csr_matrix(mask_coo.shape)
#     for i, j in zip(mask_coo.row, mask_coo.col):
#         result[i,j] = (avg_user_rating[i] + avg_item_rating[j]) / 2
#         # result[i,j] = avg_item_rating[j]  # same for every user
#     assert(result.nnz == mask_coo.nnz)
#     return result

# def getPandR(ks, predictions_coo, predictions_csr, ground_truth_csr, mask_csr):
#     assert(predictions_coo.nnz == mask_csr.nnz)
#     sorted_predictions = sort_coo(predictions_coo)
#     precisions, recalls = [], []
#     assert(len(sorted_predictions) == mask_csr.nnz)
#     for k in ks:
#         true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
#         for i, j, v in sorted_predictions[:k]:
#             if ground_truth_csr[i,j] >= 3.5:
#                 if predictions_csr[i,j] >= 3.5:
#                     true_pos += 1
#                 if predictions_csr[i,j] < 3.5:
#                     false_neg += 1
#             if ground_truth_csr[i,j] < 3.5:
#                 if predictions_csr[i,j] >= 3.5:
#                     false_pos += 1
#         precision = true_pos / k
#         recall = true_pos / (true_pos + false_neg + .00000001)
#         precisions.append(round(precision, 8))
#         recalls.append(round(recall, 8))
#     return precisions, recalls

# def MAE_and_RMSE(predictions_csr, ground_truth_csr, mask_coo, predictions=None):
#     if predictions:
#         return accuracy.mae(predictions), accuracy.rmse(predictions)
#     mae, rmse = 0, 0
#     total = mask_coo.nnz
#     for i, j in zip(mask_coo.row, mask_coo.col):
#         mae += abs(predictions_csr[i,j] - ground_truth_csr[i,j])
#         rmse += (predictions_csr[i,j] - ground_truth_csr[i,j])**2
#     mae /= total
#     rmse /= total
#     return mae, np.sqrt(rmse)