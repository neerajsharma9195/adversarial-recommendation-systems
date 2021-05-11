import numpy as np
from scipy import sparse
from numba import njit, prange
from src.cf.utils.numba_utils import getitem_by_row_col

from src.cf.evaluation.math_numba import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score
)

from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    log_loss,
)

from src.cf.utils.constants import (
    DEFAULT_K
)

from typing import Tuple


@njit
def merge_rating_true_pred(
    rating_true: np.ndarray,
    rating_pred: np.ndarray,
    true_uid: np.ndarray,
    true_iid: np.ndarray) -> Tuple[np.ndarray]:

    pred = getitem_by_row_col(rating_pred, true_uid, true_iid)
    actual = getitem_by_row_col(rating_true, true_uid, true_iid)

    return actual, pred


def rmse(
    rating_true,
    rating_pred,
):
    """Calculate Root Mean Squared Error
    Args:
        rating_true (pd.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pd.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
    Returns:
        float: Root mean squared error
    """
    rating_true_coo = sparse.coo_matrix(rating_true)
    true_uid, true_iid = rating_true_coo.row, rating_true_coo.col
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        true_uid=true_uid,
        true_iid=true_iid
    )
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(
    rating_true,
    rating_pred,
):
    """Calculate Mean Absolute Error.
    Args:
        rating_true (pd.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pd.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
    Returns:
        float: Mean Absolute Error.
    """

    rating_true_coo = sparse.coo_matrix(rating_true)
    true_uid, true_iid = rating_true_coo.row, rating_true_coo.col
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        true_uid=true_uid,
        true_iid=true_iid
    )
    return mean_absolute_error(y_true, y_pred)


def rsquared(
    rating_true,
    rating_pred
):
    """Calculate R squared
    Args:
        rating_true (pd.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pd.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
    
    Returns:
        float: R squared (min=0, max=1).
    """

    rating_true_coo = sparse.coo_matrix(rating_true)
    true_uid, true_iid = rating_true_coo.row, rating_true_coo.col
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        true_uid=true_uid,
        true_iid=true_iid
    )
    return r2_score(y_true, y_pred)


def exp_var(
    rating_true,
    rating_pred,
):
    """Calculate explained variance.
    Args:
        rating_true (pd.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pd.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs

    Returns:
        float: Explained variance (min=0, max=1).
    """

    rating_true_coo = sparse.coo_matrix(rating_true)
    true_uid, true_iid = rating_true_coo.row, rating_true_coo.col
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        true_uid=true_uid,
        true_iid=true_iid
    )
    return explained_variance_score(y_true, y_pred)


def auc(
    rating_true,
    rating_pred
):
    """Calculate the Area-Under-Curve metric for implicit feedback typed
    recommender, where rating is binary and prediction is float number ranging
    from 0 to 1.
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
    Note:
        The evaluation does not require a leave-one-out scenario.
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users.
    Args:
        rating_true (pd.DataFrame): True data
        rating_pred (pd.DataFrame): Predicted data
    Returns:
        float: auc_score (min=0, max=1)
    """

    rating_true_coo = sparse.coo_matrix(rating_true)
    true_uid, true_iid = rating_true_coo.row, rating_true_coo.col
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        true_uid=true_uid,
        true_iid=true_iid
    )
    return roc_auc_score(y_true, y_pred)


def logloss(
    rating_true,
    rating_pred
):
    """Calculate the logloss metric for implicit feedback typed
    recommender, where rating is binary and prediction is float number ranging
    from 0 to 1.
    https://en.wikipedia.org/wiki/Loss_functions_for_classification#Cross_entropy_loss_(Log_Loss)
    Args:
        rating_true (pd.DataFrame): True data
        rating_pred (pd.DataFrame): Predicted data
    Returns:
        float: log_loss_score (min=-inf, max=inf)
    """

    rating_true_coo = sparse.coo_matrix(rating_true)
    true_uid, true_iid = rating_true_coo.row, rating_true_coo.col
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        true_uid=true_uid,
        true_iid=true_iid
    )
    return log_loss(y_true, y_pred)


@njit(parallel=True)
def get_top_k_items(interaction: np.ndarray, k: int) -> np.ndarray:
    num_row, num_col = interaction.shape

    neg_interaction = -interaction
    col_idx = np.empty((num_row, k), np.int64)
    for i in prange(num_row):
        col_idx[i] = np.argsort(neg_interaction[i])[:k]
    col_idx = col_idx.flatten()
    
    row_idx = np.repeat(np.arange(num_row), k)
    
    return row_idx, col_idx

@njit(parallel=True)
def merge_ranking_true_pred(
    rating_true: np.ndarray,
    rating_pred: np.ndarray,
    relevancy_method: str,
    k: int,
    threshold: float) -> Tuple[np.ndarray]:

    num_row_rating_true, num_col_rating_true = rating_true.shape
    num_row_rating_pred, num_col_rating_pred = rating_pred.shape
    assert num_row_rating_true == num_row_rating_pred
    assert num_col_rating_true == num_col_rating_pred

    if relevancy_method == "top_k":
        top_k = k
    else:
        raise NotImplementedError("Invalid relevancy_method")
    
    top_k_pred_row, top_k_pred_col = get_top_k_items(
        interaction=rating_pred,
        k=top_k,
    )

    top_k_pred = getitem_by_row_col(rating_pred, top_k_pred_row, top_k_pred_col)
    top_k_pred = top_k_pred.reshape(num_row_rating_pred, top_k)
    top_k_pred_true = getitem_by_row_col(rating_true, top_k_pred_row, top_k_pred_col)
    top_k_pred_true = top_k_pred_true.reshape(num_row_rating_true, top_k)

    # pred_nonzero_row, pred_nonzero_col = np.nonzero(top_k_pred)
    # true_pred_nonzero_row, true_pred_nonzero_col = np.nonzero(top_k_pred_true)

    bool_rating_true = rating_true > 0
    bool_top_k_pred_true = top_k_pred_true > 0
    hit_count = np.sum(bool_top_k_pred_true, axis=1)
    actual_count = np.sum(bool_rating_true, axis=1)
    n_users = hit_count.shape[0]

    return (
        top_k_pred, top_k_pred_true,
        hit_count, actual_count, n_users
    )


def precision_at_k(
    rating_true: np.ndarray,
    rating_pred: np.ndarray,
    train_row: np.ndarray,
    train_col: np.ndarray,
    real_row: np.ndarray,
    real_col: np.ndarray,
    relevancy_method,
    k=DEFAULT_K,
    threshold=0) -> float:

    """Precision at K.
    Note:
        We use the same formula to calculate precision@k as that in Spark.
        More details can be found at
        http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.precisionAt
        In particular, the maximum achievable precision may be < 1, if the number of items for a
        user in rating_pred is less than k.
    Args:
        rating_true (pd.DataFrame): True DataFrame
        rating_pred (pd.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the 
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)
    Returns:
        float: precision at k (min=0, max=1)
    """

    # (_, _, _, _, hit_count, _, n_users) = merge_ranking_true_pred(
    #     rating_true=rating_true,
    #     rating_pred=rating_pred,
    #     relevancy_method=relevancy_method,
    #     k=k
    # )

    # if hit_count.shape[0] == 0:
    #     return 0.0

    # return (hit_count / k).sum() / n_users

    # Remove ratings that are seen in the training set
    rating_true[train_row, train_col] = 0
    rating_pred[train_row, train_col] = 0

    rating_true = rating_true[real_rows, :]
    rating_true = rating_true[:, real_cols]
    
    rating_pred = rating_pred[real_rows, :]
    rating_pred = rating_pred[:, real_cols]

    (top_k_pred, top_k_pred_true, _, _, _) = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold
    )
    
    if threshold == 0:
        n_rel = np.sum(rating_true > threshold, axis=1)
        n_rec_k = np.sum(top_k_pred > threshold, axis=1)

        top_k_pred_true[top_k_pred < threshold] = 0
        n_rel_and_rec_k = np.sum(top_k_pred_true > threshold, axis=1)
    else:
        n_rel = np.sum(rating_true >= threshold, axis=1)
        n_rec_k = np.sum(top_k_pred >= threshold, axis=1)

        top_k_pred_true[top_k_pred < threshold] = 0
        n_rel_and_rec_k = np.sum(top_k_pred_true >= threshold, axis=1)

    zero_n_rec_k = (n_rec_k == 0)
    n_rec_k[zero_n_rec_k] = 1
    precision = n_rel_and_rec_k / n_rec_k
    precision[zero_n_rec_k] = 0

    return np.average(precision)


def recall_at_k(
    rating_true,
    rating_pred,
    train_row,
    train_col,
    real_row: np.ndarray,
    real_col: np.ndarray,
    relevancy_method,
    k=DEFAULT_K,
    threshold=0
):
    """Recall at K.
    Args:
        rating_true (pd.DataFrame): True DataFrame
        rating_pred (pd.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the 
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)
    Returns:
        float: recall at k (min=0, max=1). The maximum value is 1 even when fewer than 
        k items exist for a user in rating_true.
    """

    # (_, _, _, _, hit_count, actual_count, n_users) = merge_ranking_true_pred(
    #     rating_true=rating_true,
    #     rating_pred=rating_pred,
    #     relevancy_method=relevancy_method,
    #     k=k
    # )

    # if hit_count.shape[0] == 0:
    #     return 0.0

    # return (hit_count / actual_count).sum() / n_users

    # Remove ratings that are seen in the training set
    rating_true[train_row, train_col] = 0
    rating_pred[train_row, train_col] = 0

    rating_true = rating_true[real_rows, :]
    rating_true = rating_true[:, real_cols]
    
    rating_pred = rating_pred[real_rows, :]
    rating_pred = rating_pred[:, real_cols]

    (top_k_pred, top_k_pred_true, _, _, _) = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold
    )
    
    if threshold == 0:
        n_rel = np.sum(rating_true > threshold, axis=1)
        n_rec_k = np.sum(top_k_pred > threshold, axis=1)

        top_k_pred_true[top_k_pred < threshold] = 0
        n_rel_and_rec_k = np.sum(top_k_pred_true > threshold, axis=1)
    else:
        n_rel = np.sum(rating_true >= threshold, axis=1)
        n_rec_k = np.sum(top_k_pred >= threshold, axis=1)

        top_k_pred_true[top_k_pred < threshold] = 0
        n_rel_and_rec_k = np.sum(top_k_pred_true >= threshold, axis=1)

    zero_n_rel = (n_rel == 0)
    n_rel[zero_n_rel] = 1
    recall = n_rel_and_rec_k / n_rel
    recall[zero_n_rel] = 0

    return np.average(recall)
