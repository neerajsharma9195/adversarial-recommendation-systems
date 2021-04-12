import time
import progressbar
import numpy as np
from scipy import sparse

def sort_coo(m):
    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: x[2], reverse=True)

def predict_with_surprise(unmasked_R_csr, mask_coo, algo):
    result = sparse.csr_matrix(mask_coo.shape)
    for i, j in zip(mask_coo.row, mask_coo.col):
        output = algo.predict(str(i), str(j), r_ui=unmasked_R_csr[i,j])
        result[i,j] = output[3]
    return result

def get_popularity_preds(masked_R_csr, mask_coo):
    avg_item_rating = masked_R_csr.mean(axis=0).T
    avg_user_rating = masked_R_csr.mean(axis=1)
    result = sparse.csr_matrix(mask_coo.shape)
    print('num rows = ', len(mask_coo.row))
    print('avg user rating = ', avg_user_rating.shape)
    print('num cols = ', len(mask_coo.col))
    print('avg item rating = ', avg_item_rating.shape)
    for i, j in zip(mask_coo.row, mask_coo.col):
        result[i,j] = (avg_user_rating[i] + avg_item_rating[j]) / 2
    return result

def getPandR(ks, predictions, predictions_csr, ground_truth_csr, mask_csr):
    print(predictions.nnz, mask_csr.nnz)
    assert(predictions.nnz == mask_csr.nnz)
    sorted_predictions = sort_coo(predictions)
    precisions, recalls = [], []
    assert(len(sorted_predictions) == mask_csr.nnz)
    for k in ks:
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
        for i, j, v in sorted_predictions[:k]:
            if ground_truth_csr[i,j] >= 3.5:
                if predictions_csr[i,j] >= 3.5:
                    true_pos += 1
                if predictions_csr[i,j] < 3.5:
                    false_neg += 1
            if ground_truth_csr[i,j] < 3.5:
                if predictions_csr[i,j] >= 3.5:
                    false_pos += 1
        precision = true_pos / (true_pos + false_pos + .00000001)
        recall = true_pos / (true_pos + false_neg + .00000001)
        precisions.append(round(precision, 8))
        recalls.append(round(recall, 8))
    return precisions, recalls

def MAE_and_RMSE(predictions_csr, ground_truth_csr, mask_coo):
    mae, rmse = 0, 0
    total = mask_coo.nnz
    for i, j in zip(mask_coo.row, mask_coo.col):
        mae += abs(predictions_csr[i,j] - ground_truth_csr[i,j])
        rmse += (predictions_csr[i,j] - ground_truth_csr[i,j])**2
    mae /= total
    rmse /= total
    return mae, np.sqrt(rmse)