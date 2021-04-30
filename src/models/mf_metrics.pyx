import time
import numpy as np
from scipy import sparse

def sort_coo(m):
    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: x[2], reverse=True)

def sort_csr(m):
    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: x[2], reverse=True)

def getPandR(ks, predictions, ground_truth_coo, mask_coo):
    print(predictions.shape, mask_coo.toarray().shape)
    predictions = predictions * mask_coo.toarray()
    print(predictions[:100][:100])
    predictions_csr = sparse.csr_matrix(predictions)
    predictions_coo = predictions_csr.tocoo()
    ground_truth_csr = ground_truth_coo.tocsr()
    mask_csr = mask_coo.tocsr()
    # assert(predictions_coo.nnz == mask_csr.nnz)
    print('nnz = ', predictions_coo.nnz, mask_csr.nnz)
    sorted_predictions = sort_coo(predictions_coo)
    precisions, recalls = [], []
    # assert(len(sorted_predictions) == mask_csr.nnz)
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
        precision = true_pos / k
        recall = true_pos / (true_pos + false_neg + .00000001)
        precisions.append(round(precision, 8))
        recalls.append(round(recall, 8))
    return precisions, recalls

def MAE_and_RMSE(predictions, ground_truth_coo, mask_coo):
    predictions_csr = sparse.csr_matrix(predictions)
    ground_truth_csr = ground_truth_coo.tocsr()
    mae, rmse = 0, 0
    total = mask_coo.nnz
    for i, j in zip(mask_coo.row, mask_coo.col):
        mae += abs(predictions_csr[i,j] - ground_truth_csr[i,j])
        rmse += (predictions_csr[i,j] - ground_truth_csr[i,j])**2
    mae /= total
    rmse /= total
    return mae, np.sqrt(rmse)