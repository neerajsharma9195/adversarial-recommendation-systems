from src.preprocessing.dataloader import UserDataset
from src.models.MF.torchmf import (BaseModule, BPRModule, BasePipeline,
                                   bpr_loss, PairwiseInteractions)

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp


def train_test_split(interactions, n=10):
    """
    Split an interactions matrix into training and test sets.
    Parameters
    ----------
    interactions : np.ndarray
    n : int (default=10)
        Number of items to select / row to place into test.
    Returns
    -------
    train : np.ndarray
    test : np.ndarray
    """
    test = np.zeros(interactions.shape)
    train = interactions.copy()
    for user in range(interactions.shape[0]):
        if interactions[user, :].nonzero()[0].shape[0] > n:
            test_interactions = np.random.choice(interactions[user, :].nonzero()[0],
                                                 size=n,
                                                 replace=False)
            train[user, test_interactions] = 0.
            test[user, test_interactions] = interactions[user, test_interactions]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test

def explicit(train, test):
    pipeline = BasePipeline(train, test=test, model=BaseModule,
                            n_factors=10, batch_size=2048, dropout_p=0.02,
                            lr=0.01, weight_decay=0.01,
                            optimizer=torch.optim.Adam, n_epochs=80,
                            verbose=True, random_seed=2017)
    pipeline.fit()


def implicit(train, test):
    pipeline = BasePipeline(train, test=test, verbose=True,
                           batch_size=1024, num_workers=4,
                           n_factors=20, weight_decay=0,
                           dropout_p=0., lr=.2, sparse=True,
                           optimizer=torch.optim.SGD, n_epochs=40,
                           random_seed=2017, loss_function=bpr_loss,
                           model=BPRModule,
                           interaction_class=PairwiseInteractions,
                           eval_metrics=('auc', 'patk'))
    pipeline.fit()


def hogwild(train, test, nWorkers=32):
    pipeline = BasePipeline(train, test=test, verbose=True,
                            batch_size=1024, num_workers=nWorkers,
                            n_factors=20, weight_decay=0,
                            dropout_p=0., lr=.2, sparse=True,
                            optimizer=torch.optim.SGD, n_epochs=40,
                            random_seed=2017, loss_function=bpr_loss,
                            model=BPRModule, hogwild=True,
                            interaction_class=PairwiseInteractions,
                            eval_metrics=('auc', 'patk'))
    pipeline.fit()


if __name__ == '__main__':
    userData = UserDataset(data_name='food', load_full=True)
    interactions = userData.get_interactions(style='numpy')

    train, test = train_test_split(interactions, n=5)
    train = sp.coo_matrix(train)
    test = sp.coo_matrix(test)

    explicit(train, test)