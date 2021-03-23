import os
import torch
import numpy as np
import tables as tb
from src.preprocessing.preprocessing import DATASET_DIR, HDF5_DATASET, DATASET_NAME

from typing import Union, List, Tuple


class UserDataset(torch.utils.data.Dataset):
    """Implements Dataloader"""
    PATH = os.path.join(DATASET_DIR, HDF5_DATASET)

    @staticmethod
    def hdfarray_to_numpy(hdfarray: tb.CArray):
        array = np.array(shape=hdfarray.shape, dtype=hdfarray.dtype)
        array[:] = hdfarray[:]
        return array

    @staticmethod
    def hdfarray_to_tensor(hdfarray: tb.CArray):
        return torch.from_numpy(hdfarray[:])

    def __init__(self, data_name: str, path=PATH, load_full=False):
        self.load_full = load_full
        self.h5f = tb.open_file(path, 'r')
        self.review_table = self.h5f.root[data_name]['Review']
        self.interact_table = self.h5f.root[data_name]['Interactions']
        self.numIDs, self.numItems = self.interact_table.shape
        self.interactions = None
        self.reviewerIDs = None
        self.review_embeddings = None
        # self.conditional_vectors = torch.diag(torch.ones(self.numIDs, dtype=torch.int64))

        if self.load_full:
            self.interactions = self.hdfarray_to_tensor(self.interact_table)
            self.reviewerIDs = self.get_reviewerIDs()
            self.review_embeddings = self.get_userReviews()
            self.h5f.close()

    def get_reviewerID(self, idx) -> str:
        if self.reviewerIDs is None:
            return self.review_table[idx]['reviewerID'].decode('utf-8')

        return self.reviewerIDs[idx]

    def get_reviewerIDs(self) -> List[str]:
        if self.reviewerIDs is None:
            reviewerIDs = list(map(
                lambda row: row['reviewerID'].decode('utf-8'),
                self.review_table.iterrows()
            ))
            return reviewerIDs

        return self.reviewerIDs

    def get_userReviews(self) -> torch.Tensor:
        if self.review_embeddings is None:
            review_embeddings = torch.vstack(tuple(map(
                lambda row: torch.from_numpy(row['reviewText']),
                self.review_table.iterrows()
            )))
            return review_embeddings

        return self.review_embeddings

    def get_interactions(self, style='tensor') -> Union[torch.Tensor, np.ndarray]:
        if style == 'tensor':
            return self.interactions
        elif style == 'numpy':
            return self.interactions.numpy()
        else:
            raise NameError("style must be 'tensor' or 'numpy'!")

    def __len__(self) -> int:
        return self.numIDs

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.load_full:
            user_reviews_embedding = self.review_embeddings[idx]
            user_ratings = self.interactions[idx]
        else:
            user_reviews_embedding = torch.from_numpy(self.review_table[idx]['reviewText'].astype(np.float32))
            user_ratings = torch.from_numpy(self.interact_table[idx].astype(np.float32))
        # conditional_vector = self.conditional_vectors[idx]
        return user_reviews_embedding, user_ratings


class ItemDataset(UserDataset):
    def get_itemReviews(self, mask) -> torch.Tensor:
        if self.review_embeddings is None:
            idx = torch.masked_select(torch.arange(0, self.numIDs), mask)
            item_reviews = self.review_table[idx.numpy()]['reviewText'][:,-1,:]
            item_reviews_embedding = torch.from_numpy(
                np.mean(item_reviews, axis=0, keepdims=True)
            )
        else:
            item_reviews_embedding = torch.mean(
                input=self.review_embeddings[mask],
                dim=0,
                keepdim=True
            )

        return item_reviews_embedding

    def __len__(self) -> int:
        return self.numItems

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.load_full:
            item_ratings = self.interactions[:, idx]
        else:
            item_ratings = torch.from_numpy(self.interact_table[:, idx])

        mask = item_ratings > 0
        item_reviews_embedding = self.get_itemReviews(mask)

        return item_reviews_embedding, item_ratings


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    item_dataset = ItemDataset(data_name='food', load_full=True)
    user_dataset = UserDataset(data_name='food', load_full=True)

    length = int(len(item_dataset) * 0.5)
    train_set, val_set = torch.utils.data.random_split(item_dataset, [length, len(item_dataset) - length])
    loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    for i, batch in enumerate(loader):
        if i < 1:
            item_reviews_embedding, item_ratings = batch
            print(f"item_reviews_embedding: {item_reviews_embedding.size()}")
            print(item_reviews_embedding)
            print(f"item_ratings: {item_ratings.size()}")
            print(item_ratings)
        else:
            break
    
    length = int(len(user_dataset) * 0.5)
    train_set, val_set = torch.utils.data.random_split(user_dataset, [length, len(user_dataset) - length])
    loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    for i, batch in enumerate(loader):
        if i < 1:
            user_reviews_embedding, user_ratings = batch
            print(f"user_reviews_embedding: {user_reviews_embedding.size()}")
            print(user_reviews_embedding)
            print(f"user_ratings: {user_ratings.size()}")
            print(user_ratings)
        else:
            break
