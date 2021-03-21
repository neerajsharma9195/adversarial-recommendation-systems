import os
import torch
import numpy as np
import tables as tb
from preprocessing import DATASET_DIR, HDF5_DATASET, DATASET_NAME

from typing import Union, List, Tuple


class Dataset(torch.utils.data.Dataset):
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
        self.interactions = self.hdfarray_to_tensor(self.interact_table)
        self.reviewerIDs = None
        self.review_embeddings = None

        if self.load_full:
            self.reviewerIDs = self.get_reviewerIDs()
            self.review_embeddings = self.get_reviewEmbeddings()
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
        
    def get_reviewEmbeddings(self) -> List[torch.Tensor]:
        if self.review_embeddings is None:
            review_embeddings = list(map(
                lambda row: torch.from_numpy(row['reviewText']),
                self.review_table.iterrows()
            ))
            return review_embeddings
        
        return self.review_embeddings

    def get_interactions(self, style='tensor') -> Union[torch.Tensor, np.ndarray]:
        if style == 'tensor':
            return self.interactions
        elif style == 'numpy':
            return self.hdfarray_to_numpy(self.interact_table)
        else:
            raise NameError("style must be 'tensor' or 'numpy'!")
    
    def __len__(self) -> int:
        return self.numIDs
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.load_full:
            review_embedding = self.review_embeddings[idx]
        else:
            review_embedding = torch.from_numpy(self.review_table[idx]['reviewText'])
        user_ratings = self.interactions[idx]
        return review_embedding, user_ratings


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = Dataset(data_name='food', load_full=True)
    loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=6)
    for i, batch in enumerate(loader):
        print(i, batch)
