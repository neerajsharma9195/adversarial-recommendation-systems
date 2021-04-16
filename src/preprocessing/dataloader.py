import os
from scipy.sparse import coo_matrix
import torch
import numpy as np
import tables as tb
from src.preprocessing.utils import DATASET_DIR, HDF5_DATASET, DATASET_NAME

from typing import Union, List, Tuple


class UserDataset(torch.utils.data.Dataset):
    """Implements User Dataloader"""
    PATH = os.path.join(DATASET_DIR, HDF5_DATASET)
    VAL_PREFIX = 'validation_'
    TRAIN_PREFIX = 'training_'

    @staticmethod
    def hdfarray_to_numpy(hdfarray: tb.CArray):
        array = np.empty(shape=hdfarray.shape, dtype=hdfarray.dtype)
        array[:] = hdfarray[:]
        return array

    @staticmethod
    def hdfarray_to_tensor(hdfarray: tb.CArray):
        return torch.from_numpy(hdfarray[:])

    @staticmethod
    def vec_to_sparse(i: torch.tensor, j: torch.tensor, v: torch.tensor,
                      size=None, load_full=False, style='tensor') -> Union[torch.Tensor, coo_matrix, np.ndarray]:
        if style == 'tensor':
            idx = torch.stack((i, j), axis=0)
            sparsed_matrix = torch.sparse_coo_tensor(idx, v, size=size)
            if load_full:
                sparsed_matrix = sparsed_matrix.to_dense()
        elif style == 'numpy':
            if len(i.shape) + len(j.shape) + len(v.shape) != 3:
                raise ValueError("scipy.coo_matrix: row, column, and data arrays must be 1-D")
            i, j = i.numpy().astype(np.uint), j.numpy().astype(np.uint)
            v = v.numpy()
            sparsed_matrix = coo_matrix( (v, (i, j)), shape=size )
            if load_full:
                sparsed_matrix = sparsed_matrix.toarray()
        else:
            raise NameError("style must be 'tensor' or 'numpy'!")

        return sparsed_matrix

    def __init__(self, data_name: str, path=PATH, masked_uid=None, masked_iid=None, masked_vid=None, mode='full'):
        self.path = path
        self.mode = mode
        self.data_name = data_name
        self.h5f = tb.open_file(path, 'r')
        cur_group = self.h5f.root[data_name]

        mask_prefix = None
        if self.mode == 'val':
            mask_prefix = UserDataset.VAL_PREFIX
        elif self.mode == 'train':
            mask_prefix = UserDataset.TRAIN_PREFIX
        elif self.mode != 'full':
            raise ValueError(f'Supported mode: {self.mode}')

        if mask_prefix is not None:
            self.userIdx = self.hdfarray_to_tensor(cur_group[mask_prefix+'masked_uid'])
            self.itemIdx = self.hdfarray_to_tensor(cur_group[mask_prefix+'masked_iid'])
            masked_vid = self.hdfarray_to_tensor(cur_group[mask_prefix+'masked_vid'])
        else:
            if masked_uid is not None:
                self.userIdx = masked_uid
            else:
                self.userIdx = self.hdfarray_to_tensor(cur_group['userIdx'])

            if masked_iid is not None:
                self.itemIdx = masked_iid
            else:
                self.itemIdx = self.hdfarray_to_tensor(cur_group['itemIdx'])
            
        if masked_vid is not None:
            self.masked_vid = masked_vid
            self.rating    = self.hdfarray_to_tensor(cur_group['rating'])[masked_vid]
            self.embedding = self.hdfarray_to_tensor(cur_group['embedding'])[masked_vid]
        else:
            self.masked_vid = None
            self.rating    = self.hdfarray_to_tensor(cur_group['rating'])
            self.embedding = self.hdfarray_to_tensor(cur_group['embedding'])

        self.idx = torch.stack((self.userIdx, self.itemIdx), axis=0)
        self.numIDs, self.numItems = int(torch.max(self.userIdx))+1, int(torch.max(self.itemIdx))+1
        self.interactions = self.vec_to_sparse(
            self.userIdx, self.itemIdx, self.rating,
            size=(self.numIDs, self.numItems),
            load_full=False,
            style='tensor'
        )
        self.review_embeddings = self.vec_to_sparse(
            self.userIdx, self.itemIdx, self.embedding,
            size=(self.numIDs, self.numItems, self.embedding.shape[1]),
            load_full=False,
            style='tensor'
        )

        self.h5f.close()
    
    def get_indices(self, style='tensor') -> Union[torch.Tensor, np.ndarray]:
        if style == 'tensor':
            return self.idx
        elif style == 'numpy':
            return np.stack((self.userIdx, self.itemIdx), axis=0)
        else:
            raise NameError("style must be 'tensor' or 'numpy'!")


    def get_reviewEmbeddings(self, load_full=False) -> Union[torch.Tensor, coo_matrix, np.ndarray]:
        if load_full:
            self.review_embeddings.to_dense()
        return self.review_embeddings       

    def get_interactions(self, load_full=False, style='tensor') -> Union[torch.Tensor, coo_matrix, np.ndarray]:
        if style == 'tensor':
            if load_full:
                return self.interactions.to_dense()
            return self.interactions

        return self.vec_to_sparse(self.userIdx, self.itemIdx, self.rating,
                                  size=(self.numIDs, self.numItems),
                                  load_full=load_full,
                                  style=style)

    def get_mask(self, drop_ratio: float, masked_uid=None, masked_iid=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        valid_mask = self.get_interactions(load_full=True) > 0
        rand_mask = torch.rand(valid_mask.shape)
        rand_mask[~valid_mask] = 0

        ref_mask = None
        if (masked_uid is not None) and (masked_iid is not None):
            ref_mask = self.vec_to_sparse(
                i=masked_uid, j=masked_iid, v=torch.ones(masked_uid.shape),
                size=(self.numIDs, self.numItems), load_full=True
            )>0

        if ref_mask is None:
            mask = rand_mask.ge(drop_ratio)
        else:
            ref_row_mask = torch.any(ref_mask, dim=1)
            rand_mask[~ref_row_mask] = 0
            mask = rand_mask.ge(drop_ratio)
            mask = torch.logical_and(mask, ref_mask)
        
        # Sanity check for rows with all `False`
        ill_masked_idx = ~torch.any(mask, dim=1)

        # Unmask the ill-masked values
        if ref_mask is None:
            mask[ill_masked_idx] = valid_mask[ill_masked_idx]
        else:
            mask[ill_masked_idx] = ref_mask[ill_masked_idx]

        print(f"Targeted drop ratio: {drop_ratio}")
        print(f"Actual drop ratio: {(valid_mask.sum() - mask.sum()) / valid_mask.sum()}")
        
        masked_uid, masked_iid, = mask.to_sparse().indices()
        masked_vid = mask[valid_mask]
        return masked_uid, masked_iid, masked_vid
    
    def save_mask(self, masked_uid: torch.Tensor, masked_iid: torch.Tensor, masked_vid: torch.Tensor, mode: str):
        if self.masked_vid is not None:
            raise TypeError("You should save your masks from the unmasked dataset!")
        
        if mode == 'val':
            prefix = UserDataset.VAL_PREFIX
        elif mode == 'train':
            prefix = UserDataset.TRAIN_PREFIX
        else:
            raise ValueError(f"Supported mode: {mode}")

        self.h5f = tb.open_file(self.path, 'a')
        cur_group = self.h5f.root[self.data_name]
        filters = tb.Filters(complib='zlib', complevel=5)
        self.h5f.create_carray(cur_group, prefix+'masked_uid', obj=masked_uid.numpy(), filters=filters)
        self.h5f.create_carray(cur_group, prefix+'masked_iid', obj=masked_iid.numpy(), filters=filters)
        self.h5f.create_carray(cur_group, prefix+'masked_vid', obj=masked_vid.numpy(), filters=filters)
        print(self.h5f)
        self.h5f.close()

    def __len__(self) -> int:
        return self.numIDs

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_reviews_embedding = self.review_embeddings[idx].coalesce().values().mean(dim=0)
        user_ratings = self.interactions[idx].to_dense()

        return user_reviews_embedding, user_ratings, torch.tensor(idx)


class ItemDataset(UserDataset):
    """Implements Item Dataloader"""
    PATH = os.path.join(DATASET_DIR, HDF5_DATASET)
    VAL_PREFIX = 'validation_'
    TRAIN_PREFIX = 'training_'

    def __init__(self, data_name: str, path=PATH, masked_uid=None, masked_iid=None, masked_vid=None, mode='full'):
        self.path = path
        self.mode = mode
        self.data_name = data_name
        self.h5f = tb.open_file(path, 'r')
        cur_group = self.h5f.root[data_name]

        mask_prefix = None
        if self.mode == 'val':
            mask_prefix = ItemDataset.VAL_PREFIX
        elif self.mode == 'train':
            mask_prefix = ItemDataset.TRAIN_PREFIX
        elif self.mode != 'full':
            raise ValueError(f'Supported mode: {self.mode}')

        if mask_prefix is not None:
            self.userIdx = self.hdfarray_to_tensor(cur_group[mask_prefix+'masked_uid'])
            self.itemIdx = self.hdfarray_to_tensor(cur_group[mask_prefix+'masked_iid'])
            masked_vid = self.hdfarray_to_tensor(cur_group[mask_prefix+'masked_vid'])
        else:
            if masked_uid is not None:
                self.userIdx = masked_uid
            else:
                self.userIdx = self.hdfarray_to_tensor(cur_group['userIdx'])

            if masked_iid is not None:
                self.itemIdx = masked_iid
            else:
                self.itemIdx = self.hdfarray_to_tensor(cur_group['itemIdx'])
        
        if masked_vid is not None:
            self.masked_vid = masked_vid
            self.rating    = self.hdfarray_to_tensor(cur_group['rating'])[masked_vid]
            self.embedding = self.hdfarray_to_tensor(cur_group['embedding'])[masked_vid]
        else:
            self.masked_vid = None
            self.rating    = self.hdfarray_to_tensor(cur_group['rating'])
            self.embedding = self.hdfarray_to_tensor(cur_group['embedding'])

        self.idx = torch.stack((self.itemIdx, self.userIdx), axis=0)
        self.numIDs, self.numItems = int(torch.max(self.userIdx))+1, int(torch.max(self.itemIdx))+1
        self.interactions = self.vec_to_sparse(
            self.itemIdx, self.userIdx, self.rating,
            size=(self.numItems, self.numIDs),
            load_full=False,
            style='tensor'
        )
        self.review_embeddings = self.vec_to_sparse(
            self.itemIdx, self.userIdx, self.embedding,
            size=(self.numItems, self.numIDs, self.embedding.shape[1]),
            load_full=False,
            style='tensor'
        )

        self.h5f.close()        

    def __len__(self) -> int:
        return self.numItems
    
    def get_indices(self, style='tensor') -> Union[torch.Tensor, np.ndarray]:
        if style == 'tensor':
            return self.idx
        elif style == 'numpy':
            return np.stack((self.itemIdx, self.userIdx), axis=0)
        else:
            raise NameError("style must be 'tensor' or 'numpy'!")

    def get_mask(self, drop_ratio: float, ref_mask=None) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError("Please use `get_mask` in UserDataset.")
    
    def save_mask(self, masked_uid: torch.Tensor, masked_iid: torch.Tensor, masked_vid: torch.Tensor, prefix: str):
        raise NotImplementedError("Please use `save_mask` in UserDataset.")


if __name__ == '__main__':
    """
    0. Flags

        load_full: bool 
        if `load_full` is True, any function which takes this flag will return a full matrix
        otherwise, it will return in the form of a spared matrix

        style: ['tensor', 'numpy']
        if `style` is `tensor`, the return value will be a torch.Tensor
        if `style` is `numpy`, the return value will be a np.ndarray
    """


    """
    1. Load the unmodified dataset
    """
    user_dataset = UserDataset(data_name='food')

    """
    2. Access attributes within the Dataset object
    """
    # return indices for all valid entries within the dataset
    # returned value will be a 2d-matrix, 
    # with 1st row as row indices and 2nd row as col indices
    indices = user_dataset.get_indices(style='tensor')
    
    # return all of the user review embeddings
    embeddings = user_dataset.get_reviewEmbeddings(load_full=False)

    # return the user-item Interactions
    interactions = user_dataset.get_interactions(load_full=True, style='tensor')

    # ...or you can access the dataset by index (through the getter function)
    user_reviews_embedding, user_ratings, idx = user_dataset[0]

    """
    3. Create masks for training/validation/testing
    """
    # Since we can use the unmodified dataset for testing, we need two more masks
    # for training and validation

    # We can generate a masking using `get_mask`
    # `drop_ratio` is the approximate percentage of rating/review we are dropping
    """
    validation_uid, validation_iid, validation_vid = user_dataset.get_mask(drop_ratio=0.3)
    """

    # To get the masking for training set, we can use the previously generated masks to get
    # a new set of masks with higher `drop_ratio`
    """
    training_uid, training_iid, training_vid = user_dataset.get_mask(
        drop_ratio=0.6, masked_uid=validation_uid, masked_iid=validation_iid
    )
    """

    # To save the generated mask into the h4 dataset, use the `save_mask` method:
    """
    user_dataset.save_mask(validation_uid, validation_iid, validation_vid, valset_prefix, mode='val')
    user_dataset.save_mask(training_uid, training_iid, training_vid, trainset_prefix, mode='train')
    """

    # Once we get the user_idx (uid), item_idx (iid), value_idx (vid), we can create
    # two new Dataset object using the masked ids.
    """
    training_dataset = UserDataset(
        data_name='food',
        masked_uid=training_uid,
        masked_iid=training_iid,
        masked_vid=training_vid
    )

    validation_dataset = UserDataset(
        data_name='food',
        masked_uid=validation_uid,
        masked_iid=validation_iid,
        masked_vid=validation_vid
    )
    """

    # ...or we can use the preprocessed masks to get training and validation set:
    training_dataset = UserDataset(
        data_name='food',
        mode='train'
    )
    validation_dataset = UserDataset(
        data_name='food',
        mode='val'
    )

    """
    4. Import Dataset object into a torch Dataloader
    """
    from torch.utils.data import DataLoader

    # In order to get data in batches, we can import our dataset into a torch dataloader
    train_loader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=16)
    val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=16)
    test_loader = DataLoader(user_dataset, batch_size=1, shuffle=True, num_workers=16)

    # Here is an example of how you can use the dataloader
    for i, batch in enumerate(train_loader):
        if i < 3:
            user_reviews_embedding, user_ratings, idx = batch
            print(f"user_reviews_embedding: {user_reviews_embedding.size()}")
            print(user_reviews_embedding)
            print(f"user_ratings: {user_ratings.size()}, # raings: {(user_ratings>0).sum()}")
            print(user_ratings)
            print()
        else:
            break
