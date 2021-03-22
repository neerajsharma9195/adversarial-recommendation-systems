from torch.utils.data import DataLoader
import torch
from src.preprocessing.dataloader import UserDataset
from src.models.arcf import train_user_ar

dataset = UserDataset(data_name='food', load_full=False)
length = int(len(dataset) * 0.8)
train_set, test_set = torch.utils.data.random_split(dataset, [length, len(dataset) - length])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)
numUsers = dataset.numIDs
numItems = dataset.numItems
user_embedding_dim = 128
noise_size = 128

for i, batch in enumerate(train_loader):
    if i < 5:
        review_embeddings, rating_vectors, conditional_vector = batch
        print(review_embeddings.shape)
        print(rating_vectors.shape)
        print(conditional_vector.shape)
    else:
        break


#train_user_ar(train_loader, numUsers, user_embedding_dim,noise_size, numItems, review_embedding_size=128, use_reviews=False)




