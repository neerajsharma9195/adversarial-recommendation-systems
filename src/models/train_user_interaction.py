import sys
from torch.utils.data import DataLoader
import torch
from src.preprocessing.dataloader import UserDataset
from src.models.arcf import train_user_ar
import random
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# user datasets
train = UserDataset(data_name='food', load_full=True, masked='full')
val = UserDataset(data_name='food', load_full=True, masked='partial')
test = UserDataset(data_name='food', load_full=True, masked='no')
assert(train.numIDs == val.numIDs and train.numIDs == test.numIDs)
print(train.shape, val.shape, test.shape)

# user dataloaders
train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test, batch_size=1, shuffle=True, num_workers=0)

# interactions
train_interactions = train.get_interactions(style='tensor')
val_interactions = val.get_interactions(style='tensor')
test_interactions = test.get_interactions(style='tensor')

# variables
numUsers = train.numIDs
numItems = train.numItems
user_embedding_dim = 128
noise_size = 128
USE_REVIEWS = False
# USE_REVIEWS = True

train_user_ar(user_train_dataloader=train_loader, user_test_data_loader=val_loader,
              num_users=numUsers, user_embedding_dim=user_embedding_dim, noise_size=noise_size, num_items=numItems,
              review_embedding_size=128, use_reviews=USE_REVIEWS)