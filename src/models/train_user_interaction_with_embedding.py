from torch.utils.data import DataLoader
import torch
from src.preprocessing.dataloader import UserDataset
from src.models.arcf_with_embedding import train_user_ar
import random
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
dataset = UserDataset(data_name='food', load_full=False)
length = int(len(dataset) * 0.8)
train_set, test_set = torch.utils.data.random_split(dataset, [length, len(dataset) - length])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
numUsers = dataset.numIDs
numItems = dataset.numItems
user_embedding_dim = 128
noise_size = 128


train_user_ar(user_train_dataloader=train_loader, user_test_data_loader=test_loader,
              num_users=numUsers, user_embedding_dim=user_embedding_dim, noise_size=noise_size, num_items=numItems,
              review_embedding_size=128, use_reviews=True)
