from torch.utils.data import DataLoader
import torch
from src.preprocessing.dataloader import ItemDataset
from src.models.arcf import train_item_ar
import random

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
dataset = ItemDataset(data_name='food', load_full=True)
length = int(len(dataset) * 0.8)
train_set, test_set = torch.utils.data.random_split(dataset, [length, len(dataset) - length])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
numUsers = dataset.numIDs
numItems = dataset.numItems
item_embedding_dim = 128
noise_size = 128

train_item_ar(item_train_dataloader=train_loader, item_test_dataloader=test_loader,
              num_users=numUsers, item_embedding_dim=item_embedding_dim, noise_size=noise_size, num_items=numItems,
              review_embedding_size=128, use_reviews=True)
