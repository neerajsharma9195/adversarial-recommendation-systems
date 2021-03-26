from torch.utils.data import DataLoader
import torch
from src.preprocessing.dataloader import UserDataset
from src.models.collaborative_filtering import collaborative_filter
import random

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
dataset = UserDataset(data_name='food', load_full=False)

# length = int(len(dataset) * 0.8)
# train_set, test_set = torch.utils.data.random_split(dataset, [length, len(dataset) - length])
# val_size = int(len(train_set) * 0.2)
# train_data, val_data = torch.utils.data.random_split(train_set, [len(train_set) - val_size, val_size])
# train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=0)
# test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

tiny_size = 35000
tiny_dataset, remainder = torch.utils.data.random_split(dataset, [tiny_size, len(dataset)-tiny_size])
length = int(len(tiny_dataset) * 0.8)
train_set, test_set = torch.utils.data.random_split(tiny_dataset, [length, len(tiny_dataset) - length])
val_size = int(len(tiny_dataset) * 0.2)
train_data, val_data = torch.utils.data.random_split(train_set, [len(train_set) - val_size, val_size])
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

numUsers = dataset.numIDs
numItems = dataset.numItems
user_embedding_dim = 128
noise_size = 128

collaborative_filter(val_loader.dataset)
