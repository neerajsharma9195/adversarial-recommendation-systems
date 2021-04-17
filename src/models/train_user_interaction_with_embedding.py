import torch
from torch.utils.data import DataLoader
from src.preprocessing.dataloader import UserDataset
from src.models.arcf_with_embedding import train_user_ar
import random

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

training_dataset = UserDataset(data_name='food', mode='train')
train_loader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=16)

numUsers = training_dataset.numIDs
numItems = training_dataset.numItems
print("train numUsers {}".format(numUsers))
print("train numItems {}".format(numItems))

user_embedding_dim = 128
noise_size = 128

train_user_ar(user_train_dataloader=train_loader, user_test_data_loader=None,
              num_users=numUsers, user_embedding_dim=user_embedding_dim, noise_size=noise_size, num_items=numItems,
              review_embedding_size=128, use_reviews=True,
              output_path='/mnt/nfs/scratch1/neerajsharma/model_params/complete_data_results')
