from torch.utils.data import DataLoader
import torch
from src.preprocessing.dataloader import UserDataset
from src.models.arcf_with_embedding import train_user_ar
import random
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# todo: small experiment settings currently. Update it to full data usage later.
train_dataset = UserDataset(data_name='food', load_full=True, subset_only=True, masked='full')
#val_dataset = UserDataset(data_name='food', load_full=True, subset_only=True, masked='partial') # todo: uncomment it later
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0) # todo: uncomment it later
numUsers = train_dataset.numIDs
numItems = train_dataset.numItems
print("train numUsers {}".format(numUsers))
print("train numItems {}".format(numItems))
# print("val numUsers {}".format(val_dataset.numIDs))
# print("val numItems {}".format(val_dataset.numItems))
user_embedding_dim = 128
noise_size = 128

train_user_ar(user_train_dataloader=train_loader, user_test_data_loader=None,
               num_users=numUsers, user_embedding_dim=user_embedding_dim, noise_size=noise_size, num_items=numItems,
               review_embedding_size=128, use_reviews=True, output_path='/mnt/nfs/scratch1/neerajsharma/model_params/small_dataset_results')

# train_user_ar(user_train_dataloader=train_loader, user_test_data_loader=val_loader,
#               num_users=numUsers, user_embedding_dim=user_embedding_dim, noise_size=noise_size, num_items=numItems,
#               review_embedding_size=128, use_reviews=False)