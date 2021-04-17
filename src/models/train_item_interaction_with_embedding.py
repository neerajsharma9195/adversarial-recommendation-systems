from torch.utils.data import DataLoader
import torch
from src.preprocessing.dataloader import ItemDataset
from src.models.arcf_with_embedding import train_item_ar
import random

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
item_training_dataset = ItemDataset(data_name='food', mode='train')

train_loader = DataLoader(item_training_dataset, batch_size=1, shuffle=True, num_workers=16)

numUsers = item_training_dataset.numIDs
numItems = item_training_dataset.numItems
item_embedding_dim = 128
noise_size = 128

print("train numUsers {}".format(numUsers))
print("train numItems {}".format(numItems))

train_item_ar(item_train_dataloader=train_loader, item_test_dataloader=None,
              num_users=numUsers, item_embedding_dim=item_embedding_dim,
              noise_size=noise_size,
              num_items=numItems,
              review_embedding_size=128,
              use_reviews=True,
              output_path='/mnt/nfs/scratch1/neerajsharma/model_params/complete_data_results')
