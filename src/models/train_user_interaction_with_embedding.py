import torch
from torch.utils.data import DataLoader
from src.preprocessing.dataloader import UserDataset
from src.models.arcf_with_embedding import train_user_ar
import random

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

user_dataset = UserDataset(data_name='food')
validation_uid, validation_iid, validation_vid = user_dataset.get_mask(drop_ratio=0.3)
training_uid, training_iid, training_vid = user_dataset.get_mask(drop_ratio=0.6, masked_uid=validation_uid, masked_iid=validation_iid)

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

train_loader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=16)
val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=16)
test_loader = DataLoader(user_dataset, batch_size=1, shuffle=True, num_workers=16)

# Here is an example of how you can use the dataloader
# for i, batch in enumerate(train_loader):
#     if i < 1:
#         user_reviews_embedding, user_ratings, idx = batch
#         print(f"user_reviews_embedding: {user_reviews_embedding.shape}")
#         print("user reviews embedding type {}".format(type(user_reviews_embedding)))
#         print(f"user_ratings: {user_ratings.shape}, # raings: {(user_ratings > 0).sum()}")
#         print("user ratings type {}".format(type(user_ratings)))
#         print("idx size {}".format(idx.shape))
#         print("idx type {}".format(type(idx)))
#     else:
#         break

'''
# todo: small experiment settings currently. Update it to full data usage later.
train_dataset = UserDataset(data_name='food', load_full=True, subset_only=True, masked='full')
#val_dataset = UserDataset(data_name='food', load_full=True, subset_only=True, masked='partial') # todo: uncomment it later
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0) # todo: uncomment it later

'''

numUsers = training_dataset.numIDs
numItems = training_dataset.numItems
print("train numUsers {}".format(numUsers))
print("train numItems {}".format(numItems))

user_embedding_dim = 128
noise_size = 128

train_user_ar(user_train_dataloader=train_loader, user_test_data_loader=val_loader,
              num_users=numUsers, user_embedding_dim=user_embedding_dim, noise_size=noise_size, num_items=numItems,
              review_embedding_size=128, use_reviews=True,
              output_path='/mnt/nfs/scratch1/neerajsharma/model_params/small_dataset_results')
