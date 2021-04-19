from torch.utils.data import DataLoader
import torch
from src.preprocessing.dataloader import ItemDataset, UserDataset
from src.models.arcf_with_embedding import train_item_ar, train_user_ar
import random

import argparse

parser = argparse.ArgumentParser()

# arguments shared by the retriever and reader

parser.add_argument("--dataset_dir", default="/mnt/nfs/scratch1/neerajsharma/amazon_data/",
                    type=str, required=False,
                    help="dataset directory")
parser.add_argument("--hdf_file_path", default='new_5_dataset.h5',
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--user", default=True, type=bool, required=True,
                    help="whether to train for users or items if True train for users else for items")
parser.add_argument("--output_path", default='/mnt/nfs/scratch1/neerajsharma/model_params/small_dataset_results',
                    type=str, required=False,
                    help="Directory path to store model params")
parser.add_argument("--wandb_project_name", default='adversarial-recommendation-with-embedding-small-experiments',
                    type=str, required=False,
                    help="wandb project name")

args, unknown = parser.parse_known_args()

if args.user:
    print("Training for Users")
    training_dataset = UserDataset(data_name='food', mode='train')
else:
    print("Training for Items")
    training_dataset = ItemDataset(data_name='food', mode='train')

train_loader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=16)

numUsers = training_dataset.numIDs
numItems = training_dataset.numItems
print("train numUsers {}".format(numUsers))
print("train numItems {}".format(numItems))
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

print("Dataset path dir {} and hdf file name {}".format(args.dataset_dir, args.hdf_file_path))

print("wandb project name {}".format(args.wandb_project_name))

print("Model params file path {}".format(args.output_path))

user_or_item_embedding_dim = 128
noise_size = 128
review_embedding_dim = 128

'''

if args.user:
    train_user_ar(user_train_dataloader=train_loader, user_test_data_loader=None,
                  num_users=numUsers, user_embedding_dim=user_or_item_embedding_dim, noise_size=noise_size, num_items=numItems,
                  review_embedding_size=review_embedding_dim, use_reviews=True,
                  output_path=args.output_dir,
                  wandb_project_name=args.wandb_project_name)
else:
    train_item_ar(item_train_dataloader=train_loader, item_test_dataloader=None,
                  num_users=numUsers, item_embedding_dim=user_or_item_embedding_dim,
                  noise_size=noise_size,
                  num_items=numItems,
                  review_embedding_size=review_embedding_dim,
                  use_reviews=True,
                  output_path=args.output_dir,
                  wandb_project_name=args.wandb_project_name)
                  
'''
