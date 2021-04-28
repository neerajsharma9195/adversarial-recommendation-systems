import torch
from src.preprocessing.dataloader import ItemDataset, UserDataset
from src.models.gen_virtual_neighbors import generate_virtual_users, generate_virtual_items
import os
import random

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir", default="/mnt/nfs/scratch1/neerajsharma/amazon_data/",
                    type=str, required=False,
                    help="dataset directory")
parser.add_argument("--hdf_file_path", default='new_5_dataset.h5',
                    type=str, required=False,
                    help="hdfs file path for data")
parser.add_argument("--interaction", default="users", type=str, required=True,
                    help="whether to use model")
parser.add_argument("--model_params_path", default='/mnt/nfs/scratch1/neerajsharma/model_params/small_dataset_results',
                    type=str, required=False,
                    help="Directory path to pick model file params")
parser.add_argument("--per_user_neighbors", default=10, type=int, required=False,
                    help="number of neighbors generated per users")
parser.add_argument("--total_neighbors", default=10000, type=int, required=False,
                    help="total neighbors need to be generated")
parser.add_argument("--epoch", type=int, required=True,
                    help="epoch need to be used to generate neighbors")
parser.add_argument("--neighbors_path", default='/mnt/nfs/scratch1/neerajsharma/model_params/small_dataset_results/neighbors.npy',
                    type=str, required=False,
                    help="Saved path of neighbors")
parser.add_argument("--use_reviews", default="yes", type=str, required=False,
                    help="use reviews or not for neighbors generation")



args, unknown = parser.parse_known_args()

print("Dataset path {}".format(os.path.join(args.dataset_dir, args.hdf_file_path)))

if args.interaction == 'users':
    print("Generating Users")
    training_dataset = UserDataset(data_name='food', mode='train',
                                   path=os.path.join(args.dataset_dir, args.hdf_file_path))
elif args.interaction == 'items':
    print("Generating Items")
    training_dataset = ItemDataset(data_name='food', mode='train',
                                   path=os.path.join(args.dataset_dir, args.hdf_file_path))
else:
    print("***** ERROR: WRONG ARGS *******")
    exit(0)


numUsers = training_dataset.numIDs
numItems = training_dataset.numItems
print("train numUsers {}".format(numUsers))
print("train numItems {}".format(numItems))

model_params_path = args.model_params_path
total_neighbors = args.total_neighbors
per_user_neighbors = args.per_user_neighbors
best_epoch = args.epoch
neighbors_path = args.neighbors_path
use_reviews = True if args.use_reviews == 'yes' else False

print("model params path {}".format(model_params_path))
print("total neighbors need to be generated {}".format(total_neighbors))
print("per user neighbors need to be generated {}".format(per_user_neighbors))
print("best epoch {}".format(best_epoch))
print("neighbors path {}".format(neighbors_path))

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if args.interaction == 'users':
    print("Generating Users")
    generate_virtual_users(dataset=training_dataset, num_users=numUsers, num_items=numItems, model_params_path=model_params_path, total_neighbors=total_neighbors, per_user_neighbors=per_user_neighbors,
                           best_epoch=best_epoch, neighbors_path=neighbors_path, use_reviews=use_reviews)
elif args.interaction == 'items':
    print("Generating Items")
    generate_virtual_items(dataset=training_dataset, num_users=numUsers, num_items=numItems, model_params_path=model_params_path, total_neighbors=total_neighbors, per_user_neighbors=per_user_neighbors,
                           best_epoch=best_epoch, neighbors_path=neighbors_path, use_reviews=use_reviews)
else:
    print("***** ERROR: WRONG ARGS *******")
    exit(0)


