import torch
import random
from torch.utils.data import DataLoader
import os
import numpy as np
from src.models.base_models import Generator, Discriminator
from src.preprocessing.dataloader import UserDataset

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# todo: small experiment settings currently. Update it to full data usage later.
train_dataset = UserDataset(data_name='food', load_full=True, subset_only=True, masked='full')
# val_dataset = UserDataset(data_name='food', load_full=True, subset_only=True, masked='partial') # todo: uncomment it later
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0) # todo: uncomment it later
num_users = train_dataset.numIDs
num_items = train_dataset.numItems
print("train numUsers {}".format(num_users))
print("train numItems {}".format(num_items))
user_embedding_dim = 128
noise_size = 128
review_embedding_size = 128
use_reviews = True
best_epoch = 51

user_rating_generator = Generator(num_inputs=num_users, input_size=noise_size,
                                  item_count=num_items,
                                  c_embedding_size=user_embedding_dim,
                                  review_embedding_size=review_embedding_size,
                                  use_reviews=use_reviews).to(device)
user_missing_generator = Generator(num_inputs=num_users, input_size=noise_size, item_count=num_items,
                                   c_embedding_size=user_embedding_dim,
                                   review_embedding_size=review_embedding_size,
                                   use_reviews=use_reviews).to(device)
user_rating_discriminator = Discriminator(num_inputs=num_users, input_size=num_items,
                                          c_embedding_size=user_embedding_dim,
                                          rating_dense_representation_size=review_embedding_size,
                                          review_embedding_size=review_embedding_size,
                                          use_reviews=use_reviews).to(device)
user_missing_discriminator = Discriminator(num_inputs=num_users, input_size=num_items,
                                           c_embedding_size=user_embedding_dim,
                                           rating_dense_representation_size=review_embedding_size,
                                           review_embedding_size=review_embedding_size,
                                           use_reviews=use_reviews).to(device)


model_params_path = "/mnt/nfs/scratch1/neerajsharma/model_params/small_dataset_results"

user_rating_generator.load_state_dict(torch.load(os.path.join(model_params_path, "users_rating_generator_epoch_{}.pt".format(best_epoch))))
user_missing_generator.load_state_dict(torch.load(os.path.join(model_params_path, "users_missing_generator_epoch_{}.pt".format(best_epoch))))

def generate_neighbor(index, review_embeddings):
    noise_vector = torch.tensor(np.random.normal(0, 1, noise_size).reshape(1, noise_size),
                                dtype=torch.float32).to(device)
    index = index.type(torch.long).to(device)
    review_embeddings = review_embeddings.to(device)
    rating_vector = user_rating_generator(noise_vector, index, review_embeddings)
    missing_vector = user_missing_generator(noise_vector, index, review_embeddings)
    return rating_vector*missing_vector


count = 0

while count < 2:
    indexes = random.sample(num_users, 2)
    for index in indexes:
        user_reviews_embedding, user_ratings, idx = train_dataset.__getitem__(index)
        print("user_reviews_embedding type {} and shape {}".format(type(user_reviews_embedding), user_reviews_embedding.shape))
        print("user ratings type {} and shape {}".format(type(user_ratings), user_ratings.shape))
        print("idx type and shape {}".format(type(idx), idx.shape))
    count += 1



# while num_added_users / num_original_users < .40:
#     selected_user = sample real user(using alpha)
#     generated_user = generator(selected_user, noise)
#     data.append(generated_user)
#     num_added_users += 1
