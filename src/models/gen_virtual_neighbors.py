import torch
import os
import numpy as np
import random
from src.models.base_models import Generator, Discriminator

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''''
# todo: small experiment settings currently. Update it to full data usage later.
train_dataset = UserDataset(data_name='food', mode='train')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
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

user_rating_generator.load_state_dict(
    torch.load(os.path.join(model_params_path, "users_rating_generator_epoch_{}.pt".format(best_epoch))))
user_missing_generator.load_state_dict(
    torch.load(os.path.join(model_params_path, "users_missing_generator_epoch_{}.pt".format(best_epoch))))

index_arr = [i for i in range(len(num_users))]
weights = [1 / len(torch.nonzero(train_dataset.__getitem__(i)[1])) for i in range(num_users)]


def get_sample(sample_size):
    return random.choices(index_arr, weights, k=sample_size)



generate_users_count = 10000

sample_batch_size = 100

count = 0

generated_neighbors = []

while count < generate_users_count:
    indexes = get_sample(sample_size=sample_batch_size)
    count += len(indexes)
    for index in indexes:
        user_reviews_embedding, user_ratings, idx = train_dataset.__getitem__(index)
        user_reviews_embedding = torch.unsqueeze(user_reviews_embedding, 0)
        user_ratings = torch.unsqueeze(user_ratings, 0)
        idx = torch.unsqueeze(idx, 0)
        neighbor = generate_neighbor(idx, user_reviews_embedding)
        print("neighbor type {} shape {}".format(type(neighbor), neighbor.shape))
        generated_neighbors.append(neighbor.squeeze(0).cpu().detach().numpy())

'''


def generate_neighbor(rating_generator, missing_generator, index, review_embeddings, noise_size=128):
    noise_vector = torch.tensor(np.random.normal(0, 1, noise_size).reshape(1, noise_size),
                                dtype=torch.float32).to(device)
    index = index.type(torch.long).to(device)
    review_embeddings = review_embeddings.float().to(device)
    rating_vector = rating_generator(noise_vector, index, review_embeddings)
    missing_vector = missing_generator(noise_vector, index, review_embeddings)
    return rating_vector, missing_vector


def generate_virtual_users(dataset, num_users, num_items, model_params_path, total_neighbors, per_user_neighbors,
                           best_epoch, neighbors_path, missing_threshold=0.8):
    user_embedding_dim = 128
    noise_size = 128
    review_embedding_size = 128
    use_reviews = True
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
    user_rating_generator.load_state_dict(
        torch.load(os.path.join(model_params_path, "users_rating_generator_epoch_{}.pt".format(best_epoch))))
    user_missing_generator.load_state_dict(
        torch.load(os.path.join(model_params_path, "users_missing_generator_epoch_{}.pt".format(best_epoch))))
    user_rating_generator.eval()
    user_missing_generator.eval()
    index_arr = [i for i in range(num_users)]
    weights = [1 / len(torch.nonzero(dataset.__getitem__(i)[1])) for i in range(num_users)]
    indexes = random.choices(index_arr, weights, k=total_neighbors // per_user_neighbors)
    all_generated_neighbors = np.empty((0, num_items), np.float)
    for index in indexes:
        user_reviews_embedding, user_ratings, idx = dataset.__getitem__(index)
        user_reviews_embedding = torch.unsqueeze(user_reviews_embedding, 0)
        idx = torch.unsqueeze(idx, 0)
        for j in range(per_user_neighbors):
            neighbor_rating, neighbor_missing = generate_neighbor(user_rating_generator, user_missing_generator, idx,
                                                                  user_reviews_embedding)
            neighbor_missing = torch.tensor((neighbor_missing >= missing_threshold) * 1).float().cpu()
            # todo: add check for generated ratings and missing vector using discriminator
            neighbor = torch.ceil(neighbor_rating.cpu() * neighbor_missing * 5)
            print("neighbor type {} shape {}".format(type(neighbor), neighbor.shape))
            all_generated_neighbors = np.append(all_generated_neighbors, neighbor.cpu().detach().numpy(), axis=0)
    np.save(neighbors_path, all_generated_neighbors)


def generate_virtual_items(dataset, num_users, num_items, model_params_path, total_neighbors, per_user_neighbors,
                           best_epoch, neighbors_path, missing_threshold=0.8):
    item_embedding_dim = 128
    noise_size = 128
    review_embedding_size = 128
    use_reviews = True
    item_rating_generator = Generator(num_inputs=num_items, input_size=noise_size,
                                      item_count=num_users,
                                      c_embedding_size=item_embedding_dim,
                                      review_embedding_size=review_embedding_size, use_reviews=use_reviews).to(
        device)
    item_missing_generator = Generator(num_inputs=num_items, input_size=noise_size, item_count=num_users,
                                       c_embedding_size=item_embedding_dim,
                                       review_embedding_size=review_embedding_size, use_reviews=use_reviews).to(
        device)
    item_rating_discriminator = Discriminator(num_inputs=num_items, input_size=num_users,
                                              c_embedding_size=item_embedding_dim,
                                              review_embedding_size=review_embedding_size,
                                              rating_dense_representation_size=review_embedding_size,
                                              use_reviews=use_reviews).to(device)
    item_missing_discriminator = Discriminator(num_inputs=num_items, input_size=num_users,
                                               c_embedding_size=item_embedding_dim,
                                               rating_dense_representation_size=review_embedding_size,
                                               review_embedding_size=review_embedding_size,
                                               use_reviews=use_reviews).to(device)
    item_rating_generator.load_state_dict(
        torch.load(os.path.join(model_params_path, "items_rating_generator_epoch_{}.pt".format(best_epoch))))
    item_missing_generator.load_state_dict(
        torch.load(os.path.join(model_params_path, "items_missing_generator_epoch_{}.pt".format(best_epoch))))
    item_rating_generator.eval()
    item_missing_generator.eval()
    index_arr = [i for i in range(num_items)]
    weights = [1 / len(torch.nonzero(dataset.__getitem__(i)[1])) for i in range(num_items)]
    indexes = random.choices(index_arr, weights, k=total_neighbors // per_user_neighbors)
    all_generated_neighbors = np.empty((0, num_users), np.float)
    for index in indexes:
        item_reviews_embedding, item_ratings, idx = dataset.__getitem__(index)
        item_reviews_embedding = torch.unsqueeze(item_reviews_embedding, 0)
        idx = torch.unsqueeze(idx, 0)
        for j in range(per_user_neighbors):
            neighbor_rating, neighbor_missing = generate_neighbor(item_rating_generator, item_missing_generator, idx,
                                                                  item_reviews_embedding)
            # todo: add check for generated ratings and missing vector using discriminator
            neighbor_missing = torch.tensor((neighbor_missing >= missing_threshold) * 1).float().cpu()
            neighbor = neighbor_rating.cpu() * neighbor_missing
            print("neighbor type {} shape {}".format(type(neighbor), neighbor.shape))
            all_generated_neighbors = np.append(all_generated_neighbors, neighbor.cpu().detach().numpy(), axis=0)

    np.save(neighbors_path, all_generated_neighbors)
