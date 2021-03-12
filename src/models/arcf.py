import argparse
import os
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
from src.preprocessing.utils import get_all_reviews_of_user, get_conditional_vector, get_missing_vector, \
    get_rating_vector, get_noise_vector

import torch.nn as nn
import torch.nn.functional as F
import torch

import wandb

# 1. Start a new run
wandb.init(project="adversarial-recommendation")

# 2. Save model inputs and hyperparameters
config = wandb.config


# rating discriminator for user inputs dimensions:
# rating vector: (total_number_of_items, 1)
# c_u : (total_number_of_users, 1)
# review_embedding: (final_embedding_size, 1)

class UserEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UserEmbedding, self).__init__()
        self.input_layer = nn.Embedding(input_dim, 1024)
        self.first_layer = nn.Linear(1024, 2048)
        self.output_layer = nn.Linear(2048, output_dim)

    def forward(self, user_one_hot_vector):
        user_embedding = self.input_layer(user_one_hot_vector)
        user_embedding = self.first_layer(user_embedding)
        user_embedding = nn.functional.relu(user_embedding)
        return self.output_layer(user_embedding)


class ReviewEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reviews):
        # define pre trained BERT
        pass


class Discriminator(nn.Module):
    def __init__(self, input_size, c_embedding_size, review_embedding_size, use_reviews=True):
        super(Discriminator, self).__init__()
        self.use_reviews = use_reviews
        if self.use_reviews:
            input_dim = input_size + c_embedding_size + review_embedding_size
        else:
            input_dim = input_size + c_embedding_size
        self.dis = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, rating_vector, c_vector, user_reviews=None):
        c_embedding = UserEmbedding(c_vector)
        if self.use_reviews and use_reviews != None:
            review_embedding = ReviewEmbedding(user_reviews)
            data_c = torch.cat((rating_vector, c_embedding, review_embedding), 1)
        else:
            data_c = torch.cat((rating_vector, c_embedding), 1)
        result = self.dis(data_c)
        return result


class Generator(nn.Module):
    def __init__(self, input_size, c_embedding_size, review_embedding_size, use_reviews=True):
        self.input_size = input_size
        self.use_reviews = use_reviews
        super(Generator, self).__init__()
        if use_reviews:
            input_dim = self.input_size + c_embedding_size + review_embedding_size
        else:
            input_dim = self.input_size + c_embedding_size
        self.gen = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, noise_vector, c_vector, user_reviews=None):
        c_embedding = UserEmbedding(c_vector)
        if use_reviews and use_reviews != None:
            review_embedding = ReviewEmbedding(user_reviews)
            G_input = torch.cat([noise_vector, c_embedding, review_embedding], 1)
        else:
            G_input = torch.cat([noise_vector, c_embedding], 1)
        result = self.gen(G_input)
        return result


class ItemEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ItemEmbedding, self).__init__()
        self.input_layer = nn.Embedding(input_dim, 1024)
        self.first_layer = nn.Linear(1024, 2048)
        self.output_layer = nn.Linear(2048, output_dim)

    def forward(self, item_one_hot_vector):
        item_embedding = self.input_layer(item_one_hot_vector)
        item_embedding = self.first_layer(item_embedding)
        item_embedding = nn.functional.relu(item_embedding)
        return self.output_layer(item_embedding)


use_reviews = True
use_meta_condition = False

# training
num_epochs = 100
# for user
c_embedding_size = 128
item_counts = 10000  # total number of items
review_embedding_size = 128
if use_reviews:
    user_rating_generator = Generator(input_size=item_counts, c_embedding_size=c_embedding_size,
                                      review_embedding_size=review_embedding_size, use_reviews=True)
    user_missing_generator = Generator(input_size=item_counts, c_embedding_size=c_embedding_size,
                                       review_embedding_size=review_embedding_size, use_reviews=True)
    user_rating_discriminator = Discriminator(input_size=item_counts, c_embedding_size=c_embedding_size,
                                              review_embedding_size=review_embedding_size, use_reviews=True)
    user_missing_discriminator = Discriminator(input_size=item_counts, c_embedding_size=c_embedding_size,
                                               review_embedding_size=review_embedding_size, use_reviews=True)
else:
    user_rating_generator = Generator(input_size=item_counts, c_embedding_size=c_embedding_size,
                                      review_embedding_size=review_embedding_size, use_reviews=False)
    user_missing_generator = Generator(input_size=item_counts, c_embedding_size=c_embedding_size,
                                       review_embedding_size=review_embedding_size, use_reviews=False)
    user_rating_discriminator = Discriminator(input_size=item_counts, c_embedding_size=c_embedding_size,
                                              review_embedding_size=review_embedding_size, use_reviews=False)
    user_missing_discriminator = Discriminator(input_size=item_counts, c_embedding_size=c_embedding_size,
                                               review_embedding_size=review_embedding_size, use_reviews=False)

wandb.watch(user_rating_generator)
wandb.watch(user_missing_generator)
wandb.watch(user_rating_discriminator)
wandb.watch(user_missing_discriminator)

g_step = 5
d_step = 2
batch_size_g = 32
batch_size_d = 32
user_rating_g_optimizer = torch.optim.Adam(user_rating_generator.parameters(), lr=0.0001)
user_rating_d_optimizer = torch.optim.Adam(user_rating_discriminator.parameters(), lr=0.0001)
user_missing_g_optimizer = torch.optim.Adam(user_missing_generator.parameters(), lr=0.0001)
user_missing_d_optimizer = torch.optim.Adam(user_missing_discriminator.parameters(), lr=0.0001)

# for items
c_embedding_size = 128
user_counts = 10000  # total number of users
review_embedding_size = 128
item_rating_generator = Generator(user_counts, c_embedding_size, review_embedding_size)
item_missing_generator = Generator(user_counts, c_embedding_size, review_embedding_size)
item_rating_discriminator = Discriminator(user_counts, c_embedding_size, review_embedding_size)
item_missing_discriminator = Discriminator(user_counts, c_embedding_size, review_embedding_size)
item_rating_g_optimizer = torch.optim.Adam(item_rating_generator.parameters(), lr=0.0001)
item_rating_d_optimizer = torch.optim.Adam(item_rating_discriminator.parameters(), lr=0.0001)
item_missing_g_optimizer = torch.optim.Adam(item_missing_generator.parameters(), lr=0.0001)
item_missing_d_optimizer = torch.optim.Adam(item_missing_discriminator.parameters(), lr=0.0001)

# train
all_users_batches = []

for epoch in range(num_epochs):
    for step in range(g_step):
        for user_batch in all_users_batches:
            g_loss = 0
            for user in user_batch:
                real_rating_vector = get_rating_vector(user)
                real_missing_vector = get_missing_vector(user)
                conditional_vector = get_conditional_vector(user)
                if use_reviews:
                    reviews = get_all_reviews_of_user(user)
                else:
                    reviews = None

                noise_vector = get_noise_vector()
                fake_rating_vector = user_rating_generator(noise_vector, conditional_vector, reviews)

                fake_missing_vector = user_missing_generator(noise_vector, conditional_vector, reviews)

                fake_rating_vector_with_missing = fake_rating_vector * real_missing_vector
                fake_rating_results = user_rating_discriminator(fake_rating_vector_with_missing, conditional_vector,
                                                                reviews)
                fake_missing_results = user_missing_discriminator(fake_missing_vector, conditional_vector, reviews)

                '''
                so for generator: if results are fake it should give values near to 0 and if its give values near to 1 then its not good so 
                penalize. 
                '''

                g_loss += (np.log(1. - fake_rating_results.detach().numpy()) + np.log(
                    1. - fake_missing_results.detach().numpy()))

            g_loss = np.mean(g_loss)
            user_rating_g_optimizer.zero_grad()
            user_missing_g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            user_rating_g_optimizer.step()
            user_missing_g_optimizer.step()

    for step in range(d_step):
        for user_batch in all_users_batches:
            d_loss = 0
            for user in user_batch:
                real_rating_vector = get_rating_vector(user)
                real_missing_vector = get_missing_vector(user)
                conditional_vector = get_conditional_vector(user)
                if use_reviews:
                    reviews = get_all_reviews_of_user(user)
                else:
                    reviews = None
                noise_vector = get_noise_vector()
                fake_rating_vector = user_rating_generator(noise_vector, conditional_vector, reviews)

                fake_missing_vector = user_missing_generator(noise_vector, conditional_vector, reviews)
                fake_rating_vector_with_missing = fake_rating_vector * real_missing_vector
                fake_rating_results = user_rating_discriminator(fake_rating_vector_with_missing, conditional_vector,
                                                                reviews)
                real_rating_results = user_rating_discriminator(real_rating_vector, conditional_vector, reviews)
                fake_missing_results = user_missing_discriminator(fake_missing_vector, conditional_vector, reviews)
                real_missing_results = user_missing_discriminator(real_missing_vector, conditional_vector, reviews)
                d_loss += -(np.log(real_rating_results) + np.log(real_missing_results)
                            + np.log(1. - fake_rating_results.detach().numpy()) +
                            np.log(1. - fake_missing_results.detach().numpy()))
            d_loss = np.mean(d_loss)
            user_rating_d_optimizer.zero_grad()
            user_missing_d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            user_rating_d_optimizer.zero_grad()
            user_missing_d_optimizer.step()

    # todo: run on test data set

    wandb.log({
        'epoch': epoch,
        'generator_loss': 0.00,  # todo: update it when get loss on test data
        'discriminator_loss': 0.123  # # todo: update it when get loss on test data
    })
