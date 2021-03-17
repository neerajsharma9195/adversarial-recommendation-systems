import numpy as np
from torch.autograd import Variable
from src.preprocessing.utils import get_all_reviews_of_user, get_conditional_vector, get_missing_vector, \
    get_rating_vector, get_noise_vector
from src.models.base_models import UserEncoder, ItemEncoder, Generator, Discriminator

import torch

import wandb

# 1. Start a new run
wandb.init(project="adversarial-recommendation")

# 2. Save model inputs and hyperparameters
config = wandb.config


def train(rating_generator, missing_generator, rating_discriminator,
          missing_discriminator, rating_g_optimizer, missing_g_optimizer,
          rating_d_optimizer, missing_d_optimizer,
          train_dataloader, test_dataloader, epochs, g_step, d_step, num_users, num_items, embedding_size=128,
          is_user=True, use_reviews=False):
    if is_user:
        embedding = UserEncoder(num_users, embedding_size)
    else:
        embedding = ItemEncoder(num_items, embedding_size)
    embedding.requires_grad_(True)
    rating_generator.train()
    missing_generator.train()
    rating_discriminator.train()
    missing_discriminator.train()
    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0

        for step in g_step:
            g_loss = Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
            for data in train_dataloader:
                real_missing_vector = get_missing_vector(data)
                conditional_vector = get_conditional_vector(data)
                noise_vector = get_noise_vector(data)
                if use_reviews:
                    reviews = get_all_reviews(data, is_user=is_user)
                else:
                    reviews = None

                embedding_representation = embedding(conditional_vector)
                fake_rating_vector = rating_generator(noise_vector, embedding_representation, reviews)

                fake_missing_vector = missing_generator(noise_vector, embedding_representation, reviews)

                fake_rating_vector_with_missing = fake_rating_vector * real_missing_vector
                fake_rating_results = rating_discriminator(fake_rating_vector_with_missing, embedding_representation,
                                                           reviews)
                fake_missing_results = missing_discriminator(fake_missing_vector, embedding_representation, reviews)
                g_loss = g_loss.detach().numpy() + (np.log(1. - fake_rating_results.detach().numpy()) +
                                                    np.log(1. - fake_missing_results.detach().numpy()))
                g_loss = Variable(g_loss, requires_grad=True)
            g_loss = torch.mean(g_loss)
            rating_g_optimizer.zero_grad()
            missing_g_optimizer.zero_grad()
            epoch_g_loss += g_loss.data
            g_loss.backward(retain_graph=True)
            rating_g_optimizer.step()
            missing_g_optimizer.step()

        for step in range(d_step):
            d_loss = Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
            for data in train_dataloader:
                real_rating_vector = get_rating_vector(data)
                real_missing_vector = get_missing_vector(data)
                conditional_vector = get_conditional_vector(data)
                noise_vector = get_noise_vector(data)
                if use_reviews:
                    reviews = get_all_reviews(data, is_user=is_user)
                else:
                    reviews = None

                embedding_representation = embedding(conditional_vector)
                fake_rating_vector = rating_generator(noise_vector, embedding_representation, reviews)

                fake_missing_vector = missing_generator(noise_vector, embedding_representation, reviews)

                fake_rating_vector_with_missing = fake_rating_vector * real_missing_vector
                fake_rating_results = rating_discriminator(fake_rating_vector_with_missing, embedding_representation,
                                                           reviews)
                real_rating_results = rating_discriminator(real_rating_vector, embedding_representation, reviews)
                fake_missing_results = missing_discriminator(fake_missing_vector, embedding_representation, reviews)
                real_missing_results = missing_discriminator(real_missing_vector, embedding_representation, reviews)
                d_loss = d_loss.detach().numpy() - (
                        np.log(real_rating_results.detach().numpy()) + np.log(real_missing_results.detach().numpy())
                        + np.log(1. - fake_rating_results.detach().numpy()) +
                        np.log(1. - fake_missing_results.detach().numpy()))
                d_loss = Variable(d_loss, requires_grad=True)
            d_loss = torch.mean(d_loss)
            rating_d_optimizer.zero_grad()
            missing_d_optimizer.zero_grad()
            epoch_d_loss += d_loss.data
            d_loss.backward(retain_graph=True)
            rating_d_optimizer.zero_grad()
            missing_d_optimizer.step()

        embedding.zero_grad()

        if is_user:
            wandb.log({
                'epoch': epoch,
                'user_generator_loss': epoch_g_loss,
                'user_discriminator_loss': epoch_d_loss
            })
        else:
            wandb.log({
                'epoch': epoch,
                'item_generator_loss': epoch_g_loss,
                'item_discriminator_loss': epoch_d_loss
            })

        if epoch % 20 == 0:
            evaluate_cf(test_dataloader, rating_generator, missing_generator)
            wandb.log({
                "epoch": epoch,
                "cf_performance": 0 # todo: update CF performance
            })


def evaluate_cf(test_data, rating_generator, missing_generator):
    missing_generator.eval()
    rating_generator.eval()
    # todo: fill this function
    pass


def train_user_ar(user_dataloader, num_users, user_embedding_dim,
                  noise_size, num_items, review_embedding_size=128,
                  use_reviews=False):
    if use_reviews:
        user_rating_generator = Generator(input_size=noise_size, item_count=num_items,
                                          c_embedding_size=user_embedding_dim,
                                          review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        user_missing_generator = Generator(input_size=noise_size, item_count=num_items,
                                           c_embedding_size=user_embedding_dim,
                                           review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        user_rating_discriminator = Discriminator(input_size=num_items, c_embedding_size=user_embedding_dim,
                                                  review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        user_missing_discriminator = Discriminator(input_size=num_items, c_embedding_size=user_embedding_dim,
                                                   review_embedding_size=review_embedding_size, use_reviews=use_reviews)
    else:
        user_rating_generator = Generator(input_size=noise_size, item_count=num_items,
                                          c_embedding_size=user_embedding_dim,
                                          review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        user_missing_generator = Generator(input_size=noise_size, item_count=num_items,
                                           c_embedding_size=user_embedding_dim,
                                           review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        user_rating_discriminator = Discriminator(input_size=num_items, c_embedding_size=user_embedding_dim,
                                                  review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        user_missing_discriminator = Discriminator(input_size=num_items, c_embedding_size=user_embedding_dim,
                                                   review_embedding_size=review_embedding_size, use_reviews=use_reviews)

    wandb.watch(user_rating_generator)
    wandb.watch(user_missing_generator)
    wandb.watch(user_rating_discriminator)
    wandb.watch(user_missing_discriminator)
    g_step = 5
    d_step = 2
    num_epochs = 100
    user_rating_g_optimizer = torch.optim.Adam(user_rating_generator.parameters(), lr=0.0001)
    user_rating_d_optimizer = torch.optim.Adam(user_rating_discriminator.parameters(), lr=0.0001)
    user_missing_g_optimizer = torch.optim.Adam(user_missing_generator.parameters(), lr=0.0001)
    user_missing_d_optimizer = torch.optim.Adam(user_missing_discriminator.parameters(), lr=0.0001)

    train(user_rating_generator, user_missing_generator, user_rating_discriminator, user_missing_discriminator,
          user_rating_g_optimizer, user_missing_g_optimizer,
          user_rating_d_optimizer, user_missing_d_optimizer,
          user_dataloader, num_epochs, g_step, d_step, num_users, num_items, is_user=True)


def train_item_ar(item_dataloader, num_users, item_embedding_dim,
                  noise_size, num_items, review_embedding_size=128,
                  use_reviews=False):
    if use_reviews:
        item_rating_generator = Generator(input_size=noise_size, item_count=num_users,
                                          c_embedding_size=item_embedding_dim,
                                          review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        item_missing_generator = Generator(input_size=noise_size, item_count=num_users,
                                           c_embedding_size=item_embedding_dim,
                                           review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        item_rating_discriminator = Discriminator(input_size=num_users, c_embedding_size=item_embedding_dim,
                                                  review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        item_missing_discriminator = Discriminator(input_size=num_users, c_embedding_size=item_embedding_dim,
                                                   review_embedding_size=review_embedding_size, use_reviews=use_reviews)
    else:
        item_rating_generator = Generator(input_size=noise_size, item_count=num_users,
                                          c_embedding_size=item_embedding_dim,
                                          review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        item_missing_generator = Generator(input_size=noise_size, item_count=num_users,
                                           c_embedding_size=item_embedding_dim,
                                           review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        item_rating_discriminator = Discriminator(input_size=num_users, c_embedding_size=item_embedding_dim,
                                                  review_embedding_size=review_embedding_size, use_reviews=use_reviews)
        item_missing_discriminator = Discriminator(input_size=num_users, c_embedding_size=item_embedding_dim,
                                                   review_embedding_size=review_embedding_size, use_reviews=use_reviews)

    wandb.watch(item_rating_generator)
    wandb.watch(item_missing_generator)
    wandb.watch(item_rating_discriminator)
    wandb.watch(item_missing_discriminator)
    g_step = 5
    d_step = 2
    num_epochs = 100
    item_rating_g_optimizer = torch.optim.Adam(item_rating_generator.parameters(), lr=0.0001)
    item_rating_d_optimizer = torch.optim.Adam(item_rating_discriminator.parameters(), lr=0.0001)
    item_missing_g_optimizer = torch.optim.Adam(item_missing_generator.parameters(), lr=0.0001)
    item_missing_d_optimizer = torch.optim.Adam(item_missing_discriminator.parameters(), lr=0.0001)

    train(item_rating_generator, item_missing_generator, item_rating_discriminator, item_missing_discriminator,
          item_rating_g_optimizer, item_missing_g_optimizer,
          item_rating_d_optimizer, item_missing_d_optimizer,
          item_dataloader, num_epochs, g_step, d_step, num_users, num_items, is_user=False)
