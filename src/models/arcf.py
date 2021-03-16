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
          train_dataloader, test_dataloader, epochs, g_step, d_step, is_user=True, use_reviews=False):
    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        for step in g_step:
            g_loss = Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
            for data in train_dataloader:
                real_rating_vector = get_rating_vector(data)
                real_missing_vector = get_missing_vector(data)
                conditional_vector = get_conditional_vector(data)
                noise_vector = get_noise_vector(data)
                if use_reviews:
                    reviews = get_all_reviews(data, is_user=is_user)
                else:
                    reviews = None

                fake_rating_vector = rating_generator(noise_vector, conditional_vector, reviews)

                fake_missing_vector = missing_generator(noise_vector, conditional_vector, reviews)

                fake_rating_vector_with_missing = fake_rating_vector * real_missing_vector
                fake_rating_results = rating_discriminator(fake_rating_vector_with_missing, conditional_vector,
                                                           reviews)
                fake_missing_results = missing_discriminator(fake_missing_vector, conditional_vector, reviews)
                g_loss = g_loss.detach().numpy() + (torch.log(1. - fake_rating_results.detach().numpy()) +
                                                    torch.log(1. - fake_missing_results.detach().numpy()))
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

                fake_rating_vector = rating_generator(noise_vector, conditional_vector, reviews)

                fake_missing_vector = missing_generator(noise_vector, conditional_vector, reviews)

                fake_rating_vector_with_missing = fake_rating_vector * real_missing_vector
                fake_rating_results = rating_discriminator(fake_rating_vector_with_missing, conditional_vector,
                                                           reviews)
                real_rating_results = rating_discriminator(real_rating_vector, conditional_vector, reviews)
                fake_missing_results = missing_discriminator(fake_missing_vector, conditional_vector, reviews)
                real_missing_results = missing_discriminator(real_missing_vector, conditional_vector, reviews)
                d_loss = d_loss.detach.numpy() - (
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


def train_user_ar(user_dataloader, num_users, user_embedding_dim,
                  noise_size, num_items, review_embedding_size=128,
                  use_reviews=False):
    user_embedding_obj = UserEncoder(num_users, user_embedding_dim)
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
          user_dataloader, num_epochs, g_step, d_step, is_user=True)


def train_item_ar(item_dataloader, num_users, item_embedding_dim,
                  noise_size, num_items, review_embedding_size=128,
                  use_reviews=False):
    item_embedding_obj = ItemEncoder(num_items, item_embedding_dim)
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
          item_dataloader, num_epochs, g_step, d_step, is_user=False)


'''
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
'''
