import numpy as np
from torch.autograd import Variable
from src.models.base_models import UserEncoder, ItemEncoder, Generator, Discriminator, RatingDenseRepresentation
import torch.nn as nn
import os
import torch

import wandb

# 1. Start a new run
# todo: change name from small experiments to complete experiments
wandb.init(project="adversarial-recommendation-with-embedding-small-experiments")

# 2. Save model inputs and hyperparameters
config = wandb.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def RMSELoss(generated_rating_vector, true_rating_vector):
    return torch.sqrt(torch.mean((generated_rating_vector - true_rating_vector) ** 2))


def train(rating_generator, missing_generator, rating_discriminator,
          missing_discriminator, rating_g_optimizer, missing_g_optimizer,
          rating_d_optimizer, missing_d_optimizer,
          train_dataloader, test_dataloader, epochs, g_step, d_step, num_users, num_items, noise_size,
          embedding_size=128,
          is_user=True, use_reviews=False, output_path="/mnt/nfs/scratch1/rbialik/model_params/arcf_embeddings/"):
    rating_generator.train()
    missing_generator.train()
    rating_discriminator.train()
    missing_discriminator.train()
    best_performance = 0
    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        rmse_rating_loss = 0
        for step in range(g_step):
            g_loss = Variable(torch.tensor(0, dtype=torch.float32, device=device), requires_grad=True)
            for i, batch in enumerate(train_dataloader):
                review_embedding, rating_vector, index_item = batch
                rating_vector = rating_vector.type(torch.float32).to(device)
                review_embedding = review_embedding.type(torch.float32).to(device)
                real_missing_vector = torch.tensor((rating_vector > 0) * 1).to(device)
                index_item = index_item.type(torch.long).to(device)
                noise_vector = torch.randn(1, noise_size, dtype=torch.float32, device=device)
                if not use_reviews:
                    review_embedding = None
                fake_rating_vector = rating_generator(noise_vector, index_item, review_embedding)

                fake_missing_vector = missing_generator(noise_vector, index_item, review_embedding)

                fake_rating_vector_with_missing = fake_rating_vector * real_missing_vector
                fake_rating_results = rating_discriminator(fake_rating_vector_with_missing, index_item,
                                                           review_embedding)
                fake_missing_results = missing_discriminator(fake_missing_vector, index_item, review_embedding)
                g_loss = g_loss + torch.log(1. - fake_rating_results) + torch.log(1. - fake_missing_results)
                rmse_rating_loss += RMSELoss(fake_rating_vector_with_missing, rating_vector)
                if not is_user:
                    if i % 1000 == 0:
                        print("epoch {} g step {} processed {}".format(epoch, step, i))
                if i % 10000 == 0:
                    print("epoch {} g step {} processed {}".format(epoch, step, i))

                if i % 100 == 0:
                    g_loss = torch.mean(g_loss)
                    rating_g_optimizer.zero_grad()
                    missing_g_optimizer.zero_grad()
                    epoch_g_loss += g_loss.data
                    g_loss.backward(retain_graph=True)
                    rating_g_optimizer.step()
                    missing_g_optimizer.step()
                    g_loss = Variable(torch.tensor(0, dtype=torch.float32, device=device), requires_grad=True)
            g_loss = torch.mean(g_loss)
            rating_g_optimizer.zero_grad()
            missing_g_optimizer.zero_grad()
            epoch_g_loss += g_loss.data
            g_loss.backward(retain_graph=True)
            rating_g_optimizer.step()
            missing_g_optimizer.step()

        for step in range(d_step):
            d_loss = Variable(torch.tensor(0, dtype=torch.float32, device=device), requires_grad=True)
            for i, batch in enumerate(train_dataloader):
                review_embedding, real_rating_vector, index_item = batch
                real_rating_vector = real_rating_vector.type(torch.float32).to(device)
                review_embedding = review_embedding.type(torch.float32).to(device)
                real_missing_vector = torch.tensor((real_rating_vector > 0) * 1).type(torch.float32).to(device)
                index_item = index_item.type(torch.long).to(device)
                noise_vector = torch.randn(1, noise_size, dtype=torch.float32, device=device)
                if not use_reviews:
                    review_embedding = None
                fake_rating_vector = rating_generator(noise_vector, index_item, review_embedding)

                fake_missing_vector = missing_generator(noise_vector, index_item, review_embedding)
                fake_rating_vector_with_missing = fake_rating_vector * real_missing_vector

                fake_rating_results = rating_discriminator(fake_rating_vector_with_missing, index_item,
                                                           review_embedding)
                real_rating_results = rating_discriminator(real_rating_vector, index_item,
                                                           review_embedding)
                fake_missing_results = missing_discriminator(fake_missing_vector, index_item,
                                                             review_embedding)
                real_missing_results = missing_discriminator(real_missing_vector, index_item,
                                                             review_embedding)
                d_loss = d_loss - (torch.log(real_rating_results) + torch.log(real_missing_results) +
                                   torch.log(1. - fake_rating_results) + torch.log(1. - fake_missing_results))

                if not is_user:
                    if i % 1000 == 0:
                        print("epoch {} d step {} processed {}".format(epoch, step, i))

                if i % 10000 == 0:
                    print("epoch {} d step {} processed {}".format(epoch, step, i))

                if i % 100 == 0:
                    d_loss = torch.mean(d_loss)
                    rating_d_optimizer.zero_grad()
                    missing_d_optimizer.zero_grad()
                    epoch_d_loss += d_loss.data
                    d_loss.backward(retain_graph=True)
                    rating_d_optimizer.step()
                    missing_d_optimizer.step()
                    d_loss = Variable(torch.tensor(0, dtype=torch.float32, device=device), requires_grad=True)
            d_loss = torch.mean(d_loss)
            rating_d_optimizer.zero_grad()
            missing_d_optimizer.zero_grad()
            epoch_d_loss += d_loss.data
            d_loss.backward(retain_graph=True)
            rating_d_optimizer.step()
            missing_d_optimizer.step()
        if is_user:
            wandb.log({
                'epoch': epoch,
                'user_generator_loss': epoch_g_loss,
                'user_discriminator_loss': epoch_d_loss,
                'rmse_generator_loss': rmse_rating_loss
            })
        else:
            wandb.log({
                'epoch': epoch,
                'item_generator_loss': epoch_g_loss,
                'item_discriminator_loss': epoch_d_loss
            })
        path_name = "users" if is_user else "items"

        torch.save(rating_generator.state_dict(),
                   os.path.join(output_path, "{}_rating_generator_epoch_{}.pt".format(path_name, epoch)))
        torch.save(missing_generator.state_dict(),
                   os.path.join(output_path, "{}_missing_generator_epoch_{}.pt".format(path_name, epoch)))
        torch.save(rating_discriminator.state_dict(),
                   os.path.join(output_path, "{}_rating_discriminator_epoch_{}.pt".format(path_name, epoch)))
        torch.save(missing_discriminator.state_dict(),
                   os.path.join(output_path, "{}_missing_discriminator_epoch_{}.pt".format(path_name, epoch)))

        if epoch % 20 == 0:
            performance = evaluate_cf(test_dataloader, rating_generator, missing_generator)
            wandb.log({
                "epoch": epoch,
                "cf_performance": performance  # todo: update CF performance
            })
            if performance > best_performance:
                best_performance = performance
                torch.save(rating_generator.state_dict(),
                           os.path.join(output_path, "{}_rating_generator_best.pt".format(path_name)))
                torch.save(missing_generator.state_dict(),
                           os.path.join(output_path, "{}_missing_generator_best.pt".format(path_name)))
                torch.save(rating_discriminator.state_dict(),
                           os.path.join(output_path, "{}_rating_discriminator_best.pt".format(path_name)))
                torch.save(missing_discriminator.state_dict(),
                           os.path.join(output_path, "{}_missing_discriminator_best.pt".format(path_name)))
            rating_generator.train()  # back to training mode
            missing_generator.train()


def evaluate_cf(test_dataloader, rating_generator, missing_generator):
    missing_generator.eval()
    rating_generator.eval()
    # mask test data
    # print(len(test_dataloader.dataset))
    # for i, batch in enumerate(test_dataloader):
    #     review_embedding, rating_vector, index_item = batch
    #     print('rating vec = ', rating_vector)
    #     print([(i, rating_vector[i]) for i in rating_vector if rating_vector[i] != 0])
    # generate augmented users & items
    # make CF matrix
    # compare predictions with masks
    # calculate & return:
    #   precision, recall, nDCG(?) MRR(?)
    # precision = true positives / true positives and false positives
    # recall = true positives / true positives and false negatives
    # MAE = 1/N sumN |x_i - x|
    return 0


def train_user_ar(user_train_dataloader, user_test_data_loader, num_users, user_embedding_dim,
                  noise_size, num_items, review_embedding_size=128,
                  use_reviews=False, output_path='/mnt/nfs/scratch1/neerajsharma/model_params/small_dataset_results'):
    user_rating_generator = Generator(num_inputs=num_users, input_size=noise_size, item_count=num_items,
                                      c_embedding_size=user_embedding_dim,
                                      review_embedding_size=review_embedding_size, use_reviews=use_reviews).to(
        device)
    user_missing_generator = Generator(num_inputs=num_users, input_size=noise_size, item_count=num_items,
                                       c_embedding_size=user_embedding_dim,
                                       review_embedding_size=review_embedding_size, use_reviews=use_reviews).to(
        device)
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
    wandb.watch(user_rating_generator)
    wandb.watch(user_missing_generator)
    wandb.watch(user_rating_discriminator)
    wandb.watch(user_missing_discriminator)
    g_step = 1
    d_step = 1
    num_epochs = 200
    user_rating_g_optimizer = torch.optim.Adam(user_rating_generator.parameters(), lr=0.0001, weight_decay=0.001)
    user_rating_d_optimizer = torch.optim.Adam(user_rating_discriminator.parameters(), lr=0.0001, weight_decay=0.001)
    user_missing_g_optimizer = torch.optim.Adam(user_missing_generator.parameters(), lr=0.0001, weight_decay=0.001)
    user_missing_d_optimizer = torch.optim.Adam(user_missing_discriminator.parameters(), lr=0.0001, weight_decay=0.001)
    # todo: currently running experiments for a small dataset
    train(rating_generator=user_rating_generator, missing_generator=user_missing_generator,
          rating_discriminator=user_rating_discriminator, missing_discriminator=user_missing_discriminator,
          rating_g_optimizer=user_rating_g_optimizer, missing_g_optimizer=user_missing_g_optimizer,
          rating_d_optimizer=user_rating_d_optimizer, missing_d_optimizer=user_missing_d_optimizer,
          train_dataloader=user_train_dataloader, test_dataloader=user_test_data_loader,
          epochs=num_epochs, g_step=g_step, d_step=d_step, num_users=num_users, num_items=num_items,
          noise_size=noise_size, is_user=True, use_reviews=use_reviews,
          output_path=output_path)


def train_item_ar(item_train_dataloader, item_test_dataloader, num_users, item_embedding_dim,
                  noise_size, num_items, review_embedding_size=128,
                  use_reviews=False):
    if use_reviews:
        item_rating_generator = Generator(input_size=noise_size, item_count=num_users,
                                          c_embedding_size=item_embedding_dim,
                                          review_embedding_size=review_embedding_size, use_reviews=use_reviews).to(
            device)
        item_missing_generator = Generator(input_size=noise_size, item_count=num_users,
                                           c_embedding_size=item_embedding_dim,
                                           review_embedding_size=review_embedding_size, use_reviews=use_reviews).to(
            device)
        item_rating_discriminator = Discriminator(input_size=review_embedding_size, c_embedding_size=item_embedding_dim,
                                                  review_embedding_size=review_embedding_size,
                                                  use_reviews=use_reviews).to(device)
        item_missing_discriminator = Discriminator(input_size=review_embedding_size,
                                                   c_embedding_size=item_embedding_dim,
                                                   review_embedding_size=review_embedding_size,
                                                   use_reviews=use_reviews).to(device)
    else:
        item_rating_generator = Generator(input_size=noise_size, item_count=num_users,
                                          c_embedding_size=item_embedding_dim,
                                          review_embedding_size=review_embedding_size, use_reviews=use_reviews).to(
            device)
        item_missing_generator = Generator(input_size=noise_size, item_count=num_users,
                                           c_embedding_size=item_embedding_dim,
                                           review_embedding_size=review_embedding_size, use_reviews=use_reviews).to(
            device)
        item_rating_discriminator = Discriminator(input_size=review_embedding_size, c_embedding_size=item_embedding_dim,
                                                  review_embedding_size=review_embedding_size,
                                                  use_reviews=use_reviews).to(device)
        item_missing_discriminator = Discriminator(input_size=review_embedding_size,
                                                   c_embedding_size=item_embedding_dim,
                                                   review_embedding_size=review_embedding_size,
                                                   use_reviews=use_reviews).to(device)

    wandb.watch(item_rating_generator)
    wandb.watch(item_missing_generator)
    wandb.watch(item_rating_discriminator)
    wandb.watch(item_missing_discriminator)
    g_step = 5
    d_step = 2
    num_epochs = 100
    item_rating_g_optimizer = torch.optim.Adam(item_rating_generator.parameters(), lr=0.0001, weight_decay=0.001)
    item_rating_d_optimizer = torch.optim.Adam(item_rating_discriminator.parameters(), lr=0.0001, weight_decay=0.001)
    item_missing_g_optimizer = torch.optim.Adam(item_missing_generator.parameters(), lr=0.0001, weight_decay=0.001)
    item_missing_d_optimizer = torch.optim.Adam(item_missing_discriminator.parameters(), lr=0.0001, weight_decay=0.001)

    train(rating_generator=item_rating_generator, missing_generator=item_missing_generator,
          rating_discriminator=item_rating_discriminator, missing_discriminator=item_missing_discriminator,
          rating_g_optimizer=item_rating_g_optimizer, missing_g_optimizer=item_missing_g_optimizer,
          rating_d_optimizer=item_rating_d_optimizer, missing_d_optimizer=item_missing_d_optimizer,
          train_dataloader=item_train_dataloader, test_dataloader=item_test_dataloader, epochs=num_epochs,
          g_step=g_step, d_step=d_step, num_items=num_items, num_users=num_users, noise_size=noise_size, is_user=False,
          use_reviews=use_reviews)
