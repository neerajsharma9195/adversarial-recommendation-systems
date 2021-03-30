import torch.nn as nn
import torch.nn.functional as F
import torch


class UserEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, 128)

        self.first_layer = nn.Linear(128, 256)

        self.output_layer = nn.Linear(256, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, user_one_hot_vector):
        # src = [src len, batch size]
        embedded = self.embedding(user_one_hot_vector)
        first_output = nn.functional.relu(self.first_layer(embedded))

        final_output = self.dropout(nn.functional.relu(self.output_layer(first_output)))
        return final_output # shape=(input_dim, output_dim)


class ItemEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, 128)

        self.first_layer = nn.Linear(128, 256)

        self.output_layer = nn.Linear(256, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, item_one_hot_vector):
        # src = [src len, batch size]
        embedded = self.embedding(item_one_hot_vector)
        first_output = nn.functional.relu(self.first_layer(embedded))

        final_output = self.dropout(nn.functional.relu(self.output_layer(first_output)))
        return final_output


class RatingDenseRepresentation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.firstLayer = nn.Linear(input_dim, 2048)
        self.secondLayer = nn.Linear(2048, 1024)
        self.thirdLayer = nn.Linear(1024, output_dim)

    def forward(self, rating_vector):
        dense_representation = self.firstLayer(rating_vector)
        dense_representation = nn.functional.relu(dense_representation)
        dense_representation = self.secondLayer(dense_representation)
        dense_representation = nn.functional.relu(dense_representation)
        return self.thirdLayer(dense_representation)


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

    def forward(self, rating_vector, embedding_vector, review_embedding):
        if self.use_reviews:
            data_c = torch.cat((rating_vector, embedding_vector, review_embedding), dim=1)
        else:
            data_c = torch.cat((rating_vector, embedding_vector), dim=1)
        result = self.dis(data_c)
        return result


class Generator(nn.Module):
    def __init__(self, input_size, item_count, c_embedding_size, review_embedding_size, use_reviews=True):
        self.input_size = input_size
        self.output_size = item_count
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
            nn.Linear(1024, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, noise_vector, embedding_vector, review_embedding):
        if self.use_reviews:
            G_input = torch.cat((noise_vector, embedding_vector, review_embedding), dim=1)
        else:
            G_input = torch.cat((noise_vector, embedding_vector), dim=1)
        result = self.gen(G_input)
        return result
