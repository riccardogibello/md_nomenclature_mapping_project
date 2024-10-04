from typing import Optional

from torch.nn import Embedding
from sentence_transformers import SentenceTransformer

import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            lstm_is_bidirectional: bool,
            lstm_layers: int,
            lstm_layer_dropout: float,
            lstm_hidden_dim: int,
            lstm_embed_dim: int,
            output_dim: int,
            dropout: float,
            embedding_model: Optional[SentenceTransformer] = None,
            ff_layer_sizes: Optional[list[int]] = None
    ):
        if ff_layer_sizes is None:
            # Set two default fully connected layer sizes
            ff_layer_sizes = [600, 400]

        super(CustomModel, self).__init__()

        if lstm_layers < 2:
            lstm_layer_dropout = 0.0

        self.lstm_layers = lstm_layers
        self.is_lstm_bidirectional = lstm_is_bidirectional

        if embedding_model is not None:
            self.embedding_layer = embedding_model
        else:
            # Build the embedding layer
            self.embedding_layer = Embedding(
                vocab_size,
                lstm_embed_dim
            )

        if self.lstm_layers >= 1:
            # Find the right LSTM input size, given that the embedding are built
            # with the SentenceTransformer model or not
            if type(self.embedding_layer) is SentenceTransformer:
                lstm_input_size = self.embedding_layer.get_sentence_embedding_dimension()
            else:
                lstm_input_size = lstm_embed_dim

            # Build the LSTM layer
            self.lstm = nn.LSTM(
                # Set the size of the input tensor,
                # which is the embedding dimension
                input_size=lstm_input_size,
                # Specify that the input tensor has the
                # batch size as the first dimension
                batch_first=True,
                # Specify the size of the hidden state
                # and cell state of the LSTM
                hidden_size=lstm_hidden_dim,
                # Specify the number of stacked LSTM layers
                num_layers=lstm_layers,
                bidirectional=self.is_lstm_bidirectional,
                # Add a dropout layer to avoid overfitting on the
                # output of each LSTM layer
                dropout=lstm_layer_dropout,
            )

        # Set the right input size for the first fully connected layer
        first_input_size: int
        if self.lstm_layers > 0:
            # Multiply by 2 if bidirectional LSTM
            first_input_size = lstm_hidden_dim * 2 if self.is_lstm_bidirectional else lstm_hidden_dim
        else:
            if type(self.embedding_layer) is not SentenceTransformer:
                first_input_size = lstm_embed_dim
            else:
                first_input_size = self.embedding_layer.get_sentence_embedding_dimension()

        # Create a number of fully connected layers equal to the given
        # list of sizes
        self.ff_layer_names = []
        for i, layer_size in enumerate(ff_layer_sizes):
            tmp_layer_name = f'ff_layer_{i}'
            self.ff_layer_names.append(tmp_layer_name)
            self.__setattr__(
                tmp_layer_name,
                nn.Linear(
                    in_features=first_input_size,
                    out_features=layer_size
                )
            )
            first_input_size = layer_size

        # Create the output layer
        self.fc = nn.Linear(
            in_features=first_input_size,
            out_features=output_dim
        )

        # Create a dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            text: torch.Tensor
    ):
        # If the embedding layer is a SentenceTransformer model,
        if type(self.embedding_layer) is SentenceTransformer:
            # Then the text has been already embedded
            with torch.no_grad():
                embedded = text
        else:
            # If the embedding layer is the default one, pass the text tensor
            # to convert it into an embedded tensor
            embedded = self.dropout(self.embedding_layer(text))

        # If the LSTM module must be used
        if self.lstm_layers > 0:
            # Pass the embedded tensor to the LSTM layer, which returns
            # all the hidden states and cell states for each token and
            # the last hidden and cell states
            lstm_out, (hidden, cell) = self.lstm(embedded)
            # If the LSTM is bidirectional, concatenate the last hidden states
            if self.is_lstm_bidirectional:
                # Take the last two hidden states and concatenate them
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            else:
                hidden = hidden[-1, :, :]
        else:
            # Otherwise, initialize the hidden state with the embedded tensor
            hidden = embedded

        # Pass the hidden state through the series of FC layers
        for layer_name in self.ff_layer_names:
            hidden = self.dropout(F.relu(self.__getattr__(layer_name)(hidden)))

        # Pass the hidden state to the output layer
        out = self.fc(hidden)

        # If the output is composed by a matrix in which the second dimension is 1,
        if out.size(1) == 1:
            # Squeeze the second dimension
            out = out.squeeze(1)

        # Return the output
        return out

    def print_summary(
            self,
            detailed: bool = False
    ):
        if not detailed:
            print(self)
        else:
            print(summary(self, (2000,)))
