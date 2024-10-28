from typing import Optional

from sentence_transformers import SentenceTransformer

import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(
            self,
            output_dim: int,
            dropout: float,
            embedding_model: SentenceTransformer,
            ff_layer_sizes: Optional[list[int]] = None
    ):
        if ff_layer_sizes is None:
            # Set two default fully connected layer sizes
            ff_layer_sizes = [600, 400]

        super(CustomModel, self).__init__()
        self.embedding_layer = embedding_model

        # Set the right input size for the first fully connected layer
        first_input_size: int = self.embedding_layer.get_sentence_embedding_dimension()

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

        # Initialize the hidden state with the embedded tensor
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
