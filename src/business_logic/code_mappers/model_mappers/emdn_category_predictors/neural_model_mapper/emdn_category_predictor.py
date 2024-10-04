import os
from pickle import UnpicklingError

import numpy as np

from nltk import word_tokenize, download, FreqDist
from src.__file_paths import TEST_FILE_PATH, TRAIN_FILE_PATH
from transformers import TextDataset
from sentence_transformers import SentenceTransformer

from sklearn.base import BaseEstimator, TransformerMixin
from torch import Tensor, no_grad, stack, save, load, long, softmax
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import max
from torch import tensor

from typing import Optional, Any
from torch.utils.data import WeightedRandomSampler

from src.business_logic.code_mappers.model_mappers.emdn_category_predictors.abstract_emdn_category_predictor import \
    AbstractEmdnCategoryPredictor, \
    output_test_results
from src.business_logic.code_mappers.model_mappers.emdn_category_predictors.neural_model_mapper.custom_network import \
    CustomModel
from src.business_logic.code_mappers.model_mappers.emdn_category_predictors.neural_model_mapper.sentence_transformer_dataset import \
    SentenceTransformerDataset
from src.business_logic.utilities.os_utilities import convert_to_device
from src.business_logic.code_mappers.model_mappers.clean_code_text import clean_text
from src.__directory_paths import MODELS_DIRECTORY_PATH


def build_weighted_sampler(
        _labels: np.ndarray[int]
) -> WeightedRandomSampler:
    """
    This method builds a WeightedRandomSampler based on the given labels. It calculates the class weights and returns the sampler. This is used to address the class imbalance problem.

    :param _labels: The labels to be used to calculate the class weights.

    :return: The WeightedRandomSampler built based on the given labels.
    """
    # Get every label with the corresponding count
    class_sample_count_tuple = np.unique(
        _labels,
        return_counts=True
    )
    # Get the counts of each class
    class_count = class_sample_count_tuple[1]
    # Calculate the weight of each class
    samples_weight = 1. / class_count

    return WeightedRandomSampler(
        samples_weight,
        len(samples_weight)
    )


def build_vocab_from_iterator(
        iterator,
        specials=None
) -> dict[Any, int]:
    # Create a frequency distribution from the iterator
    freq_dist = FreqDist(token for tokens in iterator for token in tokens)

    # Create a vocabulary dictionary
    vocab = {token: idx for idx, (token, _) in enumerate(freq_dist.items(), start=len(specials) if specials else 0)}

    # Add special tokens if provided
    if specials:
        for idx, token in enumerate(specials):
            vocab[token] = idx

    return vocab


class EmdnCategoryPredictor(
    AbstractEmdnCategoryPredictor,
    BaseEstimator,
    TransformerMixin
):
    """
    EmdnCategoryPredictor is a class that implements a neural model for the EMDN category prediction task.
    It is based on a custom neural network that can be trained on the provided data. The model can be built with a
    SentenceTransformer model or with an LSTM embedding layer. The model can be trained and tested on the provided
    data paths, and it can be used to predict the categories of new text strings.
    """

    def __init__(
            self,
            max_words: Optional[int] = 30000,
            max_input_length: Optional[int] = 2000,
            lstm_is_bidirectional: Optional[bool] = True,
            lstm_layers: Optional[int] = 2,
            lstm_layer_dropout: Optional[float] = 0.1,
            lstm_hidden_dimension: Optional[int] = 32,
            lstm_embedding_dim: Optional[int] = None,
            embedding_model: Optional[SentenceTransformer] = None,
            dropout: Optional[float] = 0.1,
            epochs: Optional[int] = 5,
            batch_size: Optional[int] = 32,
            validation_proportion: Optional[float] = None,
            model_path: str = None,
            output_directory_path: str = MODELS_DIRECTORY_PATH,
            train_file_path: str = TRAIN_FILE_PATH,
            test_file_path: str = TEST_FILE_PATH,
            ff_layer_sizes: Optional[list[int]] = None,
    ):
        if lstm_embedding_dim is None and embedding_model is None:
            raise ValueError("Either an embedding dimension or a SentenceTransformer model must be provided.")
        elif lstm_embedding_dim is not None and embedding_model is not None:
            raise ValueError("Only one of the embedding dimension or the SentenceTransformer model must be provided.")

        super().__init__(
            output_directory_path=output_directory_path,
            validation_proportion=validation_proportion,
            train_file_path=train_file_path,
            test_file_path=test_file_path,
        )
        download('punkt', quiet=True)
        self.tokenizer = word_tokenize
        # This is the vocabulary that will be built from the training data
        self.vocab = None
        self.max_words = max_words
        self.max_input_length = max_input_length
        self.n_classes = len(self.labels_dictionary)
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = convert_to_device(
            CustomModel(
                vocab_size=max_words + 2,
                lstm_is_bidirectional=lstm_is_bidirectional,
                lstm_layers=lstm_layers,
                lstm_layer_dropout=lstm_layer_dropout,
                lstm_hidden_dim=lstm_hidden_dimension,
                lstm_embed_dim=lstm_embedding_dim,
                output_dim=self.n_classes,
                dropout=dropout,
                embedding_model=embedding_model,
                ff_layer_sizes=ff_layer_sizes
            ),
            device_name='cuda'
        )

        self.model_path = model_path
        # Load the last best model state
        if self.model_path is not None and os.path.exists(self.model_path):
            self.load_pretrained()

        self._encode_data(
            use_dimensionality_reduction=False
        )

    def _encode_data(
            self,
            use_dimensionality_reduction: bool = False
    ) -> None:
        # Clean all the strings in the train and test data
        self.train_data = [
            [clean_text(_string, perform_classical_cleaning=False)]
            for _string in self.train_data
        ]
        self.test_data = [
            [clean_text(_string, perform_classical_cleaning=False)]
            for _string in self.test_data
        ]
        # If validation must be performed
        if self.validation_proportion is not None:
            # Clean all the strings in the validation data
            self.validation_data = [
                [clean_text(_string, perform_classical_cleaning=False)]
                for _string in self.validation_data
            ]

        # Build the vocabulary from the training data,
        # if the embedding model is not a SentenceTransformer model
        if type(self.model.embedding_layer) is not SentenceTransformer:
            def yield_tokens(
                    data_iter: list[str]
            ):
                # Iterate over all the text strings of the iterator
                for text in data_iter:
                    # And singularly yield the tokens of the text
                    yield self.tokenizer(text)

            # Call the method to build the vocabulary from the iterator
            self.vocab = build_vocab_from_iterator(
                yield_tokens([_list[0] for _list in self.train_data]),
                specials=["<unk>"]
            )

    def get_text_dataset(
            self,
            _texts,
            _labels
    ) -> SentenceTransformerDataset | TextDataset:
        # If the model is based on the embedding layer of the SentenceTransformer model
        if type(self.model.embedding_layer) is SentenceTransformer:
            # Return a dataset built with the SentenceTransformerDataset class
            return SentenceTransformerDataset(
                _texts=_texts,
                _labels=_labels,
                _transformer_model=self.model.embedding_layer
            )
        else:
            # Return a dataset built with the TextDataset class
            return TextDataset(
                _texts=_texts,
                _labels=_labels,
                _tokenizer=self.tokenizer,
                _vocab=self.vocab,
                _max_length=self.max_input_length
            )

    def fit(
            self,
            **kwargs,
    ) -> None:
        # Keep track of the last best validation loss
        best_val_loss = float('inf')

        # Number of epochs to wait for improvement before stopping
        patience = 10 if kwargs.get('es_patience') is None else kwargs.get('es_patience')
        # Learning rate for the Adam optimizer
        adam_lr = 0.001 if kwargs.get('adam_lr') is None else kwargs.get('adam_lr')
        # Build the resampler if necessary to balance the minority classes
        sampler = build_weighted_sampler(self.train_labels) if kwargs.get('resample') is not None else None

        validation_dataset: TextDataset | SentenceTransformerDataset | None = None
        # If the validation must be performed
        if self.validation_proportion is not None:
            # Create the validation dataset
            validation_dataset = self.get_text_dataset(
                self.validation_data,
                self.validation_labels
            )

        # Create the training dataset
        train_dataset = self.get_text_dataset(
            self.train_data,
            self.train_labels
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True if sampler is None else False,
            sampler=sampler
        )

        # Define loss function and optimizer
        criterion = CrossEntropyLoss()
        optimizer = Adam(
            lr=adam_lr,
            betas=(0.9, 0.999),
            eps=1e-07,
            weight_decay=0,
            amsgrad=False,
            params=self.model.parameters()
        )

        scheduler = None
        # If a scheduler builder is provided, create the scheduler
        # to update the parameters during the validation phase
        if kwargs.get('scheduler_builder') is not None:
            scheduler = kwargs.get('scheduler_builder')(optimizer)

        # Bring the model to the GPU
        self.model = convert_to_device(
            self.model,
            'cuda'
        )
        # Keep track of the number of epochs without improvement over the validation loss
        epochs_no_improve = 0
        # For each epoch that must be performed
        for epoch in range(self.epochs):
            # Set model to training mode
            self.model.train()
            total_loss = 0
            # For each batch in the training data
            for _texts, _labels in train_loader:
                # Bring the texts and labels to the GPU
                _texts, _labels = convert_to_device(_texts, 'cuda'), convert_to_device(_labels, 'cuda')
                # Reset all the gradients to zero before the new forward pass
                optimizer.zero_grad()
                # Compute the forward pass by passing the tensor as input
                outputs = self.model(_texts)
                # Compute the loss by comparing the outputs with the labels
                loss = criterion(outputs, _labels)
                # Compute the gradient of the loss with respect to the model parameters
                loss.backward()
                # Update the model's parameters based on the gradients
                # to try to reduce the current loss
                optimizer.step()
                # Accumulate the loss over the entire training set
                total_loss += loss.item()

            # Compute the average loss over the training set
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}')

            # If the validation data are provided
            if self.validation_proportion is not None:
                # Evaluate the model on the validation data and compute the average loss and accuracy
                avg_loss, accuracy = self._perform_validation(
                    validation_dataset,
                    scheduler
                )

                # If the average loss is lower than the best validation loss
                if avg_loss < best_val_loss:
                    # Update the best validation loss and
                    # reset the counter for no improvement
                    best_val_loss = avg_loss
                    epochs_no_improve = 0
                    # Save the best model state if necessary
                    if self.model_path is not None:
                        save(
                            self.model.state_dict(),
                            self.model_path
                        )
                else:
                    # Update the counter for no improvement
                    epochs_no_improve += 1

                # If early stopping is triggered, exit the loop
                if epochs_no_improve == patience:
                    break

        # Load the last best model state and test the model
        if self.model_path is not None:
            self.load_pretrained()
        self.perform_test()

    def _perform_validation(
            self,
            validation_dataset: TextDataset | SentenceTransformerDataset,
            scheduler: Optional[ReduceLROnPlateau] = None
    ) -> tuple[float, float]:
        # Create the data loader for the validation dataset, with no need to shuffle
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Set model to evaluation mode to avoid updating weights
        self.model.eval()

        # Disable gradient computation
        with no_grad():
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            # Set the criterion to compute the loss over the multi-class
            # classification problem
            criterion = CrossEntropyLoss()

            # For each batch loaded by the DataLoader
            for texts, label_indexes in validation_loader:
                texts, label_indexes = convert_to_device(texts, 'cuda'), convert_to_device(label_indexes, 'cuda')
                # Compute the forward pass given the text strings
                outputs = self.model(texts)
                # Compute the loss over the batch of the validation set
                loss = criterion(outputs, label_indexes)
                # Get the current loss as float and add it to the total loss
                total_loss += loss.item()

                # For each validation sample, get the predicted label
                _, predicted_label_indexes = max(
                    outputs,
                    # Get the maximum value for each row
                    1
                )
                # For each validation sample, check if the predicted label is equal to the true label
                # and compute the total amount of correct predictions
                for predicted_label_index, label_index in zip(predicted_label_indexes, label_indexes):
                    if predicted_label_index == label_index:
                        correct_predictions += 1
                # Update the total number of predictions of the validation set
                total_predictions += label_indexes.size(0)

            # Compute the average loss and accuracy of the validation dataset
            avg_loss = total_loss / len(validation_loader)
            accuracy = correct_predictions / total_predictions
            print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

            if scheduler is not None:
                # Adjust the learning rate based on the validation loss and the initial
                # given parameters (patience, factor)
                scheduler.step(avg_loss)

            # Return the average loss and accuracy for early stopping checks
            return avg_loss, accuracy

    def perform_test(
            self
    ) -> None:
        # Set the model to evaluation mode
        self.model.eval()
        # Bring the model to the GPU
        self.model = convert_to_device(
            self.model,
            'cuda'
        )

        # Define loss function
        criterion = CrossEntropyLoss()
        # Build the test dataset
        _test_dataset = self.get_text_dataset(
            self.test_data,
            self.test_labels
        )
        # Create a DataLoader for the test dataset, with no need
        # to shuffle the data
        test_loader = DataLoader(
            _test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        test_loss = 0
        correct_predictions = 0
        total_predictions = 0
        # Disable gradient computation
        with no_grad():
            total_true_labels = []
            total_predicted_labels = []
            # For each batch of strings and labels in the test dataset
            for _texts, _labels in test_loader:
                _texts, _labels = convert_to_device(_texts, 'cuda'), convert_to_device(_labels, 'cuda')
                # Compute the forward pass given the text strings
                outputs = self.model(_texts)
                # Compute the loss over the batch of the test set
                loss = criterion(outputs, _labels)
                # Add it to the total loss
                test_loss += loss.item()

                # Find the predicted labels and compute the total number of correct predictions
                _, predicted_labels = max(outputs, dim=1)
                for predicted_label, label in zip(predicted_labels, _labels):
                    total_true_labels.append(label.item())
                    total_predicted_labels.append(predicted_label.item())
                    if predicted_label.item() == label.item():
                        correct_predictions += 1
                # Update the counter of all the predictions
                total_predictions += _labels.size(0)

            output_test_results(
                test_labels=total_true_labels,
                predictions=total_predicted_labels,
                class_labels=list(self.labels_dictionary.keys()),
                display_labels=list(self.labels_dictionary.values()),
                output_folder=self.output_folder,
            )

        # Compute the average loss and accuracy for the test dataset
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct_predictions / total_predictions

        # Set the model to the CPU
        self.model = convert_to_device(self.model, 'cpu')

        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    def load_pretrained(
            self
    ):
        if self.model_path is None:
            raise ValueError("The model path must be specified to load a pre-trained model.")
        else:
            try:
                # Load the state dictionary
                state_dict = load(self.model_path)

                # Load the state dictionary into the model
                self.model.load_state_dict(state_dict)

                self.model = convert_to_device(
                    self.model,
                    'cuda'
                )

                # Set the model to evaluation mode
                self.model.eval()
            except (
                    FileNotFoundError,
                    RuntimeError,
                    UnpicklingError
            ):
                raise Exception(f"Failed to load the model.")

    def predict(
            self,
            input_texts: list[str] | str
    ):
        # Set the model in cuda if available
        self.model = convert_to_device(
            self.model,
            'cuda'
        )

        # Check if input_texts is a single string and wrap it in a list if true
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # Preprocess the input texts
        preprocessed_texts = EmdnCategoryPredictor._preprocess(input_texts)

        if type(self.model.embedding_layer) is SentenceTransformer:
            # Convert texts to sequences of integers
            tensor_sequences: list[Tensor] = [
                self.model.embedding_layer.encode(
                    text,
                    convert_to_tensor=True,
                    device='cuda'
                )
                for text in preprocessed_texts
            ]
            _tensor: Tensor = stack(tensor_sequences)
            # Add the batch dimension in the first position
            _tensor = _tensor.unsqueeze(0)
        else:
            # Convert texts to sequences of integers
            sequences = [self.vocab(self.tokenizer(text)) for text in preprocessed_texts]

            # Pad sequences to have the same length
            max_length = self.max_input_length
            padded_sequences: list[list[int]] = [
                seq + [0] * (max_length - len(seq))
                if len(seq) < max_length
                else seq[:max_length]
                for seq in sequences
            ]

            # Build a tensor from the padded sequences
            # that contains integers and move it to the GPU
            _tensor = convert_to_device(
                tensor(padded_sequences, dtype=long),
                'cuda'
            )

        returned_results: list[dict[str, float]] = []
        # Set the environment not to compute gradients
        with no_grad():
            # Set the model to evaluation mode
            self.model.eval()
            _tensor = _tensor.squeeze(0)
            for tensor_sequence in _tensor:
                tensor_sequence = tensor_sequence.unsqueeze(0).unsqueeze(0)
                # Get predictions from the model (un-normalized scores of the last layer)
                logits = self.model(tensor_sequence)
                # Transform the logits into probabilities
                probability_matrix = softmax(logits, dim=1)

                for probability_row in probability_matrix:
                    # Translate the predicted classes into the original labels
                    predicted_classes: dict[str, float] = {
                        self.labels_dictionary[_index]: predicted_class.item()
                        for _index, predicted_class in enumerate(probability_row)
                    }
                    returned_results.append(predicted_classes)

        return returned_results

    @staticmethod
    def _preprocess(
            texts
    ):
        return [
            clean_text(
                text,
                perform_classical_cleaning=False
            )
            for text in texts
        ]

    def print_summary(
            self
    ):
        self.model.print_summary()
