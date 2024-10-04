from typing import Optional, Any

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import gridspec
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

from src.business_logic.utilities.os_utilities import create_directory
from src.__constants import RANDOM_SEED_INT, X_TICKS_SIZE, Y_TICKS_SIZE, X_LABEL_SIZE, Y_LABEL_SIZE
from src.__directory_paths import MODELS_DIRECTORY_PATH


def load_training_data_file(
        train_file_path: str,
        test_file_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load the train dataset containing all the mappings between EMDN categories
    # and text strings (either GMDN Term Names or EMDN descriptions)
    train_dataset = pd.read_csv(
        train_file_path
    )[['text', 'emdn_category']].drop_duplicates()
    # Load the test dataset containing all the mappings between EMDN categories,
    # EMDN codes, and GMDN Term Names, and keep only the GMDN Term Names and the EMDN categories
    selected_test_dataset = pd.read_csv(
        test_file_path
    )[['gmdn_term_name', 'emdn_category']].drop_duplicates()
    # Rename the 'gmdn_term_name' column to 'text'
    selected_test_dataset = selected_test_dataset.rename(
        columns={'gmdn_term_name': 'text'}
    )

    return train_dataset.to_numpy(), selected_test_dataset.to_numpy()


def output_test_results(
        test_labels: np.ndarray[int] | list[int],
        predictions: np.ndarray[int] | list[int],
        class_labels: list[int],
        display_labels: list[str] = None,
        output_folder: str = MODELS_DIRECTORY_PATH,
):
    _confusion_matrix = confusion_matrix(
        y_true=test_labels,
        y_pred=predictions,
        labels=class_labels,
    )

    plt.figure(figsize=(10, 10))
    gs: Any = gridspec.GridSpec(1, 2, width_ratios=[10, 1])

    # Matrix plot
    ax0 = plt.subplot(gs[0])

    # Sort the display labels alphabetically
    sorted_display_labels = sorted(display_labels)

    # Display the confusion matrix with sorted labels
    ConfusionMatrixDisplay(
        confusion_matrix=_confusion_matrix,
        display_labels=sorted_display_labels,
    ).plot(
        ax=ax0,
        cmap='Reds',
        colorbar=False,
        text_kw={'fontsize': 18}
    )

    plt.ylabel(
        'True Label',
        fontsize=22,
        labelpad=10
    )
    plt.xlabel(
        'Predicted Label',
        fontsize=22,
        labelpad=10
    )
    plt.yticks(
        fontsize=20
    )
    plt.xticks(
        fontsize=20,
        ha="right"
    )

    # Right bar (e.g., colorbar or another plot)
    ax1 = plt.subplot(gs[1])

    # Example: Plot a colorbar (replace with your actual right bar content)
    sm = plt.cm.ScalarMappable(
        cmap='Reds',
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    cbar = plt.colorbar(sm, cax=ax1)
    cbar.ax.yaxis.set_tick_params(labelsize=18)

    plt.tight_layout()
    # Adjust subplot parameters to reduce white space
    plt.subplots_adjust(top=0.8, bottom=0.05)

    ax0.grid(False)

    # Save the figure in the output directory
    plt.savefig(
        output_folder + 'prediction_confusion_matrix.png',
        bbox_inches='tight'
    )
    plt.close()


def output_dataset_distribution(
        _dataset: pd.DataFrame,
        file_name: str,
        int_to_medical_specialty: dict[int, str],
        output_folder: str = MODELS_DIRECTORY_PATH,
):
    # Store a bar plot of the labels
    plt.figure(figsize=(20, 10))
    bar_dataset = _dataset.groupby('label').size()
    # Add a column with the medical specialty and the code in parentheses
    new_index_values = []
    for _index in bar_dataset.index:
        index_value = f"{int_to_medical_specialty[int(_index)]}"
        new_index_values.append(index_value)

    # Replace the original index with the new list of index values
    bar_dataset.index = new_index_values
    bar_dataset.plot(kind='bar')

    plt.ylabel(
        'Number of Samples',
        fontsize=Y_LABEL_SIZE,
        labelpad=10
    )
    plt.xlabel(
        'EMDN Category',
        fontsize=X_LABEL_SIZE,
        labelpad=10
    )
    plt.xticks(
        fontsize=X_TICKS_SIZE
    )
    plt.yticks(
        fontsize=Y_TICKS_SIZE
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_folder + file_name)
    plt.close()


class AbstractEmdnCategoryPredictor:
    def __init__(
            self,
            train_file_path: str,
            test_file_path: str,
            validation_proportion: Optional[float] = None,
            output_directory_path: str = MODELS_DIRECTORY_PATH,
    ):
        self.output_folder = output_directory_path
        create_directory(self.output_folder)

        # Set the random seed
        self.random_seed = RANDOM_SEED_INT
        np.random.seed(self.random_seed)

        # Store the proportion of the training set that
        # must be used for validation
        self.validation_proportion = validation_proportion

        # Set all the numpy arrays for containing the datasets
        # and the labels
        self.train_data: np.ndarray[str] | None = None
        self.validation_data: np.ndarray[str] | None = None
        self.test_data: np.ndarray[str] | None = None
        self.train_labels: np.ndarray[str] | None = None
        self.validation_labels: np.ndarray[str] | None = None
        self.test_labels: np.ndarray[str] | None = None

        # This is a dictionary that contains the relationship
        # between the integer labels and the original ones
        self.labels_dictionary: dict[int, str] | None = None

        # Load the needed data to train the predictor
        self._load_corpus(
            train_file_path=train_file_path,
            test_file_path=test_file_path
        )

    def _load_corpus(
            self,
            train_file_path: str,
            test_file_path: str
    ) -> None:
        # Load the train and test datasets
        _train_dataset, _test_dataset = load_training_data_file(
            train_file_path,
            test_file_path
        )

        # Get all the train and test labels
        _train_labels = _train_dataset[:, -1]
        _test_labels = _test_dataset[:, -1]
        # Concatenate the train and test labels to get all the possible labels
        all_labels: list[str] = list(
            np.concatenate((_train_labels, _test_labels))
        )
        # Encode the labels
        all_labels_encoded = self._encode_labels(
            all_labels,
        )
        # Split the encoded labels back into train and test labels
        _train_dataset[:, -1] = all_labels_encoded[:len(_train_labels)]
        _test_dataset[:, -1] = all_labels_encoded[len(_train_labels):]

        # If validation must be performed
        if self.validation_proportion is not None:
            # Split the train set into train and validation sets, keeping the same
            # proportion of the labels
            total_train_data, total_validation_data = sklearn.model_selection.train_test_split(
                _train_dataset,
                test_size=self.validation_proportion,
                shuffle=True,
                random_state=self.random_seed,
                stratify=_train_dataset[:, -1]
            )
            self.validation_data = total_validation_data[:, :-1]
            self.validation_labels = total_validation_data[:, -1]
        else:
            total_train_data = _train_dataset

        # Set the train data and labels
        self.train_data = total_train_data[:, :-1]
        self.train_labels = total_train_data[:, -1]
        self.test_data = _test_dataset[:, :-1]
        self.test_labels = _test_dataset[:, -1]
        # Convert all the labels to int values
        self.train_labels = self.train_labels.astype(int)
        self.test_labels = self.test_labels.astype(int)

        # Store the image containing the distribution of the test dataset
        # across the different labels
        output_dataset_distribution(
            pd.DataFrame(
                self.test_labels,
                columns=['label'],
            ),
            int_to_medical_specialty=self.labels_dictionary,
            output_folder=self.output_folder,
            file_name='test_dataset_distribution.png'
        )
        # Store the image containing the distribution of the train dataset
        # across the different labels
        output_dataset_distribution(
            pd.DataFrame(
                _train_dataset[:, -1],
                columns=['label'],
            ),
            int_to_medical_specialty=self.labels_dictionary,
            output_folder=self.output_folder,
            file_name='train_dataset_distribution.png'
        )

        del _train_dataset
        del total_train_data
        del total_validation_data
        del _test_dataset
        del _train_labels
        del _test_labels
        del all_labels
        del all_labels_encoded

    def _encode_data(
            self,
            use_dimensionality_reduction: bool = False
    ) -> None:
        raise NotImplementedError("The method '_encode_data' must be implemented in the child class.")

    def _perform_dimensionality_reduction(
            self
    ) -> None:
        raise NotImplementedError(
            "The method '_perform_dimensionality_reduction' must be implemented in the child class.")

    def _encode_labels(
            self,
            _labels: list[str]
    ):
        # Existing initialization code
        self.label_encoder = LabelEncoder()
        # Fit and transform the labels to integers
        returned_labels = self.label_encoder.fit_transform(_labels)

        # Store the relationship between each integer label and the original one
        for original_label, returned_label in zip(_labels, returned_labels):
            if self.labels_dictionary is None:
                self.labels_dictionary = {}
            self.labels_dictionary[returned_label] = original_label

        return returned_labels

    def fit(
            self,
            **kwargs,
    ) -> tuple[Any, Any]:
        raise NotImplementedError("The method 'fit' must be implemented in the child class.")
