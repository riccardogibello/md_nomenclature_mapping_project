import time

import pandas as pd
from sentence_transformers import SentenceTransformer
from src.__file_paths import TEST_FILE_PATH, TRAIN_FILE_PATH
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.business_logic.code_mappers.model_mappers.data_driven_emdn_code_predictor import DataDrivenEmdnCodePredictor
from src.business_logic.code_mappers.model_mappers.emdn_category_predictors.neural_model_mapper.emdn_category_predictor import \
    EmdnCategoryPredictor

from src.business_logic.code_mappers.model_mappers.emdn_code_predictor import EmdnCodePredictor
from src.business_logic.pipeline.model_training.datasets_handling import load_test_dataset
from src.business_logic.pipeline.model_training.output_utilities import compute_aggregated_statistics, \
    compute_index_level_heatmap, compute_models_comparison, output_test_distribution
from src.business_logic.utilities.os_utilities import create_directory
from src.data_model.nomenclature_codes.emdn_code import find_common_emdn_code
from src.__directory_paths import MODELS_DIRECTORY_PATH, \
    GMDN_EMDN_VALIDATION_DIRECTORY_PATH


def build_emdn_category_predictor(
        model_name: str,
        model_validation_folder_path: str,
        train_file_path: str,
        test_file_path: str
) -> EmdnCategoryPredictor:
    """
    This method creates the instance of the EMDN category predictor.

    :param model_name: The name of the pretrained language model to be used.
    :param model_validation_folder_path: The path to the folder where the model will be saved.
    :param train_file_path: The path to the training dataset.
    :param test_file_path: The path to the test dataset.

    :return: The instance of the EMDN category predictor.
    """
    return EmdnCategoryPredictor(
        lstm_layers=0,
        embedding_model=SentenceTransformer(model_name),
        dropout=0.1,
        epochs=50,
        batch_size=16,
        validation_proportion=0.1,
        output_directory_path=model_validation_folder_path,
        model_path=model_validation_folder_path + 'emdn_category_predictor.pth',
        train_file_path=train_file_path,
        test_file_path=test_file_path,
        ff_layer_sizes=[96],
    )


def _get_model_parameters(
        _model
) -> int:
    """
    This method returns the number of trainable parameters of a given model.

    :param _model: The model for which the number of trainable parameters is to be computed.

    :return: The number of trainable parameters of the given model.
    """
    pp = 0
    for p in list(_model.parameters()):
        _nn = 1
        for s in list(p.size()):
            _nn = _nn * s
        pp += _nn
    return pp


def validate_model(
        base_folder_path: str,
        _model: EmdnCodePredictor | DataDrivenEmdnCodePredictor,
        results_file_name: str,
        _test_dataset: pd.DataFrame,
) -> str:
    """
    This method validates the EMDN code predictor on the test dataset.

    :param base_folder_path: The path to the folder where the results will be saved.
    :param _model: The EMDN code predictor to be validated.
    :param results_file_name: The name of the file where the mapping results will be saved.
    :param _test_dataset: The test dataset containing the GMDN Term Names with related EMDN codes.

    :return: The path to the file where the results have been saved.
    """
    # Create the directory of the base folder path
    create_directory(base_folder_path)

    # Get the list of GMDN codes to be tested
    gmdn_term_names = _test_dataset['gmdn_term_name'].to_list()
    # Divide the GMDN term names into batches
    batch_size = 32
    n_batches = len(gmdn_term_names) // batch_size
    if len(gmdn_term_names) % batch_size != 0:
        n_batches += 1

    import time
    dataset_rows = _test_dataset.to_numpy()
    # For each batch
    _start_time = time.time()
    for batch_index in range(n_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(gmdn_term_names))
        # Get the GMDN term names of the batch
        batch_gmdn_term_names = gmdn_term_names[start_index:end_index]
        # Predict the EMDN codes for each of the GMDN term names
        # Predict the EMDN codes for each of the GMDN term names
        _predicted_emdn_probability: dict[str, list[tuple[str, float]]] = _model.predict(
            batch_gmdn_term_names
        )
        # For each starting GMDN term name
        dataset_index = 0

        # For each GMDN Term Name that has been predicted
        for _gmdn_term_name in batch_gmdn_term_names:
            current_dataset_index = start_index + dataset_index
            # Get the current EMDN code of the gold standard
            gold_standard_emdn_code = dataset_rows[current_dataset_index, 1]
            # Get the gold standard category
            gold_standard_category = gold_standard_emdn_code[0]
            # Get the ordered list of the most probable EMDN codes
            _predicted_emdn = _predicted_emdn_probability[_gmdn_term_name]
            tuple_index = 0
            for option_index, emdn_code_score_tuple in enumerate(_predicted_emdn):
                # Get the EMDN code
                _emdn_code = emdn_code_score_tuple[0]
                # Get its EMDN category
                _emdn_category = _emdn_code[0]
                # Get the tuple containing the EMDN code and probability
                # associated to the given EMDN category
                _score = emdn_code_score_tuple[1]
                # If the EMDN gold standard has the same category as the predicted one
                if _emdn_category == gold_standard_category:
                    # Compute the max level of concordance between the two codes
                    _, match_level = find_common_emdn_code(
                        str(gold_standard_emdn_code),
                        _emdn_code
                    )
                    # Set the value in the cell indicating the position of the right option
                    dataset_rows[current_dataset_index, -1] = option_index
                    # Set the level of concordance in the dataset
                    dataset_rows[current_dataset_index, -2] = match_level

                # Set the EMDN code and the probability in the dataset
                dataset_rows[current_dataset_index, tuple_index + 2] = _emdn_code
                dataset_rows[current_dataset_index, tuple_index + 3] = _score
                # Update the next index of the dataset to two positions ahead
                tuple_index += 2
            dataset_index += 1

    # Replace the dataset rows with the new ones
    _test_dataset = pd.DataFrame(
        dataset_rows,
        columns=_test_dataset.columns
    )

    # Save the results to a file
    file_path = base_folder_path + results_file_name + '.csv'
    _test_dataset.to_csv(
        file_path
    )

    return file_path


def train_emdn_category_predictor(
        _gmdn_emdn_test_dataframe: pd.DataFrame
) -> None:
    """
    This method trains the EMDN category predictors. Two tests are performed, by using a neural network on top
    of either a pre-trained BioBERT or MPNet model. Then, each EMDN category predictor is used to compare the
    baseline and data-driven models on the EMDN code prediction task.

    :param _gmdn_emdn_test_dataframe: The test dataset containing the GMDN Term Names with related EMDN codes.
    """
    ffnn_biobert_subfolder_path = 'ffnn/biobert/'
    ffnn_mpnet_subfolder_path = 'ffnn/mpnet/'
    # Instantiate the predictor
    model_folder_paths = [
        MODELS_DIRECTORY_PATH + ffnn_biobert_subfolder_path,
        MODELS_DIRECTORY_PATH + ffnn_mpnet_subfolder_path,
    ]
    model_names = [
        'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
        'all-mpnet-base-v2',
    ]
    model_subfolders = [
        ffnn_biobert_subfolder_path,
        ffnn_mpnet_subfolder_path,
    ]
    divider = '----------------------------------------'

    for model_validation_folder_path, model_name, model_subfolder in zip(
            model_folder_paths,
            model_names,
            model_subfolders
    ):
        # Train the EMDN category predictor
        _emdn_category_predictor = build_emdn_category_predictor(
            model_name=model_name,
            model_validation_folder_path=model_validation_folder_path,
            train_file_path=TRAIN_FILE_PATH,
            test_file_path=TEST_FILE_PATH
        )
        # Print a summary of the model
        print(divider)
        print(_emdn_category_predictor.model)
        print(f"The model has {_get_model_parameters(_emdn_category_predictor.model):,} trainable parameters.")
        print(divider)
        start_time = time.time()
        # Train the regressor
        _emdn_category_predictor.fit(
            scheduler_builder=lambda optimizer: ReduceLROnPlateau(
                optimizer,
                'min',
                patience=5,
                factor=0.5,
                min_lr=1e-5
            )
        )
        print(f"Training time: {time.time() - start_time:.2f} seconds.")
        print(divider)

        # Instantiate the EMDN code predictors (baseline and data-driven)
        _data_driven_emdn_code_predictor = DataDrivenEmdnCodePredictor(
            _emdn_category_predictor=_emdn_category_predictor
        )
        _base_emdn_code_predictor = EmdnCodePredictor(
            pretrained_model=_data_driven_emdn_code_predictor.pretrained_model,
            emdn_nomenclature=_data_driven_emdn_code_predictor.emdn_nomenclature,
            emdn_code_to_embedding=_data_driven_emdn_code_predictor.emdn_code_to_embedding
        )

        model_validation_folder_path = GMDN_EMDN_VALIDATION_DIRECTORY_PATH + model_subfolder
        aggregated_statistics_dfs = []
        for _model, _model_name, _model_validation_folder_path, output_file_name in zip(
                [
                    _base_emdn_code_predictor,
                    _data_driven_emdn_code_predictor
                ],
                [
                    'baseline',
                    'data_driven'
                ],
                [
                    model_validation_folder_path + 'base_emdn_code_predictor/',
                    model_validation_folder_path + 'data_driven_emdn_code_predictor/'
                ],
                [
                    'base_emdn_code_predictor_results',
                    'data_driven_emdn_code_predictor_results'
                ]
        ):
            # Validate the current model and get the path to the validation file
            _output_file_path = validate_model(
                _model=_model,
                results_file_name=output_file_name,
                _test_dataset=_gmdn_emdn_test_dataframe.copy(),
                base_folder_path=_model_validation_folder_path
            )
            # Add to the list of aggregated statistics the results of the validation
            # for the current model
            aggregated_statistics_dfs.append(
                compute_aggregated_statistics(
                    output_file_prefix=_model_name,
                    output_folder=_model_validation_folder_path,
                    input_file_path=_output_file_path,
                    # Pass the name by splitting on the underscore
                    # and setting each first letter to uppercase
                    _name='-'.join(_model_name.split('_')).capitalize()
                )
            )
            # Compute the index-level heatmap for the current model
            # on its validation results
            compute_index_level_heatmap(
                pd.read_csv(
                    _output_file_path,
                    index_col=0
                ),
                _model_name + '_index_level_heatmap',
                _name='-'.join(_model_name.split('_')).capitalize(),
                base_folder_path=_model_validation_folder_path,
            )

        # Concatenate horizontally the two dataframes containing the aggregated statistics
        # for the baseline and the data-driven models
        aggregated_statistics = pd.concat(aggregated_statistics_dfs, axis=1)
        # Reorder the columns in alphabetical order
        aggregated_statistics = aggregated_statistics.reindex(sorted(aggregated_statistics.columns), axis=1)
        # Reorder the rows based on the 'emd_category' column
        aggregated_statistics = aggregated_statistics.sort_index()
        # Store the aggregated statistics in a file
        aggregated_statistics.to_csv(
            model_validation_folder_path + 'correct_option_index_comparison.csv'
        )

        del _emdn_category_predictor
        del _data_driven_emdn_code_predictor
        del _base_emdn_code_predictor

    # Save the test distribution used to evaluate the EMDN code predictors
    output_test_distribution()


def train_and_test() -> None:
    """
    This method executes the pipeline to train and test both the EMDN code and category predictors. Then, it outputs
    the files containing the comparison between the baseline and data-driven models for the EMDN code predictor, both
    using the MPNet and BioBERT models.
    """
    # Load the test dataset containing the GMDN Term Names with related
    # EMDN codes
    gmdn_emdn_test_dataframe = load_test_dataset()

    # Execute the pipeline to train and test the EMDN category predictor
    # and to test the EMDN code predictor on the same test set
    train_emdn_category_predictor(
        _gmdn_emdn_test_dataframe=gmdn_emdn_test_dataframe
    )
    # Print the comparison between the MPNet and BioBERT models both
    # for the baseline and the data-driven EMDN code predictors
    for minimum_level_to_match in range(1, 5):
        compute_models_comparison(
            _models_base_folder_path=GMDN_EMDN_VALIDATION_DIRECTORY_PATH,
            _minimum_level_to_match=minimum_level_to_match,
            _maximum_option_considered=1,
            _file_name=f'models_comparison_min_level_{minimum_level_to_match}.csv'
        )
