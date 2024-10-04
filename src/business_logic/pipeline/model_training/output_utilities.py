import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.business_logic.utilities.os_utilities import find_files_containing_string
from src.data_model.nomenclature_codes.emdn_code import get_code_level
from src.__directory_paths import MODELS_DIRECTORY_PATH, GMDN_EMDN_VALIDATION_DIRECTORY_PATH, FILES_DIRECTORY


def compute_aggregated_statistics(
        output_file_prefix: str,
        output_folder: str,
        input_file_path: str,
        _name: str = 'name'
) -> pd.DataFrame:
    """
    This method computes the aggregated statistics for the given input file path. It outputs the histograms containing
    the prediction results by EMDN category representing the level of match and the position of the correct option.

    :param output_file_prefix: The prefix of the output files.
    :param output_folder: The output folder where the files will be saved.
    :param input_file_path: The input file path containing the validation results.
    :param _name: The name of the model.

    :return: A dataframe containing the aggregated statistics.
    """
    # Load the files with the validation results
    validation_results = pd.read_csv(
        input_file_path,
        index_col=0
    )
    # Add a new column containing the EMDN category
    validation_results['emdn_category'] = validation_results[
        'emdn_code'
    ].apply(lambda x: x[0])

    # Reorder the rows based on the 'emdn_category' column
    validation_results = validation_results.sort_values(by='emdn_category')

    # Add 1 to the 'index' column to have a 1-based index
    validation_results['index'] = validation_results['index'] + 1
    # Rename the 'index' column as 'Position of Correct Option'
    new_index_name = 'Position of Correct Option'
    new_emdn_category_name = 'EMDN Category'
    validation_results.rename(
        columns={'emdn_category': new_emdn_category_name},
        inplace=True
    )
    validation_results.rename(
        columns={'index': new_index_name},
        inplace=True
    )

    # Build the histogram containing the aggregated bars for each EMDN category; each of the group presents
    # the distribution of the 'level' column
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.histplot(
        data=validation_results,
        x=new_emdn_category_name,
        hue='level',
        multiple='stack',
        ax=ax,
        palette='viridis'
    )
    # Save the histogram in the given directory, prefixed by the name
    plt.tight_layout()
    plt.savefig(output_folder + output_file_prefix + '_match_level_distribution.png')
    plt.close()
    # Build the histogram containing the aggregated bars for each EMDN category; each of the group presents
    # the distribution of the 'index' column
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.histplot(
        data=validation_results,
        x=new_emdn_category_name,
        hue=new_index_name,
        multiple='stack',
        ax=ax,
        palette='viridis',
        # Set the legend with two columns
        legend=True
    )
    # Set the all the text size to 18
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel('EMDN Category', fontsize=18)
    plt.ylabel('Number of Samples', fontsize=18)
    plt.setp(ax.get_legend().get_texts(), fontsize='14')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='14')  # for legend title
    plt.title(_name + ' Mapping Results by EMDN Category', fontsize=18)

    # Save the histogram in the given directory, prefixed by the name
    plt.tight_layout()
    plt.savefig(output_folder + output_file_prefix + '_correct_option_index_distribution.png')
    plt.close()

    # Build a dataframe which creates a pivot table between the EMDN category and the 'index' column
    correct_option_index = validation_results.pivot_table(
        index=new_emdn_category_name,
        columns=new_index_name,
        values='gmdn_term_name',
        aggfunc='count',
        fill_value=0,
    )
    new_cols = []
    for col in correct_option_index.columns:
        col_string = str(col) if col >= 10 else f'0{col}'
        # Prefix any column name with the 'name' parameter
        new_cols.append(f'{col_string}_index_{output_file_prefix}')
    correct_option_index.columns = new_cols

    return correct_option_index


def compute_index_level_heatmap(
        _validation_results: pd.DataFrame,
        file_name: str = 'index_level_heatmap',
        show_percentage: bool = False,
        _name: str = 'title',
        base_folder_path: str = ''
) -> None:
    """
    This method computes the heatmap presented in the paper, which links the model certainty with the position of the
    correct option for the prediction of the different EMDN categories.

    :param _validation_results: The validation results dataframe.
    :param file_name: The name of the file to be saved.
    :param show_percentage: A boolean flag indicating whether to show the percentage or not in the heatmap.
    :param _name: The name of the model.
    :param base_folder_path: The base folder path where the file will be saved.
    """
    # Pivot the table to have the 'index' and 'level' columns as indexes
    _validation_results = _validation_results.pivot_table(
        index='index',
        columns='level',
        values='gmdn_term_name',
        aggfunc='count',
        fill_value=0
    )

    # Rename 'index' as 'Order of Correct Option' and 'level' as 'Level of EMDN Match'
    _validation_results.index.name = 'Order of Correct Option'
    _validation_results.columns.name = 'Level of EMDN Match'

    # Save this as a heatmap
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.heatmap(
        data=_validation_results,
        cmap='Reds',
        ax=ax,
        annot=True,
        fmt="d" if not show_percentage else ".2f",
        annot_kws={"size": 18},
        cbar=False,
    )
    # Set the y and x labels to 18
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel('Level of EMDN Match', fontsize=18)
    plt.ylabel('Position of Correct Option', fontsize=18)
    # Set the title to 18
    plt.title(_name + ' Correct Category Position vs Match Level', fontsize=18)
    # Set the tight layout
    plt.tight_layout()
    plt.savefig(
        base_folder_path + file_name + '.png'
    )
    plt.close()


def _load_and_manipulate_results(
        _file_path: str,
        _minimum_level_to_match: int = 1,
        _maximum_option_considered: int = 1
) -> pd.DataFrame:
    """
    This method manipulates the validation results to prepare them for the metrics calculation. It filters the results
    based on the minimum level to match and the maximum option considered. It also creates a new column called
    'predicted_category' which contains the predicted EMDN category, to be used in the metrics calculation.

    :param _file_path: The file path containing the validation results.
    :param _minimum_level_to_match: The minimum level to match in order to consider the prediction as correct.
    :param _maximum_option_considered: The maximum option considered for the prediction.

    :return: The dataframe containing the refactored results.
    """
    # Build a range of values between 2 and 2 + 22, with a step of 2
    match_levels = [
        str(_opt_index)
        for _opt_index in range(1, 23)
    ]
    # Keep only the first values in match_levels given the maximum option considered
    match_levels = match_levels[:_maximum_option_considered]

    columns_to_keep = ['gmdn_term_name', 'emdn_code', 'level', 'index']
    columns_to_keep.extend(match_levels)

    # Open the file
    _test_dataset: pd.DataFrame = pd.read_csv(_file_path, index_col=0).loc[:, columns_to_keep]
    # Create a category column
    _test_dataset['emdn_category'] = _test_dataset['emdn_code'].apply(lambda x: x[0])
    # Build a new column called 'true_level' which is equal to the level of the true EMDN code
    _test_dataset['true_level'] = _test_dataset['emdn_code'].apply(
        lambda x: get_code_level(x)
    )
    # Keep all the rows in which the true level is greater than the minimum level to match
    _test_dataset = _test_dataset[_test_dataset['true_level'] >= _minimum_level_to_match]
    # Reset the index
    _test_dataset.reset_index(drop=True, inplace=True)
    # Drop the 'true_level' column
    _test_dataset.drop(columns=['true_level'], inplace=True)
    # Create a predicted category column
    _test_dataset['predicted_category'] = [''] * len(_test_dataset)

    _dataset_index = 0
    for _index, _row in _test_dataset.iterrows():
        # Get the current EMDN code, the level of match and the index at which the
        # EMDN category was identified
        _emdn_code = _row.loc['emdn_code']
        _level = _row.loc['level']
        _index = _row.loc['index'] + 1
        # Get the true EMDN category
        _true_category = _emdn_code[0]
        # If the index in which the EMDN category was identified is higher than the maximum option considered
        if _index > _maximum_option_considered:
            # Set the level to 0
            _test_dataset.loc[_dataset_index, 'level'] = 0
            # Set the 'predicted_category' to the category of the last possible option
            _predicted_emdn_code = _row.loc[match_levels[-1]]
            _test_dataset.loc[_dataset_index, 'predicted_category'] = _predicted_emdn_code[0]
        else:
            # If the level of match is lower than the minimum level to match
            if _level < _minimum_level_to_match:
                # Set the level to 0
                _test_dataset.loc[_dataset_index, 'level'] = 0
                # Set the 'predicted_category' to the category of the last possible option
                _predicted_emdn_code = _row.loc[match_levels[-1]]
                _test_dataset.loc[_dataset_index, 'predicted_category'] = 'AZ'
            else:
                # Get the EMDN code at the specified index
                _predicted_emdn_code = _row.loc[match_levels[_index - 1]]
                # Set the 'predicted_category' to the category of the predicted EMDN code
                _test_dataset.loc[_dataset_index, 'predicted_category'] = _predicted_emdn_code[0]

        _dataset_index += 1

    return _test_dataset


def _build_by_category_statistics(
        _true_labels: pd.Series,
        _predicted_labels: pd.Series,
        _model: str,
        _algorithm: str
) -> list:
    """
    This method builds the statistics by EMDN category. It calculates the accuracy for each category.

    :param _true_labels: The gold standard labels.
    :param _predicted_labels: The predicted labels.
    :param _model: The model name.
    :param _algorithm: The algorithm name.

    :return: A list containing the statistics by EMDN category.
    """
    row_list = []
    category_labels: dict[str, list[tuple[str, str]]] = {}
    for _true_label, _predicted_label in zip(_true_labels, _predicted_labels):
        # Get the EMDN category
        _emdn_category = _true_label[0]
        if _emdn_category not in category_labels:
            category_labels[_emdn_category] = []
        category_labels[_emdn_category].append((_true_label, _predicted_label))

    for _category, _labels in category_labels.items():
        _true_labels, _predicted_labels = zip(*_labels)
        accuracy = accuracy_score(
            _true_labels,
            _predicted_labels
        )
        row_list.append(
            [_model, _algorithm, _category, accuracy]
        )

    return row_list


def compute_models_comparison(
        _models_base_folder_path: str = MODELS_DIRECTORY_PATH,
        _minimum_level_to_match: int = 1,
        _maximum_option_considered: int = 1,
        _file_name: str = 'models_comparison.csv'
) -> pd.DataFrame:
    """
    This method computes the comparison between the different models. It loads the results of the models and calculates
    the accuracy, precision, recall and F1 score for each model. It also calculates the metrics by EMDN category.

    :param _models_base_folder_path: The base folder path containing the models' data.
    :param _minimum_level_to_match: The minimum level to match in order to consider the prediction as correct.
    :param _maximum_option_considered: The maximum option considered for the prediction.
    :param _file_name: The name of the file to be saved.

    :return: The dataframe containing the comparison results.
    """
    # Find all the files in the subdirectories that contain the 'code_predictor_results.csv' string
    results_files = find_files_containing_string(
        _models_base_folder_path,
        'code_predictor_results.csv'
    )
    assert len(results_files) == 4, "The number of files must be 4."
    # Reorder the file paths in alphabetical order
    results_files = sorted(results_files)

    aggregated_models_stats = []
    aggregated_models_stats_by_category = []
    columns = ['Model', 'Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    columns_by_category = ['Model', 'Algorithm', 'EMDN Category', 'Accuracy']
    for _file_path in results_files:
        file_path_splits = _file_path.split('/')
        # Extract the current model structure
        model_name = file_path_splits[-3].split('_')[0]
        if model_name == 'biobert':
            model_name = 'BioBERT'
        elif model_name == 'mpnet':
            model_name = 'MPNet'
        algorithm = file_path_splits[-1].split('_')[0]
        if algorithm == 'base':
            algorithm = 'Baseline'
        else:
            algorithm = 'Data-Driven'

        # Load the results
        _test_dataset = _load_and_manipulate_results(
            _file_path,
            _minimum_level_to_match=_minimum_level_to_match,
            _maximum_option_considered=_maximum_option_considered
        )

        y_true = _test_dataset['emdn_category']
        y_pred = _test_dataset['predicted_category']

        # Calculate metrics
        accuracy = accuracy_score(
            y_true,
            y_pred,
        )
        precision = precision_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )
        recall = recall_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )
        f1 = f1_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )
        # Print the metrics
        aggregated_models_stats.append(
            [model_name, algorithm, accuracy, precision, recall, f1]
        )
        aggregated_models_stats_by_category.extend(
            _build_by_category_statistics(
                y_true,
                y_pred,
                model_name,
                algorithm
            )
        )

    _result_dataframe = pd.DataFrame(
        aggregated_models_stats,
        columns=columns
    )
    _result_dataframe.to_csv(
        GMDN_EMDN_VALIDATION_DIRECTORY_PATH + _file_name,
        index=False
    )
    # Save the results by category
    _result_dataframe = pd.DataFrame(
        aggregated_models_stats_by_category,
        columns=columns_by_category
    )
    _file_name = 'models_comparison_by_category_min_lev_' + str(_minimum_level_to_match) + '.csv'
    _result_dataframe.to_csv(
        GMDN_EMDN_VALIDATION_DIRECTORY_PATH + _file_name,
        index=False
    )

    return _result_dataframe


def output_test_distribution() -> None:
    """
    This method loads the test results of the EMDN code predictor and builds a histogram. Note that all the models
    (i.e., BioBERT or MPNet based, Baseline or Data-Driven) are evaluated on the same test set, so only one file is
    loaded.
    """
    file_paths = find_files_containing_string(
        base_folder_path=FILES_DIRECTORY,
        search_substring='emdn_code_predictor_results.csv'
    )
    file_path = file_paths[0]

    file_df = pd.read_csv(file_path)
    # Keep the 'emdn_code' column
    file_df = file_df[['emdn_code']]
    file_df['emdn_category'] = file_df['emdn_code'].apply(lambda x: x[0])
    # Keep only the 'emdn_category' column
    file_df = file_df[['emdn_category']]
    # Build a count for each category
    file_df = file_df.groupby('emdn_category').size().reset_index(name='count')
    # Save an image of the count
    ax = file_df.plot(
        x='emdn_category',
        y='count',
        kind='bar',
        legend=False
    )
    ax.set_xlabel('EMDN category')
    ax.set_ylabel('# Samples')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.get_figure().savefig(
        GMDN_EMDN_VALIDATION_DIRECTORY_PATH + 'emdn_code_prediction_test_distribution.png'
    )
