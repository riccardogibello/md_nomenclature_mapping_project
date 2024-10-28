from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.__constants import RANDOM_SEED_INT, LAST_TEST_EMDN_DF_INDEX, FIRST_TEST_EMDN_DF_INDEX
from src.__directory_paths import SOURCE_DATA_DIRECTORY_PATH, SQL_CSV_TABLES_DIRECTORY_PATH
from src.__file_paths import EMDN_GMDN_FDA_CLEANED_FILE_PATH, TEST_FILE_PATH, TRAIN_FILE_PATH


def _load_emdn_description_category_correspondences() -> pd.DataFrame:
    """
    This method loads a dataframe of EMDN category - EMDN description correspondences.

    :return: A dataframe containing the EMDN category and the related description.
    """
    # Load the file containing the entire EMDN list of codes
    _emdn_dataset = pd.read_csv(
        SOURCE_DATA_DIRECTORY_PATH + 'EMDN.csv',
        sep=';'
    )
    # Keep the columns corresponding to the EMDN category and the
    # related description
    _emdn_dataset = _emdn_dataset.iloc[:, [1, -3]]
    # Drop any missing value
    _emdn_dataset.dropna(inplace=True)
    _emdn_dataset.columns = ['emdn_category', 'emdn_description']

    return _emdn_dataset


def _resample_category(
        _gmdn_term_name_category_df: pd.DataFrame,
        _category: str,
        _samples_to_retain: int,
) -> pd.DataFrame:
    """
    This method resamples the rows of the dataframe containing the GMDN Term Name - EMDN code - EMDN category, given a
    specific EMDN category, in order to keep only a specific number of samples.

    :param _gmdn_term_name_category_df: The dataframe containing the GMDN Term Name - EMDN code - EMDN category.
    :param _category: The EMDN category for which to resample the rows.
    :param _samples_to_retain: The number of samples to retain for the specified category.

    :return: A dataframe containing the resampled values for the specified category.
    """
    _returned_df = _gmdn_term_name_category_df.copy()
    # Keep only the rows for which 'emdn_category' is different from the specified category
    _returned_df = _returned_df[
        _returned_df['emdn_category'] != _category
        ]
    _resampled_df = _gmdn_term_name_category_df.copy()
    # Keep only the rows for which 'emdn_category' is equal to the specified category
    _resampled_df = _resampled_df[
        _resampled_df['emdn_category'] == _category
        ]
    # Sample randomly the rows of the resampled dataframe
    _resampled_df = _resampled_df.sample(
        n=_samples_to_retain,
        random_state=RANDOM_SEED_INT
    )

    # Return the concatenation of the two dataframes, containing
    # the resampled values for the specified category
    return pd.concat([_returned_df, _resampled_df])


def build_train_test_datasets(
        random_seed: int,
        outliers: list[str] = None,
        test_size: float = 0.2,
        include_emdn_in_training: bool = False,
        _train_file_path: str = TRAIN_FILE_PATH,
        _test_file_path: str = TEST_FILE_PATH,
        keep_underrepresented_categories: bool = False
) -> None:
    """
    This method creates the train and test files, containing the GMDN Term Name - EMDN code - EMDN category
    correspondences. This method cleans the data by keeping only the EMDN categories with a count higher than 200,
    and by splitting the dataset into a training and a testing dataset, keeping the same distribution of the EMDN
    categories.

    :param random_seed: The random seed to use for the shuffling of the rows.
    :param outliers: A list of EMDN categories to consider as outliers and to clip to the 75th percentile of the EMDN
    category counts.
    :param test_size: The size of the test dataset.
    :param include_emdn_in_training: A boolean indicating whether to include the EMDN descriptions in the
    training dataset. If so, the correspondence between EMDN descriptions and EMDN categories is added to the training
    dataset.
    :param _train_file_path: The file path where to store the training dataset.
    :param _test_file_path: The file path where to store the testing dataset.
    :param keep_underrepresented_categories: A boolean indicating whether to keep the underrepresented categories
    in the training dataset.
    """
    # Load the exported file of EMDN-GMDN-FDA mappings
    _emdn_gmdn_fda_df_selected = pd.read_csv(
        EMDN_GMDN_FDA_CLEANED_FILE_PATH
    )

    # Build a dataset containing correspondences between EMDN categories, EMDN codes,
    # and GMDN term names
    gmdn_term_name_category_df = _emdn_gmdn_fda_df_selected[
        ['emdn_category', 'emdn_code', 'gmdn_term_name']
    ].drop_duplicates().reset_index(drop=True)
    # Group by the emdn_category and gmdn_term_name columns, concatenating all the emdn_code values
    gmdn_term_name_category_df = gmdn_term_name_category_df.groupby(
        ['emdn_category', 'gmdn_term_name']
    )['emdn_code'].apply(
        lambda x: ','.join(x)
    ).reset_index()
    # Shuffle the rows of the gmdn_term_name_category_df
    gmdn_term_name_category_df = gmdn_term_name_category_df.sample(
        frac=1,
        random_state=random_seed
    )
    # Build a dataset of the EMDN category counts
    emdn_category_counts = gmdn_term_name_category_df['emdn_category'].value_counts()
    if not keep_underrepresented_categories:
        # Find the EMDN categories with a count higher than 200
        high_count_categories = emdn_category_counts[
            emdn_category_counts > 200
            ].index
        # Keep only the rows for which the EMDN category is in the high count categories
        gmdn_term_name_category_df = gmdn_term_name_category_df[
            gmdn_term_name_category_df['emdn_category'].isin(high_count_categories)
        ]

    # Build the counts for each EMDN category
    emdn_counts = gmdn_term_name_category_df['emdn_category'].value_counts()
    # Calculate the 75th percentile of the EMDN category counts
    percentile_75 = emdn_counts.quantile(0.75)
    # Clip the number of samples to the 75th percentile for
    # all the given outliers
    if outliers is not None:
        for _outlier_category in outliers:
            gmdn_term_name_category_df = _resample_category(
                _gmdn_term_name_category_df=gmdn_term_name_category_df,
                _category=_outlier_category,
                _samples_to_retain=int(percentile_75)
            )

    # Split the gmdn_term_name_category_df into a training and a testing dataset, keeping
    # the same distribution of the EMDN categories
    gmdn_train_df, gmdn_test_df = train_test_split(
        gmdn_term_name_category_df,
        test_size=test_size,
        stratify=gmdn_term_name_category_df['emdn_category'],
        random_state=random_seed
    )
    # Keep the gmdn_term_name, emdn_category, and emdn_code columns for the training dataset
    gmdn_train_df = gmdn_train_df[['gmdn_term_name', 'emdn_category', 'emdn_code']]
    # Rename the 'gmdn_term_name' column to 'text'
    gmdn_train_df.rename(
        columns={'gmdn_term_name': 'text'},
        inplace=True
    )

    if include_emdn_in_training:
        # Build a dataset containing correspondences between EMDN categories and EMDN descriptions
        emdn_description_category_df = _load_emdn_description_category_correspondences()
        # Rename the 'emdn_description' column to 'text'
        emdn_description_category_df.rename(
            columns={'emdn_description': 'text'},
            inplace=True
        )
        # Concatenate the two dataframes
        training_df = pd.concat([gmdn_train_df, emdn_description_category_df])
    else:
        training_df = gmdn_train_df

    training_df.drop_duplicates(inplace=True)
    training_df.reset_index(drop=True, inplace=True)
    gmdn_test_df.drop_duplicates(inplace=True)
    gmdn_test_df.reset_index(drop=True, inplace=True)
    # Shuffle the rows of the training dataset
    training_df = training_df.sample(
        frac=1,
        random_state=random_seed
    )
    # Store the training dataset in a file
    training_df.to_csv(
        _train_file_path,
        index=False
    )
    # Split every row in which the 'emdn_code' column contains a comma
    # into multiple rows, each containing a single 'emdn_code' value
    gmdn_test_df = gmdn_test_df.assign(
        emdn_code=gmdn_test_df['emdn_code'].str.split(',')
    ).explode('emdn_code')
    # Store the GMDN Term Name - EMDN code - EMDN category in a file
    gmdn_test_df.to_csv(
        _test_file_path,
        index=False
    )


def _load_gold_standard() -> pd.DataFrame:
    """
    This method loads the gold standard (i.e., test) dataset for the GMDN-EMDN mapping task by keeping only the columns
    related to the GMDN term name and EMDN code.

    :return: A dataframe in which the test data are loaded and cleaned by keeping only the GMDN Term Name and
    EMDN code.
    """
    # Load the test dataset used in the EMDN category predictor and keep
    # only the columns related to the GMDN term name and EMDN code
    _test_dataset = pd.read_csv(
        TEST_FILE_PATH,
        index_col=0
    )[['gmdn_term_name', 'emdn_code']]
    # Load the file containing the cleaned EMDN-GMDN-FDA relationships and
    # keep only the 'gmdn_term_name' and 'emdn_code' columns
    _gold_standard = pd.read_csv(
        SQL_CSV_TABLES_DIRECTORY_PATH + 'emdn_gmdn_fda.csv',
    )[['gmdn_term_name', 'emdn_code']]
    # Join the dataframes on the 'gmdn_term_name' and 'emdn_code' columns
    _test_dataset = _test_dataset.merge(
        _gold_standard,
        on=['gmdn_term_name', 'emdn_code'],
        how='inner'
    )[['gmdn_term_name', 'emdn_code']]
    del _gold_standard
    # Drop possible duplicates and NaN values
    _test_dataset = _test_dataset.drop_duplicates().dropna()
    # Reset the index of the dataframe
    _test_dataset.reset_index(drop=True, inplace=True)

    return _test_dataset


def load_test_dataset() -> pd.DataFrame:
    """
    This method loads the test dataset for the GMDN-EMDN mapping task. It keeps, for each GMDN Term Name, the most
    frequent EMDN (sub)code related to it.

    :return: A dataframe in which the test data are loaded and cleaned by keeping, for each GMDN Term Name, the EMDN
    (sub)code that is most frequently related to it.
    """
    # Load the test dataset containing the GMDN Term Names with related
    # EMDN codes and the count of each pair
    _gold_standard = _load_gold_standard()

    # Get all the unique GMDN term names and a map which
    # contains the relation between EMDN code and count
    _unique_gmdn_term_names: dict[str, list[str]] = {}
    for _row in _gold_standard.to_numpy():
        gmdn_term_name = _row[0]
        sub_emdn_code = _row[1]

        if gmdn_term_name in _unique_gmdn_term_names:
            _unique_gmdn_term_names[gmdn_term_name].append(sub_emdn_code)
        else:
            _unique_gmdn_term_names[gmdn_term_name] = [sub_emdn_code]

    gmdn_term_name_emdn_code: dict[str, str] = {}
    # For each GMDN Term Name and EMDN map containing the correspondence
    # between code and count
    for gmdn_term_name, emdn_codes in _unique_gmdn_term_names.items():
        # Get a list of empty strings, one for
        # each EMDN code
        char_lists = ['' for _ in emdn_codes]
        # Iterate in parallel over the positions
        # of the EMDN code strings, stopping
        # at the shortest one
        for chars in zip(*emdn_codes):
            # For each character of the current position
            for index, char in enumerate(chars):
                index: Any
                # Update the character list
                # of the corresponding EMDN code
                # by adding the character
                char_lists[index] += char

        # Create a map between the unique sub-codes
        # and the counts
        common_sub_codes_counts = {
            code: 0
            for code in set(char_lists)
        }

        # For each sub-code and the related number of times it
        # was voted (zero at the beginning)
        for sub_emdn_code, count in common_sub_codes_counts.items():
            # For each EMDN code of the initial map
            for complete_emdn_code in emdn_codes:
                # If the sub-code is a prefix of the EMDN code
                if complete_emdn_code.startswith(sub_emdn_code):
                    # Update the count for the current sub-code
                    count += 1

            # Store the count in the map for the current sub-code
            common_sub_codes_counts[sub_emdn_code] = count

        # Take the EMDN code with the highest count among all the
        # sub-codes
        best_emdn_code = max(
            common_sub_codes_counts,
            key=common_sub_codes_counts.get
        )
        # Set the selected EMDN code as the translation for
        # the current GMDN term name
        gmdn_term_name_emdn_code[gmdn_term_name] = best_emdn_code

    # Transform the dictionary into a DataFrame to have the
    # test EMDN-GMDN dataset
    new_columns: Any = ['gmdn_term_name', 'emdn_code']
    _gmdn_emdn_test_dataframe = pd.DataFrame(
        gmdn_term_name_emdn_code.items(),
        columns=new_columns
    )
    first_emdn_index: Any = FIRST_TEST_EMDN_DF_INDEX
    last_emdn_index: Any = LAST_TEST_EMDN_DF_INDEX
    for index in range(first_emdn_index, last_emdn_index):
        emdn_predicted_column = str(index)
        emdn_predicted_prob_column = emdn_predicted_column + '_prob'
        _gmdn_emdn_test_dataframe[emdn_predicted_column] = ''
        _gmdn_emdn_test_dataframe[emdn_predicted_prob_column] = 0.0
    # Add a column for indicating the level at which the EMDN gold standard was matched
    _gmdn_emdn_test_dataframe['level'] = -1
    # Add a column indicating the index at which the EMDN gold standard category was identified
    _gmdn_emdn_test_dataframe['index'] = -1

    return _gmdn_emdn_test_dataframe
