from typing import Callable

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src.__directory_paths import OUTPUT_DATA_DIRECTORY_PATH
from src.__file_paths import MEDICAL_SPECIALTY_FILE_PATH, EMDN_GMDN_FDA_FILE_PATH, EMDN_GMDN_FDA_CLEANED_FILE_PATH


def integrate_specialty_values(
        _df: pd.DataFrame
) -> None:
    # Build a new column, called 'integrated_specialty',
    # where the 'medical_specialty' value is placed (if present),
    # otherwise the 'panel' value is placed
    _df['integrated_specialty'] = _df['medical_specialty'].fillna(_df['panel'])

    assert _df['integrated_specialty'].isna().sum() == 0


def build_confusion_matrix(
        input_dataframe: pd.DataFrame,
        normalize_over_rows: bool = True,
        file_name: str = 'confusion_matrix',
        title: str = 'Confusion Matrix',
        show_percentage: bool = True,
):
    if not input_dataframe.columns.__contains__('emdn_category'):
        raise ValueError('The input dataframe does not contain the column "emdn_category"')
    if not input_dataframe.columns.__contains__('medical_specialty_complete'):
        raise ValueError('The input dataframe does not contain the column "medical_specialty_complete"')

    # Extract all the correspondences between 'emdn_category' and 'medical_specialty'
    _confusion_matrix = pd.crosstab(
        index=input_dataframe['medical_specialty_complete'],
        columns=input_dataframe['emdn_category']
    )
    if show_percentage:
        # Normalize each column of the confusion matrix to be a percentage between 0 and 1
        if normalize_over_rows:
            # This represents the probability of an EMDN category given a medical specialty
            # P(emdn_category|medical_specialty)
            _confusion_matrix = _confusion_matrix.div(_confusion_matrix.sum(axis=1), axis=0)
        else:
            # This represents the probability of a medical specialty given an EMDN category
            # P(medical_specialty|emdn_category)
            _confusion_matrix = _confusion_matrix.div(_confusion_matrix.sum(axis=0), axis=1)

    # Step 3: Fill missing values with zeros
    _confusion_matrix.fillna(
        0,
        inplace=True
    )

    plt.figure(
        figsize=(28, 20),
    )
    font_size = 20
    sns.heatmap(
        _confusion_matrix,
        annot=True,
        fmt="d" if not show_percentage else ".2f",
        annot_kws={"size": font_size}
    )

    plt.title(title, fontsize=24, fontweight='bold')

    # Set the font size of the labels to be 18
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # Set also the other labels to be 18
    plt.xlabel('EMDN Category', fontsize=font_size)
    plt.ylabel('FDA Medical Specialty', fontsize=font_size)

    # Set the tight layout
    plt.tight_layout()

    # Save the confusion matrix to a file
    plt.savefig(OUTPUT_DATA_DIRECTORY_PATH + file_name + '.png')
    plt.close()

    return _confusion_matrix


def add_complete_specialty_name(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    This method integrates the medical specialty code and complete name to the given dataframe.

    :param df: The dataframe to which the medical specialty code and complete name will be added.

    :return: The dataframe with the medical specialty code and complete name added.
    """
    # Load the medical specialties
    _medical_specialties_df = pd.read_csv(
        MEDICAL_SPECIALTY_FILE_PATH,
        sep=';'
    )
    # Add the complete medical specialty name to the initial dataframe
    _emdn_gmdn_fda_df_complete = df.merge(
        _medical_specialties_df,
        how='inner',
        left_on='medical_specialty',
        right_on='code'
    )

    # Rename the code column to 'medical_specialty_complete'
    _emdn_gmdn_fda_df_complete.rename(
        columns={'complete_name': 'medical_specialty_complete'},
        inplace=True
    )
    # Replace any 'medical_specialty_complete' value with itself and the 'medical_specialty' in parentheses
    _emdn_gmdn_fda_df_complete['medical_specialty_complete'] = (
            _emdn_gmdn_fda_df_complete['medical_specialty_complete'] +
            ' (' + _emdn_gmdn_fda_df_complete['medical_specialty'] + ')'
    )
    _emdn_gmdn_fda_df_complete.drop(columns=['code'], inplace=True)

    return _emdn_gmdn_fda_df_complete


def order_by_probability(
        dataframe: pd.DataFrame,
) -> pd.DataFrame:
    # Build a dataframe which contains, for each specialty, a sorted list of the EMDN categories,
    # from the most probable to the least probable
    specialty_emdn_category_rows = []

    # Get the columns index name
    index_name = dataframe.columns.name

    # For each column, representing the condition
    # fixed on the probability
    for column_name in dataframe.columns:
        # Extract the conditional probability of the indexed variable
        conditional_probabilities = dataframe[column_name]
        # Sort the conditional probabilities from the most probable variable
        # to the least probable
        sorted_specialty_emdn_category = conditional_probabilities.sort_values(ascending=False)
        # For each index, representing the conditioned variable
        for _index in sorted_specialty_emdn_category.index:
            # Append the row to the list
            specialty_emdn_category_rows.append(
                [column_name, _index, sorted_specialty_emdn_category[_index]]
            )

    # If the conditioned variable is the EMDN category
    if index_name == 'emdn_category':
        specialty_emdn_category_df = pd.DataFrame(
            specialty_emdn_category_rows,
            columns=['emdn_category', 'medical_specialty', 'conditional_probability']
        )
    else:
        specialty_emdn_category_df = pd.DataFrame(
            specialty_emdn_category_rows,
            columns=['medical_specialty', 'emdn_category', 'conditional_probability']
        )

    return specialty_emdn_category_df


def aggregate_indexes_by_key(
        _df: pd.DataFrame,
        _key: str
) -> dict[str, list[int]]:
    # Build a map to keep track of all the dataset rows containing
    # the specified key
    _key_indexes_map: dict[str, list[int]] = {}
    _index = 0
    # Get the index of the specified key
    _gmdn_t_name_index = _df.columns.get_loc(_key)
    # For each row in the dataset
    for row in _df.to_numpy():
        # Get the value of the key
        key_value = row[_gmdn_t_name_index]
        # Add the key - dataset index association to the map
        if key_value in _key_indexes_map:
            _key_indexes_map[key_value].append(_index)
        else:
            _key_indexes_map[key_value] = [_index]
        _index += 1

    return _key_indexes_map


def get_best_emdn_categories(
        _probability_df: pd.DataFrame,
        _fda_specialty: str,
        _emdn_weight_df: pd.DataFrame
) -> list[str]:
    # Get all the values of the EMDN probabilities given the specialty
    sorted_emdn_categories = _probability_df.loc[
        _fda_specialty
    ]
    # Build a dataframe in which the index is the EMDN category and the column is the probability
    sorted_emdn_categories = pd.DataFrame(sorted_emdn_categories)
    # Reset the index
    sorted_emdn_categories.reset_index(
        inplace=True
    )
    # _emdn_weight_df['count'] = 1
    sorted_emdn_categories.columns = ['emdn_category', 'count']
    # Join the dataframes over the index
    sorted_emdn_categories = sorted_emdn_categories.join(
        _emdn_weight_df,
        how='left',
        on='emdn_category',
        lsuffix='_x',
        rsuffix='_y'
    )
    # Replace any NaN value with 0
    sorted_emdn_categories.fillna(
        0,
        inplace=True
    )
    # Multiply the columns containing 'count_x' and 'count_y' values
    sorted_emdn_categories['count'] = sorted_emdn_categories['count_x'] * sorted_emdn_categories['count_y']
    # Drop the columns containing 'count_x' and 'count_y' values
    sorted_emdn_categories.drop(
        columns=['count_x', 'count_y'],
        inplace=True
    )
    # Sort the EMDN categories by the probability in descending order
    sorted_emdn_categories = sorted_emdn_categories.sort_values(
        by='count',
        ascending=False
    )
    # Get the EMDN categories
    sorted_emdn_categories = sorted_emdn_categories['emdn_category'].tolist()
    return sorted_emdn_categories


def aggregate_by_emdn_category(
        _df: pd.DataFrame,
        _searched_gmdn_term_name: str | None = None
) -> pd.DataFrame:
    if _searched_gmdn_term_name is not None:
        _selected_df = _df[
            _df['gmdn_term_name'] == _searched_gmdn_term_name
            ]
    else:
        _selected_df = _df

    # Aggregate by EMDN category and sum all the values in 'count' column
    _emdn_category_count_df = _selected_df[['emdn_category', 'count']].groupby(
        by=['emdn_category']
    ).sum()
    """# Add all the EMDN categories which are not present in the dataset
    # with a count of 0
    emdn_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for letter in emdn_letters:
        if letter not in _emdn_category_count_df.index:
            _emdn_category_count_df.loc[letter] = 0
    # Add to all the counts a small value to avoid vanishing probabilities
    _emdn_category_count_df += 10"""
    # Compute the percentage of each EMDN category
    _emdn_category_count_df = _emdn_category_count_df / _emdn_category_count_df.sum()

    return _emdn_category_count_df


def add_selected_rows(
        _df: pd.DataFrame,
        _current_rows: list,
        _key_indexes_map: dict[str, list[int]],
        _get_conditional_probability: Callable[[str, str], float],
        _get_best_emdn_categories: Callable[[str, pd.DataFrame], list[str]]
):
    for gmdn_term, row_indexes in _key_indexes_map.items():
        """tmp_gmdn_term = gmdn_term
        tmp_gmdn_term = re.sub(r'[^a-zA-Z0-9\- ]', '', tmp_gmdn_term)
        tmp_gmdn_term = tmp_gmdn_term.lower()
        if tmp_gmdn_term.__contains__('epidural anaesthesia set n'):
            print('here')"""
        # Get all the rows of the initial dataset associated to the current key
        _rows = _df.loc[row_indexes].reset_index(drop=True)
        # Get the dataframe containing the probabilities of the EMDN categories
        # for the specific GMDN Term Name
        _emdn_weight_df = aggregate_by_emdn_category(
            _df=_rows
        )

        # Get the index of the 'medical_specialty' and 'emdn_category' columns
        medical_specialty_index = _rows.columns.get_loc('medical_specialty_complete')
        emdn_category_index = _rows.columns.get_loc('emdn_category')

        # Prepare a dictionary to keep track of the association between
        # medical specialty - EMDN category - row indexes
        specialty_emdn_row_indexes: dict[str, dict[str, list[int]]] = {}
        # These two maximum indicators are used at the end to filter
        # the rows based on the medical specialty which shows the
        # best probability with all the related EMDN categories
        max_probability = 0
        max_specialty = ''
        rows_numpy = _rows.to_numpy()
        index = 0
        for emdn_category, medical_specialty in zip(
                rows_numpy[:, emdn_category_index],
                rows_numpy[:, medical_specialty_index]
        ):
            # Get the conditional probability
            # (either P(emdn_category|medical_specialty) or P(medical_specialty|emdn_category))
            conditional_probability = _get_conditional_probability(emdn_category, medical_specialty)
            # Multiply the conditional probability by the sample probability of the EMDN category
            # (P(emdn_category))
            try:
                conditional_probability *= float(_emdn_weight_df.loc[emdn_category].item())
            except KeyError:
                # If the EMDN category is not found in the probabilities dataframe
                # set the conditional probability to 0
                conditional_probability = 0

            # If the current conditional probability is greater than the maximum probability found
            if conditional_probability >= max_probability:
                # Set the current conditional probability as the maximum probability
                max_probability = conditional_probability
                # Set the current best medical specialty as the one filtering the rows
                max_specialty = medical_specialty

            # Update the map to keep the medical specialty - EMDN category - row indexes
            # relationship
            if medical_specialty in specialty_emdn_row_indexes:
                emdn_category_indexes = specialty_emdn_row_indexes[medical_specialty]
            else:
                emdn_category_indexes = {}
                specialty_emdn_row_indexes[medical_specialty] = emdn_category_indexes
            if emdn_category in emdn_category_indexes:
                emdn_category_indexes[emdn_category].append(index)
            else:
                emdn_category_indexes[emdn_category] = [index]

            index += 1

        # If the best medical specialty has been found
        if max_specialty != '':
            # Find all the EMDN category - indexes relationships
            emdn_indexes = specialty_emdn_row_indexes[max_specialty]
            # Get the ordered list of most important EMDN categories
            # for the best medical specialty
            emdn_categories = _get_best_emdn_categories(
                max_specialty,
                _emdn_weight_df
            )
            # For each of the ordered EMDN categories
            for emdn_category in emdn_categories:
                # If the EMDN category is related to the given medical specialty
                if emdn_category in emdn_indexes.keys():
                    # Add the related rows to the ones to be kept
                    _current_rows.extend(rows_numpy[emdn_indexes[emdn_category]])
                    # If the best EMDN category is found, break the loop
                    break


def clean_nomenclature_mappings() -> None:
    """
    This method loads all the mappings between EMDN-GMDN-FDA specialty codes, cleans and stores them in a
    new dataset.

    :return: The path to the cleaned dataset.
    """
    # Load the correspondences between EMDN-GMDN-FDA specialty codes
    emdn_gmdn_fda_df = pd.read_csv(
        EMDN_GMDN_FDA_FILE_PATH
    )
    # Integrate the specialty values by adding the complete medical specialty name
    emdn_gmdn_fda_df = add_complete_specialty_name(
        emdn_gmdn_fda_df
    )
    # Drop any column which is not necessary
    emdn_gmdn_fda_df.drop(
        columns=['product_code', 'device_name', 'emdn_id', 'gmdn_id'],
        inplace=True
    )

    # Compute the probability of a medical specialty given an EMDN category
    # P(medical_specialty|emdn_category)
    _ = build_confusion_matrix(
        input_dataframe=emdn_gmdn_fda_df,
        normalize_over_rows=False,
        file_name='fda_specialty_given_emdn_category',
        title='FDA Medical Specialty Probability given EMDN Categories'
    )
    # Compute the probability of an EMDN category given a medical specialty
    # P(emdn_category|medical_specialty)
    emdn_given_fda_specialty_df = build_confusion_matrix(
        input_dataframe=emdn_gmdn_fda_df,
        normalize_over_rows=True,
        file_name='emdn_category_given_specialty',
        title='EMDN Category Probabilities given FDA Medical Specialties'
    )
    # Set the columns to be dropped
    excluded_columns = ['original_device_mapping_id']
    # Drop the excluded column, to find the counts of the single
    # EMDN-GMDN-FDA correspondences
    emdn_gmdn_fda_df_grouped = emdn_gmdn_fda_df.drop(
        columns=excluded_columns
    ).groupby(
        # Group by all the columns except the excluded ones
        by=emdn_gmdn_fda_df.columns.drop(
            excluded_columns
        ).tolist()
        # Find the size of each group
    ).size().reset_index(name='count')
    # Find the map linking each GMDN Term Name to the
    # indexes of the dataset rows containing the specified key
    gmdn_t_name_map = aggregate_indexes_by_key(
        emdn_gmdn_fda_df_grouped,
        'gmdn_term_name'
    )

    # Prepare a list for the rows to be selected from the original
    # dataset of nomenclature mappings
    selected_rows = []
    # Select, in the given dataframe, for each GMDN Term Name, the rows
    # that contain the most probable EMDN category (based on the associated specialty)
    add_selected_rows(
        _df=emdn_gmdn_fda_df_grouped,
        _current_rows=selected_rows,
        _key_indexes_map=gmdn_t_name_map,
        _get_conditional_probability=lambda e_cat, fda_specialty: emdn_given_fda_specialty_df.loc[fda_specialty, e_cat],
        _get_best_emdn_categories=lambda fda_specialty, emdn_cat_w: get_best_emdn_categories(
            emdn_given_fda_specialty_df.copy(),
            fda_specialty,
            emdn_cat_w
        )
    )
    emdn_gmdn_fda_df_selected = pd.DataFrame(
        selected_rows,
        columns=emdn_gmdn_fda_df_grouped.columns
    )

    # Create the Other and In-Vitro Diagnostic categories
    # to aggregate some under-represented specialties
    other_category_label = 'OT'
    ivd_label = 'IVD'
    emdn_gmdn_fda_df_selected = replace_specialty_labels(
        _df=emdn_gmdn_fda_df_selected,
        _replaced_categories=['SU', 'OB', 'EN', 'NE', 'OP', 'PM', ],
        _new_category=other_category_label,
        _complete_new_category='Other (OT)'
    )
    emdn_gmdn_fda_df_selected = replace_specialty_labels(
        _df=emdn_gmdn_fda_df_selected,
        _replaced_categories=['CH', 'HE', 'MI', 'IM', 'TX', 'PA', ],
        _new_category=ivd_label,
        _complete_new_category='In Vitro Diagnostics (IVD)'
    )

    # Print the new FDA specialty distribution over the EMDN categories
    _ = build_confusion_matrix(
        input_dataframe=emdn_gmdn_fda_df_selected,
        normalize_over_rows=True,
        file_name='emdn_category_given_specialty_selected',
        title='EMDN Category Probabilities given FDA Medical Specialties'
    )

    # Save the cleaned dataset to a new CSV file
    emdn_gmdn_fda_df_selected.to_csv(
        EMDN_GMDN_FDA_CLEANED_FILE_PATH,
        index=False
    )


def replace_specialty_labels(
        _df: pd.DataFrame,
        _replaced_categories: list[str],
        _new_category: str,
        _complete_new_category: str
) -> pd.DataFrame:
    """
    This method replaces the specified categories in the 'medical_specialty' column with a new category. Then, the
    'medical_specialty_complete' column is updated with the new complete category name.

    :param _df: The dataframe to be updated.
    :param _replaced_categories: The categories to be replaced.
    :param _new_category: The new category to be used.
    :param _complete_new_category: The new complete category name to be used.

    :return: The updated dataframe.
    """
    _df = _df.copy()
    # Replace in the selected_rows_df all the categories in 'other_categories' with 'OT'
    _df['medical_specialty'] = _df[
        'medical_specialty'
    ].replace(
        _replaced_categories,
        [_new_category] * len(_replaced_categories)
    )
    # Set in 'medical_specialty_complete', where 'medical_specialty' is other_category_label, the 'Other' label
    _df.loc[
        _df['medical_specialty'] == _new_category,
        'medical_specialty_complete'
    ] = _complete_new_category

    return _df
