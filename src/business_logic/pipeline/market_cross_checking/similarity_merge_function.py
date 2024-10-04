import json
import re
from typing import Any, Callable, Optional

import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz

from src.business_logic.pipeline.data_preprocessing.dataframe_column_names import DataFrameColumnName
from src.business_logic.utilities.concurrentifier import perform_parallelization


def _build_column_value_row_indexes_map(
        dataframe: pd.DataFrame,
        column_name: str,
) -> dict[str, np.array]:
    """
    This method receives as input a dataframe and a column name and returns a map that contains the correspondence
    between each value of the column and the related indexes in the dataframe where that value can be found.

    :param dataframe: The dataframe from which the dictionary must be extracted.
    :param column_name: The name of the column of the dataframe from which the values must be extracted and indexed.

    :return: A dictionary that contains the correspondence between each value of the column and the related indexes in
    the dataframe where that value can be found.
    """
    tmp_column_indexes_map: dict[str, np.array] = {}
    index = 0
    for first_dataframe_value in dataframe[column_name].to_numpy():
        if first_dataframe_value is np.nan:
            pass
        else:
            first_dataframe_value = str(first_dataframe_value)
            if first_dataframe_value not in tmp_column_indexes_map:
                tmp_column_indexes_map[first_dataframe_value] = np.array([], dtype=np.int64)
            # concatenate to the numpy array the index of the first dataframe value
            tmp_column_indexes_map[first_dataframe_value] = np.concatenate(
                [tmp_column_indexes_map[first_dataframe_value], np.array([index])],
                axis=0,
            )
        index += 1

    return tmp_column_indexes_map


def catalogue_code_similarity(
        first_code: str,
        second_code: str,
) -> bool:
    """
    This method evaluates whether two catalogue codes of medical devices correspond to the same entity.

    :param first_code: The first code to be compared.
    :param second_code: The second code to be compared.

    :return: True, if the two codes correspond to the same entity. False, if both the codes do (not) contain the
    wildcard or do not match to the same code.
    """
    try:
        if first_code == second_code:
            return True
        else:
            both_without_wildcard = not first_code.__contains__('*') and not second_code.__contains__('*')
            both_with_wildcard = first_code.__contains__('*') and second_code.__contains__('*')
            # If the two codes do not contain the wildcard or both contain it, return None
            if both_without_wildcard or both_with_wildcard:
                return False
            else:
                if first_code.__contains__('*'):
                    wildcard_code = first_code
                    code = second_code
                else:
                    wildcard_code = second_code
                    code = first_code

                # Escape in the wildcard code the '+' and '.' symbols
                pattern_wildcard_code = wildcard_code
                # Split the code with the wildcard in all the subsequences
                pattern_split_list = pattern_wildcard_code.split('*')
                # Create a pattern string to match any subsequence of the wildcard code, plus any character in between
                pattern = '.*'.join(pattern_split_list)

                # create the replace string
                replace_split_list = wildcard_code.split('*')
                # Create the replacing string, which is the wildcard code (without asterisks)
                replace = ''.join(replace_split_list)

                # Try to replace in the code all the sequence with the built pattern of the wildcard code
                # (without asterisk)
                cleaned_catalogue_code = re.sub(pattern=pattern, repl=replace, string=code)
                # If it is possible to match the wildcard code with the code, return the code without the wildcard
                if cleaned_catalogue_code == wildcard_code.replace('*', ''):
                    return True
                else:
                    return False
    except Exception as e:
        print(f"Error in the catalogue code similarity function: {e}")
        return False


def _find_similar_couples(
        first_sub_dataframe: pd.DataFrame,
        second_sub_dataframe: pd.DataFrame,
        column_name: str,
        threshold: int,
        additional_filtering_callback: Optional[
            Callable[[str, str], bool]
        ] = catalogue_code_similarity,
) -> str:
    """
    This method receives as input two dataframes, one for each medical device market. Firstly, an approximate join is
    performed between the two dataframes based on the similarity of the values in the column specified by the
    column_name parameter. Then, for all the corresponding couples, the matching catalogue codes of medical devices
    are kept and stored in the returned dataframe.

    :param first_sub_dataframe: The first dataframe to join.
    :param second_sub_dataframe: The second dataframe to join.
    :param column_name: The name of the column to use for the approximated join.
    :param threshold: The similarity threshold to consider two strings as similar.
    :param additional_filtering_callback: The callback to evaluate the similarity between the catalogue codes of the
    medical devices. The default value is the catalogue_code_similarity function.

    :return: A dataframe containing the joined dataframes on the column specified by the column_name parameter and
    on the catalogue codes of the medical devices.
    """
    # Create a map that contains the correspondence between each value of the second dataframe and the related indexes
    # in the second dataframe
    second_dataframe_values_map: dict[str, np.ndarray] = _build_column_value_row_indexes_map(
        second_sub_dataframe,
        column_name,
    )
    # Create a map that contains the correspondence between each value of the first dataframe and the related indexes
    # in the first dataframe
    first_dataframe_values_map: dict[str, np.ndarray] = _build_column_value_row_indexes_map(
        first_sub_dataframe,
        column_name,
    )
    # Extract a list of all the strings of the first dataframe in the considered column
    first_string_list = list(set(first_dataframe_values_map.keys()))
    # Extract a list of all the strings of the second dataframe in the considered column
    second_string_list = list(set(second_dataframe_values_map.keys()))

    correspondences: list[tuple[str, str, float]] = []
    # For each value of the first dataframe
    for first_dataframe_value in first_string_list:
        # Compute the similarity between the first dataframe value and each value of the second dataframe, taking
        # back a list of (second_dataframe_value, similarity) tuples
        similarities = process.extract(
            first_dataframe_value,
            second_string_list,
            limit=2,
            scorer=fuzz.ratio
        )
        # For each tuple of the list of similarities
        for similarity in similarities:
            # Get the value of the second dataframe
            second_dataframe_value = similarity[0]
            # Get the similarity score
            similarity_score = similarity[1]
            # If the similarity score is higher than the threshold
            if similarity_score >= threshold:
                # Append the correspondence between the first dataframe value, the second dataframe value and the
                # similarity score to the list of correspondences
                correspondences.append((first_dataframe_value, second_dataframe_value, similarity_score))
            del similarity_score
            del second_dataframe_value
        del similarities
    del threshold
    del first_string_list
    del second_string_list

    new_rows = []
    # For each couple of strings that are similar between the first and the second dataframe
    for corresponding_value in correspondences:
        # Get the string from the first dataframe
        first_dataframe_value: str = corresponding_value[0]
        # Get the list of indexes of the first dataframe where the current value can be found
        first_indexes: np.ndarray = first_dataframe_values_map[first_dataframe_value]
        # Get the string from the second dataframe
        second_dataframe_value: str = corresponding_value[1]
        # Get the list of indexes of the second dataframe where the current value can be found
        second_indexes: np.ndarray = second_dataframe_values_map[second_dataframe_value]
        # Get all the rows of the first dataframe given the indexes and drop the duplicates
        first_dataframe_rows: pd.DataFrame = first_sub_dataframe.iloc[first_indexes].drop_duplicates()
        # Find the index of the column of the catalogue code in the first dataframe
        fdf_index_catalogue_code = first_sub_dataframe.columns.get_loc(
            DataFrameColumnName.CATALOGUE_CODE_COLUMN.value
        )
        # Find the index of the column of the device identifier in the first dataframe
        fdf_index_device_id = first_sub_dataframe.columns.get_loc(
            DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value
        )
        # Get all the rows of the second dataframe given the indexes
        second_dataframe_rows: pd.DataFrame = second_sub_dataframe.iloc[second_indexes].drop_duplicates()
        # Find the index of the column of the catalogue code in the second dataframe
        sdf_index_catalogue_code = second_sub_dataframe.columns.get_loc(
            DataFrameColumnName.CATALOGUE_CODE_COLUMN.value
        )
        # Find the index of the column of the device identifier in the second dataframe
        sdf_index_device_id = second_sub_dataframe.columns.get_loc(
            DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value
        )

        # Iterate over the rows of the first dataframe
        for first_dataframe_row in first_dataframe_rows.to_numpy():
            # Get the catalogue code of the first dataframe row
            first_dataframe_catalogue_code = first_dataframe_row[fdf_index_catalogue_code]
            # Iterate over the rows of the second dataframe
            for second_dataframe_row in second_dataframe_rows.to_numpy():
                # Get the catalogue code of the second dataframe row
                second_dataframe_catalogue_code = second_dataframe_row[sdf_index_catalogue_code]
                # If the catalogue code can be considered as equal
                if additional_filtering_callback(first_dataframe_catalogue_code, second_dataframe_catalogue_code):
                    # Get the similarity score between the two company strings
                    similarity_score: int = int(corresponding_value[2])
                    # Store the relation between the two device ids and the similarity score between the
                    # two related company names
                    tmp_row = [
                        # Set the device identifier of the first dataframe
                        first_dataframe_row[fdf_index_device_id],
                        # Set the device identifier of the second dataframe
                        second_dataframe_row[sdf_index_device_id],
                        # Set the similarity score between the two company names
                        similarity_score,
                        # This is set since there is no similarity evaluation between the two device catalogue
                        # codes
                        100
                    ]
                    new_rows.append(tmp_row)
                    del similarity_score
                    del tmp_row
                del second_dataframe_catalogue_code
            del first_dataframe_catalogue_code
        del first_dataframe_rows
        del second_dataframe_rows
        del first_dataframe_value
        del second_dataframe_value
        del first_indexes
        del second_indexes
        del fdf_index_catalogue_code
        del fdf_index_device_id
        del sdf_index_catalogue_code
        del sdf_index_device_id
    del correspondences
    del first_dataframe_values_map
    del second_dataframe_values_map
    del column_name
    del first_sub_dataframe
    del second_sub_dataframe

    # Return a dataframe which will be made
    return json.dumps(
        pd.DataFrame(
            data=new_rows,
            columns=[
                DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value + '_DF1',
                DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value + '_DF2',
                'COMPANY_NAME_SIMILARITY',
                'SIMILARITY',
            ],
        ).to_json()
    )


def join_approximated(
        first_dataframe: pd.DataFrame,
        second_dataframe: pd.DataFrame,
        column_name: str,
        threshold: Any,
) -> pd.DataFrame:
    """
    This method receives as input two dataframes and joins them based on the similarity of the values in the column
    specified by the column_name parameter. The process is carried out in parallel between different processes
    through the _find_similar_couples method.

    :param first_dataframe: The first dataframe to join.
    :param second_dataframe: The second dataframe to join.
    :param column_name: The name of the column to use for the approximated join.
    :param threshold: The minimum similarity value between the values in the column to consider them as similar.

    :return: A dataframe that contains the joined data between the two dataframes.
    """
    shrunk_first_dataframe = first_dataframe[[
        column_name,
        DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value,
        DataFrameColumnName.CATALOGUE_CODE_COLUMN.value,
    ]].drop_duplicates()
    shrunk_second_dataframe: pd.DataFrame = second_dataframe[[
        column_name,
        DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value,
        DataFrameColumnName.CATALOGUE_CODE_COLUMN.value,
    ]].drop_duplicates()

    # Perform the cross-match in parallel between the first and the second dataframes
    joined_dataframe_strings = perform_parallelization(
        input_sized_element=shrunk_first_dataframe,
        processing_batch_callback=_find_similar_couples,
        additional_batch_callback_parameters=[
            shrunk_second_dataframe,
            column_name,
            threshold,
        ],
        threads_number=1,
        processes_number=6,
    )
    assert joined_dataframe_strings is not None and len(joined_dataframe_strings) > 0, "The joined dataframe is empty."
    # Load all the returned dataframes from the strings
    joined_dataframes = [
        pd.read_json(joined_dataframe_string) for joined_dataframe_string in joined_dataframe_strings
    ]
    # Create a single dataframe from the list of dataframes
    joined_dataframe = pd.concat(joined_dataframes)
    # Join the dataframe and the first dataframe on the device identifier column
    joined_dataframe = joined_dataframe.join(
        first_dataframe.set_index(DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value),
        on=DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value + '_DF1',
        lsuffix='_DF_IT',
    )
    # Join the dataframe and the second dataframe on the device identifier column
    joined_dataframe = joined_dataframe.join(
        second_dataframe.set_index(DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value),
        on=DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value + '_DF2',
        lsuffix='_DF_US',
    )

    returned_df = joined_dataframe[[
        DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value + '_DF1',
        DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value + '_DF2',
        DataFrameColumnName.EMDN_ID_COLUMN.value,
        DataFrameColumnName.GMDN_ID_COLUMN.value,
        'COMPANY_NAME_SIMILARITY',
        'SIMILARITY',
    ]]
    # Change any NaN value to 0 in the similarity columns
    returned_df.loc[:, 'COMPANY_NAME_SIMILARITY'] = returned_df['COMPANY_NAME_SIMILARITY'].fillna(0)
    returned_df.loc[:, 'SIMILARITY'] = returned_df['SIMILARITY'].fillna(0)
    # Drop any row that has a NaN value
    returned_df = returned_df.dropna()
    # Cast the first 4 columns to integers and the last 2 columns to floats
    returned_df[[
        DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value + '_DF1',
        DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value + '_DF2',
        DataFrameColumnName.EMDN_ID_COLUMN.value,
        DataFrameColumnName.GMDN_ID_COLUMN.value,
    ]] = returned_df[[
        DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value + '_DF1',
        DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value + '_DF2',
        DataFrameColumnName.EMDN_ID_COLUMN.value,
        DataFrameColumnName.GMDN_ID_COLUMN.value,
    ]].astype(int)
    returned_df[[
        'COMPANY_NAME_SIMILARITY',
        'SIMILARITY',
    ]] = returned_df[[
        'COMPANY_NAME_SIMILARITY',
        'SIMILARITY',
    ]].astype(float)
    # Drop the duplicates from the dataframe
    returned_df = returned_df.drop_duplicates()
    # Reset the indexes of the dataframe
    returned_df = returned_df.reset_index(drop=True)

    # Concatenate all the dataframes of the list
    return returned_df
