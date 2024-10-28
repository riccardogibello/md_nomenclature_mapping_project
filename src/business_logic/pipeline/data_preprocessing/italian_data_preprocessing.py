import json
import os
from typing import Any, List

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.business_logic.database_interfaces.connection_handler import DatabaseInformation, ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.country_data_handler import CountryDataHandler
from src.business_logic.pipeline.data_preprocessing.data_cleaning import clean_md_catalogue_code, \
    clean_company_name, clean_string, clean_extra_blank_spaces
from src.business_logic.pipeline.data_preprocessing.dataframe_column_names import *
from src.business_logic.pipeline.data_preprocessing.store_batched_data import store_batched_italian_data
from src.business_logic.utilities.concurrentifier import perform_parallelization
from src.business_logic.utilities.dataframe_handling import clean_dataframe_from_null_values
from src.data_model.enums import CountryName
from src.__constants import CODE_NOT_SET
from src.__file_paths import NEW_ITALIAN_FULL_CSV_PATH


def _clean_italian_rows_subset(
        row_subset_indexes: list[list[Any]],
        italy_entity_id: int,
) -> str:
    """
    This method cleans and returns a set of rows, given in row_subset_list, in which the columns'
    values are replaced with the cleaned ones.

    :param row_subset_indexes: A list of rows from the 'italian_medical_device_full_list.csv' file to be cleaned.
    :param italy_entity_id: The id of the Italy entity in the database.

    :return: A string containing the JSON representation of the cleaned dataframe.
    """
    # Load the italian dataframe from the file containing the selected columns
    # to be inserted in the database
    italian_dataframe = pd.read_csv(
        NEW_ITALIAN_FULL_CSV_PATH,
        sep=',',
    )
    # Keep the rows corresponding to the given indexes, that are the ones
    # that the current process must add to the database
    rows_subset = italian_dataframe.iloc[row_subset_indexes].to_numpy()

    # Prepare the list of columns for the aggregated data of medical devices
    italian_dataframe_columns = ITALIAN_DATAFRAME_COLUMNS
    # Get an empty list of rows for the dataframe
    italian_dataframe_rows = get_italian_empty_rows()

    # For each row to be cleaned
    for current_row in rows_subset:
        # Get the catalogue code of the medical device
        catalogue_code: str = '' if current_row[1] == np.nan else str(current_row[1])
        # Clean the catalogue code
        catalogue_codes: list[str] = clean_md_catalogue_code(
            clean_string(catalogue_code)
        )
        if len(catalogue_codes) == 0:
            continue

        # Get the name of the company and clean it
        original_company_name = '' if current_row[0] == np.nan else str(current_row[0])
        current_company_name = clean_company_name(original_company_name)

        # Get the EMDN code of the medical device
        emdn_code: Optional[str] = str(current_row[3])
        # Clean the EMDN code
        emdn_code = emdn_code.replace(' ', '')
        if emdn_code == CODE_NOT_SET:
            emdn_code = None

        # Get the name of the device and clean it
        original_device_name = '' if current_row[2] == np.nan else str(current_row[2])
        clean_device_name = clean_extra_blank_spaces(
            clean_string(original_device_name)
        ).upper()

        # If either the company name or the device name have not been given
        if original_company_name == '' or original_device_name == '':
            # Skip the current medical device
            continue
        else:
            # For each of the given catalogue codes
            for _catalogue_code in catalogue_codes:
                # Add the information to the list of rows for the dataframe
                italian_dataframe_rows.append(
                    (
                        original_company_name,
                        current_company_name,
                        None,
                        italy_entity_id,
                        None,
                        original_device_name,
                        clean_device_name,
                        None,
                        _catalogue_code,
                        'MD',
                        None,
                        None,
                        None,
                        None,
                        emdn_code,
                    )
                )

    # Create the dataframe containing the cleaned data
    italian_dataframe = pd.DataFrame(
        data=italian_dataframe_rows,
        columns=italian_dataframe_columns
    )

    # Return the string of the JSON representation of the dataframe
    return json.dumps(italian_dataframe.to_json())


def load_italian_medical_device_data(
        old_csv_file_path: str,
        database_information: DatabaseInformation,
) -> None:
    """
    This method loads from the file of Italian medical devices, the data of the medical devices, companies and
    EMDN code mappings, and stores them in the database.

    It cleans this dataframe by removing some unused columns and by splitting the 'CODICE_CATALOGO_FABBR_ASS' when
    it is written in unusual manner (such as, 'code1; code2' or 'DA code1 A code2).

    NOTE: This type of cleaning, due to human errors and conventions during the creation of this dataframe, should
    be surrounded by a previous manual inspection of the data, in order correct some parts of these codes
    that can't be easily detected and corrected by src.

    :param old_csv_file_path: The file containing all the Italian medical devices.
    :param database_information: The information to connect to the database.
    """
    # Build the connection handler to connect to the database
    connection_handler = ConnectionHandler(
        database_information
    )
    # Instantiate the handler to add the Italy as a country, if not already added
    country_data_handler = CountryDataHandler(
        connection_handler=connection_handler,
        init_cache=True,
    )
    italy_entity = country_data_handler.add_country(
        country_name=CountryName.ITALY.value,
        is_american_state=False,
    )

    # Load the dataframe of the italian medical devices
    old_md_file_dataframe: DataFrame = pd.read_csv(
        old_csv_file_path,
        sep=';',
        encoding='unicode_escape',
    )
    column_indexes = [
        8,  # manufacturer name
        11,  # catalogue code
        12,  # device name
        13,  # emdn code
    ]
    # Keep only the columns of interest
    old_md_file_dataframe = old_md_file_dataframe.iloc[:, column_indexes]
    # Rename the columns
    old_md_file_dataframe.columns = [
        'MANUFACTURER_NAME',
        'CATALOGUE_CODE',
        'DEVICE_NAME',
        'EMDN_CODE',
    ]

    # Fill the missing values of the EMDN code with a placeholder
    old_md_file_dataframe['EMDN_CODE'] = old_md_file_dataframe['EMDN_CODE'].fillna('CODE_NOT_SET')
    # Drop every row which has at least one missing value for the company name, device name or catalogue code
    # and reset the index
    clean_dataframe_from_null_values(old_md_file_dataframe)

    # Store the new cleaned italian file to be loaded by the other processes
    old_md_file_dataframe.to_csv(
        NEW_ITALIAN_FULL_CSV_PATH,
        index=False,
    )
    # Keep memory of the size of the dataframe to build the list of indexes
    # for the processes
    max_dataframe_index = len(old_md_file_dataframe)
    del old_md_file_dataframe

    # Perform the cleaning of the italian dataset in parallel
    cleaned_dataframe_strings: List[str] = perform_parallelization(
        input_sized_element=np.arange(0, max_dataframe_index),
        processing_batch_callback=_clean_italian_rows_subset,
        additional_batch_callback_parameters=[
            italy_entity.identifier,
        ],
    )

    store_batched_italian_data(
        cleaned_dataframe_strings,
        connection_handler,
    )

    # Delete the cleaned italian dataframe file
    os.remove(NEW_ITALIAN_FULL_CSV_PATH)
