import os

import pandas as pd

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler, DatabaseInformation
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.emdn_code_data_handler import \
    EmdnCodeDataHandler
from src.business_logic.utilities.concurrentifier import perform_parallelization
from src.business_logic.utilities.dataframe_handling import clean_dataframe_from_null_values


def _load_and_clean_emdn_dataframe(
        csv_emdn_file_path: str
) -> str:
    """
    This method loads the given CSV file containing all the english EMDN codes with their descriptions.
    It drops some useless columns and refactors the column names. The cleaned dataframe is saved in the same directory
    as the given CSV file.

    :param csv_emdn_file_path: the path to the CSV file containing the EMDN codes.

    :return: The path to the cleaned CSV file.
    """
    # Load the CSV file containing the EMDN codes
    emdn_dataframe = pd.read_csv(
        csv_emdn_file_path,
        sep=';'
    )
    column_indexes_to_keep = [
        1,  # category
        2,  # code
        5,  # bottom level
        -3,  # category description
    ]
    emdn_dataframe = emdn_dataframe.iloc[:, column_indexes_to_keep]
    # Rename the columns
    emdn_dataframe.columns = [
        "CATEGORY",
        "CODE",
        "BOTTOM LEVEL",
        "CATEGORY DESCRIPTION"
    ]

    # Clean any row that contains at least one null value
    clean_dataframe_from_null_values(
        emdn_dataframe
    )

    # Save the dataframe in the same directory as the given CSV file
    cleaned_file_path = csv_emdn_file_path.replace(
        '.csv',
        '_cleaned.csv'
    )
    emdn_dataframe.to_csv(
        cleaned_file_path,
        index=False
    )

    return cleaned_file_path


def _load_emdn_codes_into_database(
        emdn_dataframe: pd.DataFrame,
        database_information: DatabaseInformation,
) -> None:
    """
    This method extracts from the given dataframe all the EMDN codes, storing them in a database table.

    :param emdn_dataframe: The dataframe containing the EMDN codes.
    :param database_information: The information needed to connect to the database.
    """
    # Instantiate the handler of EMDN codes
    emdn_code_data_handler = EmdnCodeDataHandler(
        ConnectionHandler(
            database_information=database_information
        )
    )

    # for each row of the dataframe containing the EMDN data
    for index, row in emdn_dataframe.iterrows():
        # Get the EMDN code description
        description: str = row.iloc[-1]
        # Get the alphanumeric EMDN code
        code = row.iloc[1]

        # keep the information about being or not a leaf in the EMDN tree
        if row.iloc[-2] == 'NO':
            is_leaf = False
        else:
            is_leaf = True

        # Add the EMDN code to the database and cache, which is structured as a tree to keep track
        # of the hierarchy information
        emdn_code_data_handler.add_code(
            emdn_code_string=code,
            emdn_description=description,
            is_leaf=is_leaf,
        )


def clean_and_store_emdn_codes(
        csv_emdn_file_path: str,
        database_information: DatabaseInformation,
) -> None:
    """
    This method loads and cleans the given CSV file containing all the english EMDN codes with their descriptions. It
    stores all the EMDN codes in the database, parallelizing the process.

    :param csv_emdn_file_path: The path to the CSV file containing the EMDN codes.
    :param database_information: The information needed to connect to the database.
    """
    # Load the EMDN CSV file and clean it
    cleaned_dataframe_path = _load_and_clean_emdn_dataframe(
        csv_emdn_file_path=csv_emdn_file_path
    )

    # Load the EMDN cleaned dataframe
    emdn_dataframe = pd.read_csv(cleaned_dataframe_path)

    perform_parallelization(
        input_sized_element=emdn_dataframe,
        processing_batch_callback=_load_emdn_codes_into_database,
        additional_batch_callback_parameters=[
            database_information
        ],
    )

    # Remove the EMDN codes file
    os.remove(cleaned_dataframe_path)
