import os
from typing import Optional, Tuple

import pandas as pd

from src.business_logic.database_interfaces.connection_handler import DatabaseInformation, local_database_information, \
    ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.country_data_handler import CountryDataHandler
from src.business_logic.database_interfaces.data_handlers.device_data_handler.clean_device_data_handler import \
    CleanDeviceDataHandler
from src.business_logic.database_interfaces.data_handlers.manufacturer_data_handler.clean_manufacturer_data_handler import \
    CleanManufacturerDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.emdn_correspondence_data_handler import \
    EmdnCorrespondenceDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.gmdn_correspondence_data_handler import \
    GmdnCorrespondenceDataHandler
from src.business_logic.pipeline.data_preprocessing.dataframe_column_names import DataFrameColumnName
from src.business_logic.pipeline.market_cross_checking.final_refactoring_functions import \
    store_device_and_nomenclature_correspondences
from src.business_logic.pipeline.market_cross_checking.similarity_merge_function import join_approximated
from src.business_logic.utilities.os_utilities import create_directory
from src.data_model.enums import CountryName
from src.__directory_paths import OUTPUT_DATA_DIRECTORY_PATH


def _build_device_datasets(
        connection_handler: ConnectionHandler,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This method loads all the medical device, company, and nomenclature data from the database and builds two
    dataframes: one for the American market and one for the European market.

    :param connection_handler: The connection handler to the database.

    :return: A tuple containing the two dataframes, the first one for the American market and the second one for the
    European market.
    """
    # Prepare a dataset of all the American medical device data
    # (i.e., COMPANY_ID, COMPANY_NAME, DEVICE_ID, DEVICE_NAME, GMDN_ID)
    american_dataframe: pd.DataFrame

    # Prepare a dataset of all the European medical device data
    # (i.e., COMPANY_ID, COMPANY_NAME, DEVICE_ID, DEVICE_NAME, EMDN_ID)
    european_dataframe: pd.DataFrame

    # Instantiate the EMDN and GMDN device mapping handlers
    emdn_correspondence_data_handler = EmdnCorrespondenceDataHandler(
        connection_handler=connection_handler,
        init_cache=True,
    )
    gmdn_correspondence_data_handler = GmdnCorrespondenceDataHandler(
        connection_handler=connection_handler,
        init_cache=True,
    )
    # Instantiate the handlers for the manufacturers and the devices
    clean_manufacturer_data_handler = CleanManufacturerDataHandler(
        connection_handler=connection_handler,
        init_cache=True,
    )
    clean_device_data_handler = CleanDeviceDataHandler(
        connection_handler=connection_handler,
        init_cache=True,
    )
    # Instantiate the handler for the countries
    country_data_handler = CountryDataHandler(
        connection_handler=connection_handler,
        init_cache=True,
    )
    italy = country_data_handler.get_country(
        country_name=CountryName.ITALY.value,
        is_american_state=False,
    )
    usa = country_data_handler.get_country(
        country_name=CountryName.USA.value,
        is_american_state=False,
    )
    american_row_list = []
    european_row_list = []
    # For each manufacturer
    manufacturer_list = clean_manufacturer_data_handler.get_manufacturers(
        perform_fetch_on_database=False,
    )
    # Get all the devices related to the manufacturers
    devices = clean_device_data_handler.get_devices_by_ids(
        perform_fetch_on_database=False,
    )
    device_manufacturer_device_ids = {}
    # Get all the EMDN and GMDN correspondences
    emdn_correspondences = emdn_correspondence_data_handler.get_emdn_device_correspondences(
        fetch_from_database=False,
    )
    # Index the correspondences by device id
    emdn_correspondences = {
        correspondence.clean_device_id: correspondence
        for correspondence in emdn_correspondences.values()
    }
    gmdn_correspondences = gmdn_correspondence_data_handler.get_gmdn_correspondences(
        fetch_from_database=False,
    )
    # Index the correspondences by device id
    gmdn_correspondences = {
        correspondence.clean_device_id: correspondence
        for correspondence in gmdn_correspondences.values()
    }
    for device in devices.values():
        if device.manufacturer_id not in device_manufacturer_device_ids.keys():
            device_manufacturer_device_ids[device.manufacturer_id] = []
        device_manufacturer_device_ids[device.manufacturer_id].append(device.identifier)

    for manufacturer in manufacturer_list.values():
        # Get all the devices related to the given manufacturer
        manufacturer_device_ids: list[int] = device_manufacturer_device_ids[manufacturer.identifier]
        manufacturer_devices = [
            devices[device_id]
            for device_id in manufacturer_device_ids
        ]
        # Get the state from which the information comes
        state_id = manufacturer.original_state_identifier

        for device in manufacturer_devices:
            # Get the EMDN classification, if any, for the given device identifier
            emdn_correspondence = emdn_correspondences.get(device.identifier, None)
            # Get the GMDN classification, if any, for the given device identifier
            gmdn_correspondence = gmdn_correspondences.get(device.identifier, None)

            # If the information is from Italy
            if state_id == italy.identifier:
                # Build the current row and add it to the list of rows
                current_row = [
                    int(manufacturer.identifier),
                    manufacturer.clean_name,
                    int(device.identifier),
                    device.clean_name,
                    device.catalogue_code,
                    int(emdn_correspondence.code_id) if emdn_correspondence is not None else None,
                ]
                european_row_list.append(current_row)
            if state_id == usa.identifier:
                # Build the current row
                current_row = [
                    int(manufacturer.identifier),
                    manufacturer.clean_name,
                    int(device.identifier),
                    device.clean_name,
                    device.catalogue_code,
                    int(gmdn_correspondence.code_id) if gmdn_correspondence is not None else None,
                ]
                american_row_list.append(current_row)

    american_columns = pd.Series([
        DataFrameColumnName.CLEAN_MANUFACTURER_ID_COLUMN.value,
        DataFrameColumnName.MANUFACTURER_CLEAN_NAME_COLUMN.value,
        DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value,
        DataFrameColumnName.DEVICE_CLEAN_NAME_COLUMN.value,
        DataFrameColumnName.CATALOGUE_CODE_COLUMN.value,
        DataFrameColumnName.GMDN_ID_COLUMN.value,
    ])
    american_dataframe = pd.DataFrame(
        columns=american_columns,
        data=american_row_list
    )
    european_columns = pd.Series([
        DataFrameColumnName.CLEAN_MANUFACTURER_ID_COLUMN.value,
        DataFrameColumnName.MANUFACTURER_CLEAN_NAME_COLUMN.value,
        DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value,
        DataFrameColumnName.DEVICE_CLEAN_NAME_COLUMN.value,
        DataFrameColumnName.CATALOGUE_CODE_COLUMN.value,
        DataFrameColumnName.EMDN_ID_COLUMN.value,
    ])
    european_dataframe = pd.DataFrame(
        columns=european_columns,
        data=european_row_list
    )

    return american_dataframe, european_dataframe


def cross_check_markets(
        company_name_threshold: Optional[int] = 85,
        database_information: DatabaseInformation = local_database_information,
) -> None:
    """
    This method computes the cross-checking between the American and European medical device markets. It is possible to
    set a custom threshold to find correspondences between company names in the two markets. In the end, all the device
    and nomenclature correspondences are stored in the database.

    :param company_name_threshold: The threshold for the similarity between the company names. Set to 85 by default.
    :param database_information: The information about the database to connect to. Set to the local database by default.
    """
    # Create the base directory path for the results of the current experiment
    output_path = OUTPUT_DATA_DIRECTORY_PATH + 'MappingStatsCn' + str(company_name_threshold) + '/'
    if not os.path.isdir(output_path):
        create_directory(
            directory_path=output_path,
        )
    connection_handler = ConnectionHandler(
        database_information=database_information
    )
    # Get the American and European datasets
    american_dataframe, european_dataframe = _build_device_datasets(
        connection_handler=connection_handler,
    )
    # Merge the two dataframes based on the similarity of the company names
    joined_dataframe = join_approximated(
        second_dataframe=american_dataframe,
        first_dataframe=european_dataframe,
        column_name=DataFrameColumnName.MANUFACTURER_CLEAN_NAME_COLUMN.value,
        threshold=company_name_threshold,
    )
    # Store the final resulting matches between the American and European data in the proper tables
    store_device_and_nomenclature_correspondences(
        dataframe=joined_dataframe,
        connection_handler=connection_handler,
        base_output_path=output_path,
    )
