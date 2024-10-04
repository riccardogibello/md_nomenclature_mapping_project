from typing import Callable, Any

import numpy as np
import pandas as pd

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.device_data_handler.clean_device_data_handler import \
    CleanDeviceDataHandler
from src.business_logic.database_interfaces.data_handlers.manufacturer_data_handler.clean_manufacturer_data_handler import \
    CleanManufacturerDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.emdn_code_data_handler import \
    EmdnCodeDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.emdn_correspondence_data_handler import \
    EmdnCorrespondenceDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.gmdn_code_data_handler import \
    GmdnCodeDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.gmdn_correspondence_data_handler import \
    GmdnCorrespondenceDataHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent, \
    AbstractTableHandler
from src.business_logic.pipeline.data_preprocessing.dataframe_column_names import *
from src.business_logic.utilities.dataframe_handling import get_row_value
from src.data_model.devices.clean_device import CleanDevice
from src.data_model.manufacturers.clean_manufacturer import CleanManufacturer
from src.data_model.nomenclature_codes.emdn_correspondence import EmdnCorrespondence
from src.data_model.nomenclature_codes.gmdn_code import GmdnCode
from src.data_model.nomenclature_codes.gmdn_correspondence import GmdnCorrespondence
from src.data_model.enums import MatchType


def _add_correspondences(
        correspondence_table_handler: AbstractTableHandler,
        instance_builder: Callable[[list[Any]], Any],
        clean_dataframe: pd.DataFrame,
        first_column_name: str,
        second_column_name: str,
):
    if len(clean_dataframe) == 0:
        return
    else:
        correspondence_table_handler.add(
            filters=[
                [
                    SqlQueryComponent(
                        column_name=first_column_name,
                        column_values=[int((row[0])) if not pd.isna(row[0]) else None],
                    ),
                    SqlQueryComponent(
                        column_name=second_column_name,
                        column_values=[int((row[1])) if not pd.isna(row[1]) else None],
                    ),
                    SqlQueryComponent(
                        column_name=TableColumnName.MATCH_TYPE.value,
                        column_values=[str(MatchType.EXACT.value)],
                    ),
                    SqlQueryComponent(
                        column_name=TableColumnName.SIMILARITY.value,
                        column_values=[None]
                    ),
                    SqlQueryComponent(
                        column_name=TableColumnName.MATCHED_NAME.value,
                        column_values=[None]
                    )
                ] for row in clean_dataframe.to_numpy()
            ],
            instance_builder=instance_builder,
            pk_column_names=[
                first_column_name,
                second_column_name,
            ]
        )


def _store_batched_data(
        dataframe_strings: list[str],
        connection_handler: ConnectionHandler,
) -> pd.DataFrame:
    cleaned_dataframes = []
    for cleaned_dataframe_string in dataframe_strings:
        # Load the JSON string into a dataframe
        cleaned_dataframe = pd.read_json(
            cleaned_dataframe_string
        )
        cleaned_dataframes.append(
            cleaned_dataframe
        )

    # Concatenate all the dataframes retrieved from the processes
    cleaned_dataframe = pd.concat(
        cleaned_dataframes,
    )
    # Replace in the "standardized_manufacturer_id" column the NaN values with None
    cleaned_dataframe[DataFrameColumnName.STANDARDIZED_MANUFACTURER_ID_COLUMN.value] = cleaned_dataframe[
        DataFrameColumnName.STANDARDIZED_MANUFACTURER_ID_COLUMN.value
    ].apply(
        lambda x: None if pd.isna(x) else x
    )

    # Add in a batched mode all the manufacturers to the database
    manufacturer_data_handler = CleanManufacturerDataHandler(
        connection_handler=connection_handler
    )
    filters = []
    key_indexes: dict[str, list[int]] = {}
    index = 0
    # Iterate over all the rows in the cleaned dataframe
    for row in cleaned_dataframe.to_numpy():
        # Get all the manufacturer data
        (
            manufacturer_name,
            clean_name,
            standardized_manufacturer_id,
            original_manufacturer_state_id
        ) = get_row_value(
            row=row,
            column_name=[
                0,
                1,
                2,
                3,
            ],
            casting_type=[
                str,
                str,
                int,
                int,
            ]
        )
        # If one of the PK values is None, skip the row
        if None not in [manufacturer_name, original_manufacturer_state_id]:
            # Compute the key related to the current manufacturer, made of all the database PK components
            key = '$'.join(
                [manufacturer_name, str(original_manufacturer_state_id)]
            )
            # If the key is not in the dictionary, create a new list, otherwise get the list
            if key not in key_indexes.keys():
                # This means that the manufacturer was not found before, then add its filter to the
                # filters list
                filters.append(
                    [
                        SqlQueryComponent(
                            column_name=TableColumnName.NAME.value,
                            column_values=[manufacturer_name],
                        ),
                        SqlQueryComponent(
                            column_name=TableColumnName.CLEAN_NAME.value,
                            column_values=[clean_name],
                        ),
                        SqlQueryComponent(
                            column_name=TableColumnName.STANDARDIZED_MANUFACTURER_ID.value,
                            column_values=[standardized_manufacturer_id],
                        ),
                        SqlQueryComponent(
                            column_name=TableColumnName.ORIGINAL_STATE_ID.value,
                            column_values=[original_manufacturer_state_id],
                        )
                    ]
                )
                # Create a new list of indexes for the given manufacturer
                tmp_list = []
            else:
                # Get the old list of indexes
                tmp_list = key_indexes[key]
            # Replace the list in the dictionary with the one with the new row index appended
            tmp_list.append(index)
            key_indexes[key] = tmp_list

        index += 1

    # Add all the manufacturers to the database at once
    manufacturers: list[CleanManufacturer] | CleanManufacturer = manufacturer_data_handler.table_handler.add(
        filters=filters,
        instance_builder=lambda values: CleanManufacturer(
            *values
        ),
        pk_column_names=[
            TableColumnName.NAME.value,
            TableColumnName.ORIGINAL_STATE_ID.value,
        ]
    )
    if type(manufacturers) is not list:
        manufacturers = [manufacturers]

    # Prepare a list of the same length as the cleaned dataframe, in which every element corresponds to a manufacturer
    # row in the initial dataframe and will contain the related manufacturer id
    ids_array = np.array([None] * len(cleaned_dataframe))
    # For each added manufacturer
    for manufacturer in manufacturers:
        # Get the manufacturer id
        manufacturer_id = manufacturer.identifier
        # Get the original manufacturer name
        manufacturer_name = manufacturer.name
        # Get the manufacturer original state id
        manuf_state_id = manufacturer.original_state_identifier
        key = '$'.join(
            [manufacturer_name, str(manuf_state_id)]
        )
        # Get all the dataframe indexes corresponding to the key
        indexes = key_indexes[key]
        # Set in the cleaned_dataframe the manufacturer id for all the indexes corresponding to the key with a single
        # query
        ids_array[indexes] = manufacturer_id

    # Set in the cleaned dataframe a new column containing the manufacturer ids for each row
    cleaned_dataframe[DataFrameColumnName.CLEAN_MANUFACTURER_ID_COLUMN.value] = ids_array
    # Drop any row which contains a NaN value in the manufacturer id column
    cleaned_dataframe = cleaned_dataframe.dropna(
        subset=[DataFrameColumnName.CLEAN_MANUFACTURER_ID_COLUMN.value]
    ).reset_index(inplace=False, drop=True)

    # Insert all the devices in the database
    device_data_handler = CleanDeviceDataHandler(
        connection_handler=connection_handler
    )

    # Prepare the list of device filters to add the data to the database, a map to keep track between the device
    # database PK and the dataframe indexes, and the current row index
    filters = []
    key_indexes: dict[str, list[int]] = {}
    index = 0
    # For each row in the cleaned dataframe
    for row in cleaned_dataframe.to_numpy():
        # Get all the information related to the given device
        (
            original_device_name,
            catalogue_code,
            clean_manufacturer_id,
            clean_name,
            device_type,
            high_risk_device_type,
            product_code,
            standardized_device_id
        ) = get_row_value(
            row=row,
            column_name=[
                5,
                8,
                7,
                6,
                9,
                10,
                11,
                12,
            ],
            casting_type=[
                str,
                str,
                int,
                str,
                str,
                str,
                str,
                int,
            ]
        )

        # If one of the PK values is None, skip the row
        if None not in [original_device_name, clean_manufacturer_id, catalogue_code]:
            # Build a key to identify the device, made of all the database PK components
            key = '$'.join(
                [original_device_name, catalogue_code, str(clean_manufacturer_id)]
            )
            # If the key is not in the dictionary, create a new list, otherwise get the list
            if key not in key_indexes.keys():
                # If the device is found for the first time, add its filter to the filters list
                filters.append(
                    [
                        SqlQueryComponent(
                            column_name=TableColumnName.NAME.value,
                            column_values=[original_device_name],
                        ),
                        SqlQueryComponent(
                            column_name=TableColumnName.CLEAN_MANUFACTURER_ID.value,
                            column_values=[clean_manufacturer_id],
                        ),
                        SqlQueryComponent(
                            column_name=TableColumnName.CATALOGUE_CODE.value,
                            column_values=[catalogue_code],
                        ),
                        SqlQueryComponent(
                            column_name=TableColumnName.CLEAN_NAME.value,
                            column_values=[clean_name],
                        ),
                        SqlQueryComponent(
                            column_name=TableColumnName.DEVICE_TYPE.value,
                            column_values=[device_type],
                        ),
                        SqlQueryComponent(
                            column_name=TableColumnName.HIGH_RISK_DEVICE_TYPE.value,
                            column_values=[high_risk_device_type],
                        ),
                        SqlQueryComponent(
                            column_name=TableColumnName.PRODUCT_CODE.value,
                            column_values=[product_code],
                        ),
                        SqlQueryComponent(
                            column_name=TableColumnName.STANDARDIZED_DEVICE_ID.value,
                            column_values=[standardized_device_id],
                        ),
                    ]
                )
                # Create a new list of indexes for the given device
                tmp_list = []
            else:
                # Get the old list of indexes
                tmp_list = key_indexes[key]
            # Replace the list in the dictionary with the one with the new row index appended
            tmp_list.append(index)
            key_indexes[key] = tmp_list
        index += 1

    # Add all the devices to the database at once
    devices: list[CleanDevice] = device_data_handler.table_handler.add(
        filters=filters,
        instance_builder=lambda values: CleanDevice(
            *values
        ),
        pk_column_names=[
            TableColumnName.NAME.value,
            TableColumnName.CLEAN_MANUFACTURER_ID.value,
            TableColumnName.CATALOGUE_CODE.value,
        ]
    )

    # Prepare a numpy array of the same length as the cleaned dataframe
    ids_array = np.array([None] * len(cleaned_dataframe))
    # For each of the added devices
    for device in devices:
        # Get the device id
        device_id = device.identifier
        # Get the original device name
        original_device_name = device.name
        # Get the manufacturer id
        manufacturer_id = device.manufacturer_id
        # Get the catalogue code
        catalogue_code = device.catalogue_code
        # Build the key
        key = '$'.join(
            [str(original_device_name), str(catalogue_code), str(manufacturer_id)]
        )
        # Get all the dataframe indexes corresponding to the key
        indexes = key_indexes[key]
        # Add the device id to the numpy array at the indexes corresponding to the key
        ids_array[indexes] = device_id

    # Add to the cleaned dataframe a new column containing the device ids for each row
    cleaned_dataframe[DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value] = ids_array
    # Drop any row which contains a NaN value in the device id column
    cleaned_dataframe = cleaned_dataframe.dropna(
        subset=[DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value]
    ).reset_index(inplace=False, drop=True)

    return cleaned_dataframe


def store_batched_italian_data(
        dataframe_strings: list[str],
        connection_handler: ConnectionHandler,
):
    # Add the manufacturer and device data to the database
    cleaned_dataframe = _store_batched_data(
        dataframe_strings=dataframe_strings,
        connection_handler=connection_handler,
    )

    # Get a list of all the unique EMDN codes
    emdn_codes = list(cleaned_dataframe[DataFrameColumnName.EMDN_CODE_COLUMN.value].unique())
    # Fetch all the EMDN codes from the database
    emdn_data_handler = EmdnCodeDataHandler(
        connection_handler=connection_handler,
        init_cache=True,
    )
    # Get a list of all the EMDN code instances
    emdn_codes_instances = emdn_data_handler.get_emdn_codes(
        alphanumeric_codes=emdn_codes,
    )
    # Prepare a map between EMDN code and EMDN code id
    emdn_code_to_id = {
        emdn_code_instance.emdn_code: emdn_code_id
        for emdn_code_id, emdn_code_instance in emdn_codes_instances.items()
    }
    # Replace in the dataframe the EMDN code with the corresponding EMDN code id
    index = 0
    rows = cleaned_dataframe.to_numpy()
    for row in rows:
        # Get the EMDN code of the row
        emdn_code = get_row_value(
            row=row,
            column_name=14,
            casting_type=str
        )
        try:
            # Get the EMDN code id from the map
            emdn_code_id = emdn_code_to_id[emdn_code]
            # Update the EMDN code in the dataframe
            rows[index][13] = emdn_code_id
        except KeyError:
            # If the EMDN code is not found in the map, it does not exist; therefore,
            # the row will be discarded while adding the correspondences
            pass
        index += 1

    # Replace in the dataframe the EMDN code with the corresponding EMDN code id
    cleaned_dataframe[DataFrameColumnName.EMDN_ID_COLUMN.value] = rows[:, 13]
    # Drop any row which contains a NaN value in the EMDN id column
    cleaned_dataframe = cleaned_dataframe.dropna(
        subset=[DataFrameColumnName.EMDN_ID_COLUMN.value]
    ).reset_index(inplace=False, drop=True)

    # Add to the database the EMDN id - device id correspondences
    _add_correspondences(
        correspondence_table_handler=EmdnCorrespondenceDataHandler(
            connection_handler=connection_handler
        ).table_handler,
        instance_builder=lambda values: EmdnCorrespondence(
            *values
        ),
        clean_dataframe=cleaned_dataframe[[
            DataFrameColumnName.EMDN_ID_COLUMN.value,
            DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value,
        ]].dropna().drop_duplicates(),
        first_column_name=TableColumnName.EMDN_ID.value,
        second_column_name=TableColumnName.CLEAN_DEVICE_ID.value,
    )


def store_batched_american_data(
        dataframe_strings: list[str],
        connection_handler: ConnectionHandler,
):
    # Add the manufacturer and device data to the database
    cleaned_dataframe = _store_batched_data(
        dataframe_strings=dataframe_strings,
        connection_handler=connection_handler,
    )

    # Extract a sub-dataframe containing all the GMDN Term Names, and Definitions
    # and drop the columns from the cleaned dataframe
    gmdn_data = cleaned_dataframe[[
        DataFrameColumnName.TERM_NAME_COLUMN.value,
        DataFrameColumnName.DEFINITION_COLUMN.value,
    ]].dropna()
    # Aggregate the GMDN Term Names and concatenate the definitions
    gmdn_data = gmdn_data.groupby(
        DataFrameColumnName.TERM_NAME_COLUMN.value,
    ).agg(
        lambda x: '$__$'.join(x)
    ).reset_index(inplace=False)

    # Add the GMDN data to the database
    gmdn_data_handler = GmdnCodeDataHandler(
        connection_handler=connection_handler
    )
    # Divide the total number of GMDN data into batches of 1000
    # and add them to the database
    for i in range(0, len(gmdn_data), 1000):
        gmdn_code_list: list[GmdnCode] | GmdnCode = gmdn_data_handler.table_handler.add(
            filters=[
                [
                    SqlQueryComponent(
                        column_name=TableColumnName.TERM_NAME.value,
                        column_values=[str(row[0]) if not pd.isna(row[0]) else None],
                    ),
                    SqlQueryComponent(
                        column_name=TableColumnName.DEFINITION.value,
                        column_values=[str(row[1]) if not pd.isna(row[1]) else None],
                    ),
                ] for row in gmdn_data.to_numpy()[i:i + 1000]
            ],
            instance_builder=lambda values: GmdnCode(
                *values
            ),
            pk_column_names=[
                TableColumnName.TERM_NAME.value,
            ]
        )
        if type(gmdn_code_list) is not list:
            gmdn_code_list = [gmdn_code_list]
        # Build a map to have the correspondence between GMDN term names and GMDN ids
        gmdn_term_to_id = {
            gmdn_code.description_data.sentence: gmdn_code.identifier
            for gmdn_code in gmdn_code_list
        }
        index = 0
        rows = cleaned_dataframe.to_numpy()
        # Populate the dataframe with the GMDN ids
        for row in rows:
            # Get the term name
            term_name = get_row_value(
                row=row,
                column_name=14,
                casting_type=str
            )
            try:
                # Get the GMDN id, if any
                gmdn_id = gmdn_term_to_id[term_name]
                # Update the GMDN id in the dataframe
                rows[index][13] = gmdn_id
            except KeyError:
                # If the term name is not found in the map, the row will be discarded while adding the correspondences
                pass
            index += 1

        # Replace in the dataframe the GMDN term name with the corresponding GMDN id
        cleaned_dataframe[DataFrameColumnName.GMDN_ID_COLUMN.value] = rows[:, 13]

    # Drop any row which contains a NaN value in the GMDN id column
    cleaned_dataframe = cleaned_dataframe.dropna(
        subset=[DataFrameColumnName.GMDN_ID_COLUMN.value]
    ).reset_index(inplace=False, drop=True)

    # Add the device - GMDN id correspondences to the database
    _add_correspondences(
        correspondence_table_handler=GmdnCorrespondenceDataHandler(
            connection_handler=connection_handler
        ).table_handler,
        instance_builder=lambda values: GmdnCorrespondence(
            *values
        ),
        clean_dataframe=cleaned_dataframe[[
            DataFrameColumnName.GMDN_ID_COLUMN.value,
            DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value,
        ]].dropna().drop_duplicates(),
        first_column_name=TableColumnName.GMDN_ID.value,
        second_column_name=TableColumnName.CLEAN_DEVICE_ID.value,
    )
