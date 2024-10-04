import multiprocessing
import threading
from typing import Any, Tuple, Optional

import numpy as np
import pandas as pd

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.abstract_data_handler import AbstractDataHandler, get_instances
from src.business_logic.database_interfaces.data_handlers.mappings_data_handlers.mapping_data_handler import MappingDataHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.mappings_table_handlers.device_mapping_table_handler import \
    DeviceMappingTableHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.business_logic.pipeline.data_preprocessing.dataframe_column_names import DataFrameColumnName
from src.data_model.mapping.device_mapping import DeviceMapping
from src.data_model.mapping.nomenclature_mapping import NomenclatureMapping


class DeviceMappingDataHandler(AbstractDataHandler):
    device_mapping_table_thread_lock: threading.Lock = threading.Lock()
    device_table_process_lock: multiprocessing.Lock = multiprocessing.Lock()

    id_device_mapping_dict: dict[int, DeviceMapping]

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
            mapping_data_handler: Optional[MappingDataHandler] = None,
    ):
        if mapping_data_handler is None:
            mapping_data_handler = MappingDataHandler(
                connection_handler=connection_handler,
            )

        super().__init__(
            table_handler=DeviceMappingTableHandler(
                connection_handler=connection_handler,
                reset_table=reset_table,
            )
        )

        self.id_device_mapping_dict = {}

        self.table_handler = DeviceMappingTableHandler(
            connection_handler=connection_handler,
        )

        self.mapping_data_handler = mapping_data_handler

    def _is_device_already_existing(
            self,
            mapping_id: int,
            first_device_id: str,
            second_device_id: int,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> Tuple[bool, Optional[DeviceMapping]]:
        """
        This method checks if a device is already existing in the database.
        The key used to check if the device is (original_device_name, clean_manufacturer_id).
        The method firstly tries in the cache, then in the database (if fetch_from_db is set to True).

        :param mapping_id:      The identifier of the mapping.
        :param first_device_id: The identifier of the first device.
        :param second_device_id: The identifier of the second device.

        :return:    If the device is already existing, then the method returns True and the device instance.
                    If the device is not existing, then the method returns False and None.
        """
        returned_value = get_instances(
            fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                instance_builder=lambda *values: DeviceMapping(
                    *values
                ),
                filters=_filters,
            ),
            sql_filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.MAPPING_ID.value,
                    column_values=[mapping_id],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.FIRST_DEVICE_ID.value,
                    column_values=[first_device_id],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.SECOND_DEVICE_ID.value,
                    column_values=[second_device_id],
                ),
            ],
            local_cache=self.id_device_mapping_dict,
            perform_fetch_on_database=perform_fetch_on_database
        )

        if len(returned_value) == 1:
            return True, list(returned_value.values())[0]
        else:
            return False, None

    def add_batched_device_mappings(
            self,
            mappings: pd.DataFrame,
    ) -> Optional[Tuple[dict[int, NomenclatureMapping], dict[int, DeviceMapping]]]:
        """
        This method stores the given data batch in the proper tables to keep the device and nomenclature
        mapping information.

        :param mappings:  The dataframe containing the device mappings.
        :return:
        """
        # Select the EMDN and GMDN identifiers from the given dataframe
        gmdn_emdn_mappings: pd.DataFrame = mappings[
            [
                DataFrameColumnName.GMDN_ID_COLUMN.value,
                DataFrameColumnName.EMDN_ID_COLUMN.value,
            ]
        ].dropna().drop_duplicates()
        gmdn_emdn_mappings_batch = gmdn_emdn_mappings.to_numpy()
        # Extract the third (gmdn_id) and fourth (emdn_id) columns from the tuple list
        gmdn_ids = gmdn_emdn_mappings_batch[:, 0]
        emdn_ids = gmdn_emdn_mappings_batch[:, 1]
        # Create a new matrix made of gmdn_id and emdn_id columns
        gmdn_emdn_matrix: np.ndarray[Any, np.dtype] = np.column_stack((gmdn_ids, emdn_ids))

        sql_filters = []
        for gmdn_id, emdn_id in gmdn_emdn_matrix:
            sql_filters.append(
                [
                    SqlQueryComponent(
                        column_name=TableColumnName.EMDN_ID.value,
                        column_values=[int(emdn_id)],
                    ),
                    SqlQueryComponent(
                        column_name=TableColumnName.GMDN_ID.value,
                        column_values=[int(gmdn_id)],
                    ),
                ]
            )

        gmdn_emdn_mapping_list = []
        # If the matrix is empty, do not add the mappings
        if len(sql_filters) > 0:
            # Add all the GMDN-EMDN mappings to the database
            gmdn_emdn_mapping_list: list[NomenclatureMapping] = self.mapping_data_handler.mapping_table_handler.add(
                filters=sql_filters,
                instance_builder=lambda _values: NomenclatureMapping(
                    *_values
                ),
                pk_column_names=[
                    TableColumnName.GMDN_ID.value,
                    TableColumnName.EMDN_ID.value,
                ],
            )

        mapping_id_mapping_instance_dict = {
            int(mapping.identifier): mapping
            for mapping in gmdn_emdn_mapping_list
        }
        # Map the new mappings from EMDN and GMDN identifier to the mapping instance
        gmdn_emdn_id_to_mapping_id: dict[str, int] = {
            "$".join([str(int(mapping.gmdn_id)), str(int(mapping.emdn_id))]): int(mapping.identifier)
            for mapping in gmdn_emdn_mapping_list
        }

        mappings_batch = mappings.to_numpy()
        sql_filters = []
        # For each row in the given dataframe, add the device mapping to the database
        mapping: Optional[NomenclatureMapping]
        for first_device_id, second_device_id, emdn_id, gmdn_id, company_name_similarity, similarity in mappings_batch:
            if pd.isna(gmdn_id) or pd.isna(emdn_id):
                mapping_identifier = None
            else:
                # Get from the dictionary the mapping identifier related to the current GMDN-EMDN mapping
                key = "$".join([str(int(gmdn_id)), str(int(emdn_id))])
                if key in gmdn_emdn_id_to_mapping_id.keys():
                    mapping_id = gmdn_emdn_id_to_mapping_id[key]
                    mapping = mapping_id_mapping_instance_dict[mapping_id]
                    mapping_identifier = int(mapping.identifier)
                else:
                    mapping_identifier = None

            sql_filters.append(
                [
                    SqlQueryComponent(
                        column_name=TableColumnName.MAPPING_ID.value,
                        column_values=[mapping_identifier],
                    ),
                    SqlQueryComponent(
                        column_name=TableColumnName.FIRST_DEVICE_ID.value,
                        column_values=[int(first_device_id)],
                    ),
                    SqlQueryComponent(
                        column_name=TableColumnName.SECOND_DEVICE_ID.value,
                        column_values=[int(second_device_id)],
                    ),
                    SqlQueryComponent(
                        column_name=TableColumnName.COMPANY_NAME_SIMILARITY.value,
                        column_values=[float(company_name_similarity)],
                    ),
                    SqlQueryComponent(
                        column_name=TableColumnName.SIMILARITY.value,
                        column_values=[float(similarity)],
                    ),
                ],
            )

        device_mapping_list = []
        if len(sql_filters) > 0:
            # Add the device mapping to the database
            device_mapping_list = self.table_handler.add(
                filters=sql_filters,
                instance_builder=lambda _values: DeviceMapping(
                    *_values
                ),
                pk_column_names=[
                    TableColumnName.MAPPING_ID.value,
                    TableColumnName.FIRST_DEVICE_ID.value,
                    TableColumnName.SECOND_DEVICE_ID.value,
                ],
            )
        device_mapping_dict = {
            device_mapping.identifier: device_mapping
            for device_mapping in device_mapping_list
        }

        return mapping_id_mapping_instance_dict, device_mapping_dict

    def get_device_mappings(
            self,
            nomenclature_mapping_id: int,
    ) -> dict[int, DeviceMapping] | DeviceMapping | None:
        """
        This method retrieves the device mappings related to the given nomenclature mapping identifier.

        :param nomenclature_mapping_id: The identifier of the nomenclature mapping.

        :return: The device mappings related to the given nomenclature mapping identifiers.
        """
        return get_instances(
            fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                instance_builder=lambda values: DeviceMapping(
                    *values
                ),
                filters=_filters,
            ),
            sql_filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.MAPPING_ID.value,
                    column_values=[nomenclature_mapping_id],
                ),
            ],
            local_cache=self.id_device_mapping_dict,
        )

    def is_from_exact_device_mapping(
            self,
            nomenclature_mapping_id: int,
    ) -> int:
        """
        This method checks if the given device mapping is an exact device mapping.

        :param nomenclature_mapping_id: The identifier of the nomenclature mapping.

        :return: True if the device mapping is exact, False otherwise.
        """
        device_mappings = self.get_device_mappings(
            nomenclature_mapping_id=nomenclature_mapping_id,
        )

        if device_mappings is None:
            raise Exception("The device mappings are not found.")

        is_from_exact = 0
        if type(device_mappings) is DeviceMapping:
            device_mappings = [device_mappings]
        else:
            device_mappings = list(device_mappings.values())

        for device_mapping in device_mappings:
            if device_mapping.similarity == 100:
                is_from_exact = 1
                break

        return is_from_exact
