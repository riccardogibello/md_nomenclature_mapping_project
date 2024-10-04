from typing import Optional

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler, \
    SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.utilities.table_names import DEVICE_MAPPING_TABLE_NAME
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.mapping.device_mapping import DeviceMapping


class DeviceMappingTableHandler(AbstractTableHandler):
    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            connection_handler=connection_handler,
            table_name=DEVICE_MAPPING_TABLE_NAME,
            reset_table=reset_table,
            initializing_table_statement=lambda table_name: f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                    {TableColumnName.IDENTIFIER.value} SERIAL PRIMARY KEY,
                    {TableColumnName.MAPPING_ID.value} INTEGER DEFAULT NULL,
                    {TableColumnName.FIRST_DEVICE_ID.value} INTEGER NOT NULL,
                    {TableColumnName.SECOND_DEVICE_ID.value} INTEGER NOT NULL,
                    {TableColumnName.COMPANY_NAME_SIMILARITY.value} FLOAT NOT NULL,
                    {TableColumnName.SIMILARITY.value} FLOAT NOT NULL,
                    CONSTRAINT unique_device_mapping UNIQUE (
                        {TableColumnName.MAPPING_ID.value},
                        {TableColumnName.FIRST_DEVICE_ID.value},
                        {TableColumnName.SECOND_DEVICE_ID.value}
                    )
                )
            """
        )

    def add_device_mapping(
            self,
            mapping_id: int,
            first_device_id: int,
            second_device_id: int,
            company_name_similarity: float,
            similarity: float,
    ):
        """
        This method stores the mapping between two devices.
        :param mapping_id: The identifier of the mapping.
        :param first_device_id: The identifier of the first device.
        :param second_device_id: The identifier of the second device.
        :param company_name_similarity: The similarity between the company names of the two devices.
        :param similarity: The similarity between the two devices.
        """
        return self.add(
            filters=[
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
                SqlQueryComponent(
                    column_name=TableColumnName.COMPANY_NAME_SIMILARITY.value,
                    column_values=[company_name_similarity],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.SIMILARITY.value,
                    column_values=[similarity],
                ),
            ],
            instance_builder=lambda values: DeviceMapping(
                *values
            ),
            pk_column_names=[
                TableColumnName.MAPPING_ID.value,
                TableColumnName.FIRST_DEVICE_ID.value,
                TableColumnName.SECOND_DEVICE_ID.value,
            ]
        )
