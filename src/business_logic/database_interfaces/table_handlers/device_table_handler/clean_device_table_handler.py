from typing import Optional

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler
from src.business_logic.database_interfaces.table_handlers.utilities.conversions import convert_parameters_to_sql_filters
from src.business_logic.database_interfaces.table_handlers.utilities.table_names import CLEAN_DEVICE_TABLE_NAME
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.devices.clean_device import CleanDevice


class CleanDeviceTableHandler(AbstractTableHandler):
    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            table_name=CLEAN_DEVICE_TABLE_NAME,
            connection_handler=connection_handler,
            reset_table=reset_table,
            initializing_table_statement=lambda table_name: f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                    {TableColumnName.IDENTIFIER.value} SERIAL PRIMARY KEY,
                    {TableColumnName.NAME.value} TEXT NOT NULL,
                    {TableColumnName.CLEAN_MANUFACTURER_ID.value} INTEGER NOT NULL,
                    {TableColumnName.CATALOGUE_CODE.value} TEXT DEFAULT NULL,                
                    {TableColumnName.CLEAN_NAME.value} TEXT DEFAULT NULL,
                    {TableColumnName.DEVICE_TYPE.value} VARCHAR(10) NOT NULL,
                    {TableColumnName.HIGH_RISK_DEVICE_TYPE.value} VARCHAR(100) DEFAULT NULL,
                    {TableColumnName.PRODUCT_CODE.value} VARCHAR(100) DEFAULT NULL,
                    {TableColumnName.STANDARDIZED_DEVICE_ID.value} INTEGER DEFAULT NULL,
                    CONSTRAINT unique_clean_device UNIQUE (
                        {TableColumnName.NAME.value}, 
                        {TableColumnName.CLEAN_MANUFACTURER_ID.value},
                        {TableColumnName.CATALOGUE_CODE.value}
                    )
                )
            """
        )

    def add_clean_device(
            self,
            **kwargs
    ) -> CleanDevice:
        return self.add(
            filters=convert_parameters_to_sql_filters(
                kwargs=kwargs
            ),
            instance_builder=lambda values: CleanDevice(
                *values
            ),
            pk_column_names=[
                TableColumnName.NAME.value,
                TableColumnName.CLEAN_MANUFACTURER_ID.value,
                TableColumnName.CATALOGUE_CODE.value
            ]
        )
