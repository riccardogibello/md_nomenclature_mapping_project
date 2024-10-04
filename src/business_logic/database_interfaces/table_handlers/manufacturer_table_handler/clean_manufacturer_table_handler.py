from typing import Optional

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.table_handlers.manufacturer_table_handler.abstract_manufacturer_table_handler \
    import AbstractManufacturerTableHandler
from src.business_logic.database_interfaces.table_handlers.utilities.conversions import convert_parameters_to_sql_filters
from src.business_logic.database_interfaces.table_handlers.utilities.table_names import CLEAN_MANUFACTURER_TABLE_NAME
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.manufacturers.clean_manufacturer import CleanManufacturer


class CleanManufacturerTableHandler(AbstractManufacturerTableHandler):

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            table_name=CLEAN_MANUFACTURER_TABLE_NAME,
            connection_handler=connection_handler,
            reset_table=reset_table,
            initializing_table_statement=lambda table_name: f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                    {TableColumnName.IDENTIFIER.value} SERIAL PRIMARY KEY,
                    {TableColumnName.NAME.value} VARCHAR(255) NOT NULL,
                    {TableColumnName.CLEAN_NAME.value} VARCHAR(255) NOT NULL,
                    {TableColumnName.STANDARDIZED_MANUFACTURER_ID.value} INTEGER DEFAULT NULL,
                    {TableColumnName.ORIGINAL_STATE_ID.value} INTEGER DEFAULT NULL,
                    CONSTRAINT unique_clean_manufacturer UNIQUE(
                        {TableColumnName.NAME.value}, 
                        {TableColumnName.ORIGINAL_STATE_ID.value}
                    )
                )
            """
        )

    def add_clean_manufacturer(
            self,
            **kwargs,
    ) -> CleanManufacturer:
        return self.add(
            filters=convert_parameters_to_sql_filters(
                kwargs
            ),
            instance_builder=lambda values: CleanManufacturer(
                *values
            ),
            pk_column_names=[
                TableColumnName.NAME.value,
                TableColumnName.ORIGINAL_STATE_ID.value,
            ]
        )
