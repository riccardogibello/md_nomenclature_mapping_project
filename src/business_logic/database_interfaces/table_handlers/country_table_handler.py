from typing import Optional

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler

from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler, \
    SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.utilities.table_names import COUNTRY_TABLE_NAME
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.country import Country


class CountryTableHandler(AbstractTableHandler):

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            table_name=COUNTRY_TABLE_NAME,
            connection_handler=connection_handler,
            reset_table=reset_table,
            initializing_table_statement=lambda table_name: f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                    {TableColumnName.IDENTIFIER.value} SERIAL PRIMARY KEY,
                    {TableColumnName.NAME.value} VARCHAR(255) NOT NULL,
                    {TableColumnName.IS_AMERICAN_STATE.value} BOOL NOT NULL,
                    CONSTRAINT unique_country UNIQUE(
                        {TableColumnName.NAME.value}, 
                        {TableColumnName.IS_AMERICAN_STATE.value}
                    )
                )
            """
        )

    def add_country(
            self,
            country_name: str,
            is_american_state: bool,
    ) -> Country:
        new_country: Country = self.add(
            filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.NAME.value,
                    column_values=[country_name],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.IS_AMERICAN_STATE.value,
                    column_values=[is_american_state],
                ),
            ],
            instance_builder=lambda results: Country(
                *results
            ),
            pk_column_names=[
                TableColumnName.NAME.value,
                TableColumnName.IS_AMERICAN_STATE.value,
            ]
        )

        return new_country
