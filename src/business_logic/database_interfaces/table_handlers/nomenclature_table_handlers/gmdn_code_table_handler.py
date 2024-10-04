from typing import Optional

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.nomenclature_table_handlers.code_table_handler import \
    CodeTableHandler
from src.business_logic.database_interfaces.table_handlers.utilities.table_names import GMDN_CODE_TABLE_NAME
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.nomenclature_codes.gmdn_code import GmdnCode


class GmdnCodeTableHandler(CodeTableHandler):
    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            connection_handler=connection_handler,
            table_name=GMDN_CODE_TABLE_NAME,
            reset_table=reset_table,
            initializing_table_statement=lambda table_name: f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                    {TableColumnName.IDENTIFIER.value} SERIAL PRIMARY KEY,
                    {TableColumnName.TERM_NAME.value} TEXT NOT NULL,
                    {TableColumnName.DEFINITION.value} TEXT NOT NULL,
                    CONSTRAINT unique_gmdn_code UNIQUE(
                        {TableColumnName.TERM_NAME.value}
                    )
                )
            """,
        )

    def fetch_gmdn_codes(
            self,
            identifiers: Optional[list[int | str]] = None,
            is_search_set: bool = True,
    ) -> Optional[GmdnCode | dict[int, GmdnCode]]:
        if identifiers is None or len(identifiers) == 0 or type(identifiers[0]) is int:
            field_name: TableColumnName = TableColumnName.IDENTIFIER
        else:
            field_name: TableColumnName = TableColumnName.TERM_NAME

        return super().fetch_codes(
            field_name=field_name,
            identifiers=identifiers,
            is_search_set=is_search_set,
            instance_builder=lambda values: GmdnCode(
                *values
            ),
        )

    def add_gmdn_code(
            self,
            term_name: str,
            definition: str,
    ) -> GmdnCode:
        return self.add(
            filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.TERM_NAME.value,
                    column_values=[term_name],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.DEFINITION.value,
                    column_values=[definition],
                ),
            ],
            instance_builder=lambda values: GmdnCode(
                *values
            ),
            pk_column_names=[TableColumnName.TERM_NAME.value],
        )
