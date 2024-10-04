from typing import Optional, Union

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.nomenclature_table_handlers.code_table_handler import \
    CodeTableHandler
from src.business_logic.database_interfaces.table_handlers.utilities.table_names import EMDN_CODE_TABLE_NAME
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.nomenclature_codes.emdn_code import EmdnCode


class EmdnCodeTableHandler(CodeTableHandler):
    """
    This is the handler for the EMDN code table.
    """

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            connection_handler=connection_handler,
            table_name=EMDN_CODE_TABLE_NAME,
            reset_table=reset_table,
            initializing_table_statement=lambda table_name: f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                    {TableColumnName.IDENTIFIER.value} SERIAL PRIMARY KEY,
                    {TableColumnName.CODE.value} VARCHAR(255) NOT NULL,
                    {TableColumnName.DESCRIPTION.value} VARCHAR(255) NOT NULL,
                    {TableColumnName.IS_LEAF.value} BOOLEAN DEFAULT FALSE,
                    CONSTRAINT unique_emdn_code UNIQUE ({TableColumnName.CODE.value})
                )
            """
        )

    def fetch_emdn_codes(
            self,
            identifiers: Optional[list[int | str]] = None,
            is_search_set: bool = True,
    ) -> Union[None, EmdnCode, dict[int, EmdnCode]]:
        """
        This method fetches one or more EMDN codes from the database.

        :param identifiers: The values to be searched in the identifier / code columns.
        :param is_search_set: If the values given in identifiers are the ones to be fetched or excluded from the search.

        :return: The EMDN code(s) fetched from the database.
        """
        if identifiers is None or len(identifiers) == 0 or type(identifiers[0]) is int:
            field_name: TableColumnName = TableColumnName.IDENTIFIER
        else:
            field_name: TableColumnName = TableColumnName.CODE

        return super().fetch_codes(
            field_name=field_name,
            identifiers=identifiers,
            is_search_set=is_search_set,
            instance_builder=lambda values: EmdnCode(
                *values
            ),
        )

    def add_code(
            self,
            emdn_code_string: str,
            emdn_description: str,
            is_leaf: bool,
    ) -> EmdnCode:
        """
        This method adds a new EMDN code to the database.

        :param emdn_code_string: The alphanumeric code of the EMDN code.
        :param emdn_description: The description of the EMDN code.
        :param is_leaf: If the EMDN code is a leaf node in the EMDN hierarchy.

        :return: The EMDN code instance created from the passed data.
        """
        emdn_code: EmdnCode = self.add(
            filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.CODE.value,
                    column_values=[emdn_code_string],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.DESCRIPTION.value,
                    column_values=[emdn_description],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.IS_LEAF.value,
                    column_values=[is_leaf],
                ),
            ],
            instance_builder=lambda values: EmdnCode(
                *values
            ),
            pk_column_names=[TableColumnName.CODE.value],
        )

        return emdn_code
