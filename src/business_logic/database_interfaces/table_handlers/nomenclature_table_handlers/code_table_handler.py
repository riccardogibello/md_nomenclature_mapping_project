from typing import Optional, Union, List, Callable

from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler, \
    SqlQueryComponent
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.nomenclature_codes.emdn_code import EmdnCode
from src.data_model.nomenclature_codes.gmdn_code import GmdnCode


class CodeTableHandler(AbstractTableHandler):
    """
    This is the base class for handling tables about nomenclature codes.
    """

    def fetch_codes(
            self,
            field_name: TableColumnName,
            instance_builder: Callable[[list], Union[EmdnCode, GmdnCode]],
            identifiers: Optional[list[int | str]] = None,
            is_search_set: bool = True,
    ) -> Union[None, EmdnCode | GmdnCode, dict[int, EmdnCode | GmdnCode]]:
        """
        This method fetches one or more codes from the database.

        :param field_name: The name of the field to be searched (e.g., identifier, term name, etc.).
        :param instance_builder: The function to build the instance of the code.
        :param identifiers: The values to be searched.
        :param is_search_set: If the values given in identifiers are the ones to be fetched or excluded from the search.

        :return: The code(s) fetched from the database.
        """
        sql_filters: Optional[List[SqlQueryComponent]] = None
        # If the identifiers are given
        if identifiers is not None and len(identifiers) > 0:
            # Initialize the list of SQL filters
            sql_filters = []
            # Set the values to be searched
            field_values = identifiers
            sql_filters.append(
                SqlQueryComponent(
                    column_name=field_name.value,
                    column_values=field_values,
                    must_be_comprised=is_search_set,
                )
            )

        return self.fetch(
            instance_builder=instance_builder,
            filters=sql_filters,
        )
