from typing import Any, Optional

from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler


class AbstractManufacturerTableHandler(AbstractTableHandler):
    def fetch_manufacturer(
            self,
            sql_string: str,
            values: list[Any],
    ) -> Optional[list[Any]]:
        results = self.connection_handler.fetch(
            sql_string=sql_string,
            parameters=tuple(values),
            fetch_only_one=True,
        )

        if len(results) == 0:
            return None
        else:
            # the order of the columns must be the same as the one contained in the class' constructor
            return results
