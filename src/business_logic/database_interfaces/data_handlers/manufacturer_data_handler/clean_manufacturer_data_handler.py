from typing import Optional, Tuple

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.abstract_data_handler import AbstractDataHandler, \
    get_instances
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent, SqlOperator
from src.business_logic.database_interfaces.table_handlers.manufacturer_table_handler.clean_manufacturer_table_handler import \
    CleanManufacturerTableHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.manufacturers.clean_manufacturer import CleanManufacturer


class CleanManufacturerDataHandler(AbstractDataHandler):
    manufacturer_id__manufacturer_instance: dict[int, CleanManufacturer]

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            init_cache: Optional[bool] = False,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            table_handler=CleanManufacturerTableHandler(
                connection_handler=connection_handler,
                reset_table=reset_table,
            )
        )
        self.table_handler: CleanManufacturerTableHandler = self.table_handler

        self.manufacturer_id__manufacturer_instance = {}

        if init_cache:
            self.get_manufacturers()

    def get_manufacturers(
            self,
            manufacturer_ids: Optional[list[int]] = None,
            manufacturer_name__state_identifier: Optional[list[Tuple[str, int]]] = None,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> dict[int, CleanManufacturer] | CleanManufacturer:
        if manufacturer_ids is not None and manufacturer_name__state_identifier is not None:
            raise Exception("You cannot pass both manufacturer ids and manufacturer name-state identifier pairs.")
        else:
            sql_filters: Optional[list[SqlQueryComponent]] = None
            if manufacturer_ids is not None:
                sql_filters = [
                    SqlQueryComponent(
                        column_name=TableColumnName.IDENTIFIER.value,
                        column_values=manufacturer_ids,
                    )
                ]
            elif manufacturer_name__state_identifier is not None:
                sql_filters = []
                for name, state_identifier in manufacturer_name__state_identifier:
                    sql_filters.append(
                        SqlQueryComponent(
                            column_name=TableColumnName.NAME.value,
                            column_values=[name],
                        )
                    )
                    sql_filters.append(
                        SqlQueryComponent(
                            column_name=TableColumnName.ORIGINAL_STATE_ID.value,
                            column_values=[state_identifier],
                            next_operator=SqlOperator.OR,
                        )
                    )

            returned_value = get_instances(
                fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                    instance_builder=lambda values: CleanManufacturer(
                        *values
                    ),
                    filters=_filters,
                ),
                sql_filters=sql_filters,
                local_cache=self.manufacturer_id__manufacturer_instance,
                perform_fetch_on_database=perform_fetch_on_database
            )

            if len(returned_value) == 1:
                return list(returned_value.values())[0]
            else:
                return returned_value

    def get_manufacturer(
            self,
            original_manufacturer_name: str,
            original_state_identifier: int,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> Tuple[bool, Optional[CleanManufacturer]]:
        returned_value = get_instances(
            fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                instance_builder=lambda values: CleanManufacturer(
                    *values
                ),
                filters=_filters,
            ),
            sql_filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.NAME.value,
                    column_values=[original_manufacturer_name],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.ORIGINAL_STATE_ID.value,
                    column_values=[original_state_identifier],
                ),
            ],
            local_cache=self.manufacturer_id__manufacturer_instance,
            perform_fetch_on_database=perform_fetch_on_database,
        )

        if returned_value is not None and len(returned_value) > 0:
            return True, list(returned_value.values())[0]
        else:
            return False, None

    def add_clean_manufacturer(
            self,
            clean_name: str,
            name: str,
            original_state_identifier: int,
            standardized_manufacturer_id: Optional[int] = None,
    ) -> CleanManufacturer:
        return self.table_handler.add_clean_manufacturer(
            name=name,
            clean_name=clean_name,
            standardized_manufacturer_id=standardized_manufacturer_id,
            original_state_id=original_state_identifier,
        )
