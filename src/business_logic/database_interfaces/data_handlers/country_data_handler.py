from typing import Optional, List

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.abstract_data_handler import AbstractDataHandler, \
    get_instances
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.country_table_handler import CountryTableHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.country import Country


class CountryDataHandler(AbstractDataHandler):
    country_id__country_instance: dict[int, Country]

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            init_cache: Optional[bool] = False,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            table_handler=CountryTableHandler(
                connection_handler=connection_handler,
                reset_table=reset_table,
            ),
        )

        self.country_id__country_instance = {}

        self.table_handler: CountryTableHandler = self.table_handler

        if init_cache:
            self.get_countries_by_id()

    def get_countries_by_id(
            self,
            country_ids: Optional[list[int]] = None,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> dict[int, Country]:
        filters: Optional[List[SqlQueryComponent]] = None if country_ids is None else [
            SqlQueryComponent(
                column_name=TableColumnName.IDENTIFIER.value,
                column_values=country_ids,
            ),
        ]

        return get_instances(
            fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                filters=_filters,
                instance_builder=lambda values: Country(
                    *values
                )
            ),
            sql_filters=filters,
            local_cache=self.country_id__country_instance,
            perform_fetch_on_database=perform_fetch_on_database,
        )

    def get_country(
            self,
            country_name: str,
            is_american_state: bool,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> Optional[Country]:
        returned_value = get_instances(
            fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                filters=_filters,
                instance_builder=lambda values: Country(
                    *values
                )
            ),
            sql_filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.NAME.value,
                    column_values=[country_name],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.IS_AMERICAN_STATE.value,
                    column_values=[is_american_state],
                ),
            ],
            local_cache=self.country_id__country_instance,
            perform_fetch_on_database=perform_fetch_on_database,
        )

        if returned_value is not None and len(returned_value) == 1:
            return list(returned_value.values())[0]
        else:
            return None

    def add_country(
            self,
            country_name: str,
            is_american_state: bool,
    ) -> Country:
        # Try to add the country to the database
        # if it is already present, the method will return the instance
        # fetched from the database
        new_country_instance = self.table_handler.add_country(
            country_name=country_name,
            is_american_state=is_american_state,
        )
        # Add the retrieved instance to the cache
        self.country_id__country_instance[new_country_instance.identifier] = new_country_instance

        return new_country_instance
