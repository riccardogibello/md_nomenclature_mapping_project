from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent
from src.business_logic.database_interfaces.__table_constants import TableColumnName


def convert_parameters_to_sql_filters(
        kwargs
) -> list[SqlQueryComponent]:
    filters: list[SqlQueryComponent] = []

    # For each keyword and the related argument
    for key, value in kwargs.items():
        # Get the table name related to the key
        table_name: TableColumnName = TableColumnName.get_enum_from_value(
            value=key,
            enum_class=TableColumnName
        )

        filters.append(
            SqlQueryComponent(
                column_name=table_name.value,
                column_values=[value]
            )
        )

    return filters
