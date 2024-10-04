from typing import Optional

from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler, \
    SqlQueryComponent
from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.table_handlers.utilities.conversions import convert_parameters_to_sql_filters
from src.business_logic.database_interfaces.table_handlers.utilities.table_names import MAPPING_TABLE_NAME
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.mapping.nomenclature_mapping import NomenclatureMapping


class MappingTableHandler(AbstractTableHandler):
    """
    This handler manages the 'mapping' table, used to store pre-built translations of GMDN and EMDN codes.
    """

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            connection_handler=connection_handler,
            table_name=MAPPING_TABLE_NAME,
            reset_table=reset_table,
            initializing_table_statement=lambda table_name: f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                    {TableColumnName.IDENTIFIER.value} SERIAL PRIMARY KEY,
                    {TableColumnName.EMDN_ID.value} INTEGER NOT NULL,
                    {TableColumnName.GMDN_ID.value} INTEGER NOT NULL,
                    CONSTRAINT unique_mapping UNIQUE (
                        {TableColumnName.EMDN_ID.value}, 
                        {TableColumnName.GMDN_ID.value}
                    )
                )
            """
        )

    def fetch_mappings(
            self,
    ) -> list[NomenclatureMapping]:
        """
        This method fetches from the database a list of mappings.
        :return: A list of mappings stored in the database 'mapping' table.
        """
        sql_string = f"SELECT * FROM {self.table_name}"
        rows = self.connection_handler.fetch(
            sql_string,
            parameters=(),
            fetch_only_one=False,
        )

        result: list[NomenclatureMapping] = []

        for row in rows:
            result.append(
                NomenclatureMapping(
                    identifier=row[0],
                    emdn_id=int(row[1]),
                    gmdn_id=int(row[2]),
                )
            )

        return result

    def get_mapping(
            self,
            emdn_id: int,
            gmdn_id: int,
    ) -> Optional[NomenclatureMapping]:
        """
        This method gets from the database a Mapping related to the given couple, if it exists. None, otherwise.

        :param emdn_id: The EMDN src involved in the mapping.
        :param gmdn_id: The GMDN src involved in the mapping.

        :return: A Mapping related to the given couple, if it exists. None, otherwise.
        """
        mapping = self.fetch(
            filters=convert_parameters_to_sql_filters(
                locals()
            ),
            instance_builder=lambda values: NomenclatureMapping(
                *values
            ),
        )

        return mapping

    def add_mapping(
            self,
            emdn_gmdn_ids: list[tuple[int, int]],
    ) -> NomenclatureMapping | list[NomenclatureMapping]:
        """
        This method adds a new mapping to the table.

        :param emdn_gmdn_ids: A list of tuples, each containing the EMDN and GMDN identifiers involved in the mapping.

        :return: A Mapping instance, which contains all the metadata related to the mapping.
        """
        filters: list[list[SqlQueryComponent]] = []
        for emdn_id, gmdn_id in emdn_gmdn_ids:
            filters.append(
                [
                    SqlQueryComponent(
                        column_name=TableColumnName.EMDN_ID.value,
                        column_values=[int(emdn_id)]
                    ),
                    SqlQueryComponent(
                        column_name=TableColumnName.GMDN_ID.value,
                        column_values=[int(gmdn_id)]
                    )
                ]
            )

        return self.add(
            filters=filters,
            instance_builder=lambda values: NomenclatureMapping(
                *values
            ),
            pk_column_names=[
                TableColumnName.EMDN_ID.value,
                TableColumnName.GMDN_ID.value
            ],
        )

    def remove_mapping_from_database(
            self,
            gmdn_id: int,
            emdn_id: int,
    ) -> None:
        """
        This method removes from the table the row which contains the given identifiers.

        :param emdn_id: The EMDN identifier involved in the mapping.
        :param gmdn_id: The GMDN identifier involved in the mapping.
        """
        sql_string = f"""
            DELETE 
            FROM {self.table_name} 
            WHERE {TableColumnName.GMDN_ID.value} = ? AND {TableColumnName.EMDN_ID.value} = ?
        """

        self.connection_handler.execute(
            sql_string,
            parameters=(gmdn_id, emdn_id,),
        )
