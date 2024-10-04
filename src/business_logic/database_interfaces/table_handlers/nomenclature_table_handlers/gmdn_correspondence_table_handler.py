from typing import Optional

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler, \
    SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.utilities.conversions import convert_parameters_to_sql_filters
from src.business_logic.database_interfaces.table_handlers.utilities.table_names import GMDN_CORRESPONDENCE_TABLE_NAME
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.nomenclature_codes.gmdn_correspondence import GmdnCorrespondence


class GmdnCorrespondenceTableHandler(AbstractTableHandler):

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            table_name=GMDN_CORRESPONDENCE_TABLE_NAME,
            connection_handler=connection_handler,
            reset_table=reset_table,
            initializing_table_statement=lambda table_name: f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                        {TableColumnName.IDENTIFIER.value} SERIAL PRIMARY KEY,
                        {TableColumnName.GMDN_ID.value} INTEGER NOT NULL,
                        {TableColumnName.CLEAN_DEVICE_ID.value} INTEGER NOT NULL,
                        {TableColumnName.MATCH_TYPE.value} VARCHAR(40) NOT NULL,
                        {TableColumnName.SIMILARITY.value} FLOAT DEFAULT NULL,
                        {TableColumnName.MATCHED_NAME.value} VARCHAR(1000) DEFAULT NULL,
                        CONSTRAINT unique_gmdn_correspondence UNIQUE(
                            {TableColumnName.GMDN_ID.value},
                            {TableColumnName.CLEAN_DEVICE_ID.value}
                        )
                )
            """,
        )

    def fetch_correspondence(
            self,
            clean_device_id: int,
    ) -> Optional[GmdnCorrespondence]:
        return self.fetch(
            instance_builder=lambda values: GmdnCorrespondence(
                *values
            ),
            filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.CLEAN_DEVICE_ID.value,
                    column_values=[clean_device_id],
                    must_be_comprised=True,
                )
            ]
        )

    def add_correspondence(
            self,
            **kwargs,
    ):
        filters = convert_parameters_to_sql_filters(
            kwargs
        )

        self.add(
            instance_builder=lambda values: GmdnCorrespondence(
                *values
            ),
            filters=filters,
            pk_column_names=[
                TableColumnName.GMDN_ID.value,
                TableColumnName.CLEAN_DEVICE_ID.value,
            ],
        )
