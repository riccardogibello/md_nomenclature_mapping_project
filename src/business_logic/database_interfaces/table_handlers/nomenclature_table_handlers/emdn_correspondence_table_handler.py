from typing import Optional

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler, \
    SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.utilities.table_names import EMDN_CORRESPONDENCE_TABLE_NAME
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.nomenclature_codes.emdn_correspondence import EmdnCorrespondence
from src.data_model.enums import MatchType


class EmdnCorrespondenceTableHandler(AbstractTableHandler):

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            connection_handler=connection_handler,
            table_name=EMDN_CORRESPONDENCE_TABLE_NAME,
            reset_table=reset_table,
            initializing_table_statement=lambda table_name: f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                    {TableColumnName.IDENTIFIER.value} SERIAL PRIMARY KEY,
                    {TableColumnName.EMDN_ID.value} INTEGER NOT NULL,
                    {TableColumnName.CLEAN_DEVICE_ID.value} INTEGER NOT NULL,
                    {TableColumnName.MATCH_TYPE.value} VARCHAR(40) NOT NULL,
                    {TableColumnName.SIMILARITY.value} FLOAT DEFAULT NULL,
                    {TableColumnName.MATCHED_NAME.value} VARCHAR(1000) DEFAULT NULL,
                    CONSTRAINT unique_emdn_correspondence UNIQUE(
                        {TableColumnName.EMDN_ID.value}, 
                        {TableColumnName.CLEAN_DEVICE_ID.value}
                    )
                )
            """
        )

    def fetch_correspondence(
            self,
            clean_device_id: int,
    ) -> Optional[EmdnCorrespondence]:
        return self.fetch(
            filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.CLEAN_DEVICE_ID.value,
                    column_values=[clean_device_id],
                    must_be_comprised=True,
                )
            ],
            instance_builder=lambda values: EmdnCorrespondence(
                *values
            )
        )

    def add_correspondence(
            self,
            emdn_code_id: int,
            clean_device_id: int,
            match_type: MatchType,
            similarity_value: Optional[float] = None,
            matched_name: Optional[str] = None,
    ) -> EmdnCorrespondence:
        emdn_correspondence: EmdnCorrespondence = self.add(
            filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.EMDN_ID.value,
                    column_values=[emdn_code_id],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.CLEAN_DEVICE_ID.value,
                    column_values=[clean_device_id],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.MATCH_TYPE.value,
                    column_values=[match_type.value],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.SIMILARITY.value,
                    column_values=[similarity_value],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.MATCHED_NAME.value,
                    column_values=[matched_name],
                ),
            ],
            instance_builder=lambda values: EmdnCorrespondence(
                *values
            ),
            pk_column_names=[
                TableColumnName.EMDN_ID.value,
                TableColumnName.CLEAN_DEVICE_ID.value,
            ]
        )

        return emdn_correspondence

