from typing import Optional

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler
from src.business_logic.database_interfaces.table_handlers.utilities.conversions import convert_parameters_to_sql_filters
from src.business_logic.database_interfaces.table_handlers.utilities.table_names import FDA_CODE_TABLE_NAME
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.nomenclature_codes.fda_code import FdaCode


class FdaCodeTableHandler(AbstractTableHandler):

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            connection_handler=connection_handler,
            table_name=FDA_CODE_TABLE_NAME,
            reset_table=reset_table,
            initializing_table_statement=lambda table_name: f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                    {TableColumnName.IDENTIFIER.value} SERIAL PRIMARY KEY,
                    {TableColumnName.PRODUCT_CODE.value} VARCHAR(3) NOT NULL,
                    {TableColumnName.DEVICE_NAME.value} TEXT NOT NULL,
                    {TableColumnName.DEVICE_CLASS.value} INTEGER NOT NULL,
                    {TableColumnName.PANEL.value} VARCHAR(2) DEFAULT NULL,
                    {TableColumnName.MEDICAL_SPECIALTY.value} VARCHAR(2) DEFAULT NULL,
                    {TableColumnName.SUBMISSION_TYPE_ID.value} INTEGER DEFAULT NULL,
                    {TableColumnName.DEFINITION.value} TEXT DEFAULT NULL,
                    {TableColumnName.PHYSICAL_STATE.value} TEXT DEFAULT NULL,
                    {TableColumnName.TECHNICAL_METHOD.value} TEXT DEFAULT NULL,
                    {TableColumnName.TARGET_AREA.value} TEXT DEFAULT NULL,
                    {TableColumnName.IS_IMPLANT.value} BOOLEAN DEFAULT NULL,
                    {TableColumnName.IS_LIFE_SUSTAINING.value} BOOLEAN DEFAULT NULL,
                    CONSTRAINT unique_fda_code UNIQUE(
                        {TableColumnName.PRODUCT_CODE.value}
                    )
                )
            """,
        )

    def add_fda_code(
            self,
            **kwargs
    ) -> FdaCode:
        filters = convert_parameters_to_sql_filters(
            kwargs
        )

        new_entity = self.add(
            filters=filters,
            instance_builder=lambda values: FdaCode(
                *values
            ),
            pk_column_names=[TableColumnName.PRODUCT_CODE.value]
        )

        return new_entity
