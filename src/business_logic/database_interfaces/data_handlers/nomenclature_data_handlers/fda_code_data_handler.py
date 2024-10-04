from typing import Optional, Tuple

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.abstract_data_handler import AbstractDataHandler, \
    get_instances
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.nomenclature_table_handlers.fda_code_table_handler import \
    FdaCodeTableHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.nomenclature_codes.fda_code import FdaCode


class FdaCodeDataHandler(AbstractDataHandler):
    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            table_handler=FdaCodeTableHandler(
                connection_handler=connection_handler,
                reset_table=reset_table,
            )
        )

        self.fda_code_identifier___fda_code_instance: dict[int, FdaCode] = {}

        self.table_handler: FdaCodeTableHandler = self.table_handler

    def get_fda_code(
            self,
            product_code: str,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> Tuple[bool, Optional[FdaCode]]:
        returned_value = get_instances(
            fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                filters=_filters,
                instance_builder=lambda values: FdaCode(
                    *values
                ),
            ),
            sql_filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.PRODUCT_CODE.value,
                    column_values=[product_code],
                )
            ],
            local_cache=self.fda_code_identifier___fda_code_instance,
            perform_fetch_on_database=perform_fetch_on_database
        )

        if len(returned_value) == 1:
            return True, list(returned_value.values())[0]
        else:
            return False, None

    def add_fda_code(
            self,
            review_panel: str,
            medical_specialty: str,
            product_code: str,
            device_name: str,
            device_class: int,
            submission_type_id: int,
            definition: str,
            physical_state: str,
            technical_method: str,
            target_area: str,
            is_implant: bool,
            is_life_sustaining: bool,
    ) -> FdaCode:
        new_entity = self.add_data_to_database(
            are_data_already_present=lambda fetch_from_db: self.get_fda_code(
                product_code=product_code,
            ),
            add_data_to_database_callback=lambda _: self.table_handler.add_fda_code(
                product_code=product_code,
                device_name=device_name,
                device_class=device_class,
                panel=review_panel,
                medical_specialty=medical_specialty,
                submission_type_id=submission_type_id,
                definition=definition,
                physical_state=physical_state,
                technical_method=technical_method,
                target_area=target_area,
                is_implant=is_implant,
                is_life_sustaining=is_life_sustaining,
            ),
            local_cache=self.fda_code_identifier___fda_code_instance,
        )

        return new_entity
