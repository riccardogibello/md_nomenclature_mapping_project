from typing import Tuple, Optional

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.abstract_data_handler import get_instances
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.code_correspondence_data_handler import \
    CodeCorrespondenceDataHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.nomenclature_table_handlers.gmdn_correspondence_table_handler import \
    GmdnCorrespondenceTableHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.nomenclature_codes.emdn_correspondence import EmdnCorrespondence
from src.data_model.nomenclature_codes.gmdn_correspondence import GmdnCorrespondence
from src.data_model.enums import MatchType


class GmdnCorrespondenceDataHandler(CodeCorrespondenceDataHandler):

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            init_cache: Optional[bool] = False,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            table_handler=GmdnCorrespondenceTableHandler(
                connection_handler=connection_handler,
                reset_table=reset_table
            )
        )

        self.table_handler: GmdnCorrespondenceTableHandler = self.table_handler

        if init_cache:
            self.get_gmdn_correspondences()

    def get_gmdn_correspondences(
            self,
            clean_device_ids: Optional[list[int]] = None,
            correspondence_ids: Optional[list[int]] = None,
            fetch_from_database: Optional[bool] = True,
    ) -> Optional[dict[int, GmdnCorrespondence] | GmdnCorrespondence | None]:
        if fetch_from_database:
            gmdn_device_correspondences = super().fetch_correspondences(
                correspondence_table_handler=self.table_handler,
                clean_device_ids=clean_device_ids,
                correspondence_ids=correspondence_ids
            )
        else:
            gmdn_device_correspondences = self.correspondence_id___correspondence_instance.copy()

        return gmdn_device_correspondences

    def get_gmdn_device_correspondence(
            self,
            clean_device_id: int,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> Tuple[bool, Optional[EmdnCorrespondence]]:
        returned_value = get_instances(
            fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                filters=_filters,
                instance_builder=lambda values: GmdnCorrespondence(
                    *values
                ),
            ),
            sql_filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.CLEAN_DEVICE_ID.value,
                    column_values=[clean_device_id],
                )
            ],
            local_cache=self.correspondence_id___correspondence_instance,
            perform_fetch_on_database=perform_fetch_on_database
        )

        if len(returned_value) == 1:
            return True, list(returned_value.values())[0]
        else:
            return False, None

    def add_correspondence(
            self,
            gmdn_code_id: int,
            clean_device_id: int,
            match_type: MatchType,
            similarity_value: Optional[float] = None,
            matched_device_name: Optional[str] = None,
    ) -> EmdnCorrespondence:
        new_entity = self.add_data_to_database(
            are_data_already_present=lambda fetch_from_db: self.get_gmdn_device_correspondence(
                clean_device_id=clean_device_id,
            ),
            add_data_to_database_callback=lambda _: self.table_handler.add_correspondence(
                gmdn_id=gmdn_code_id,
                clean_device_id=clean_device_id,
                match_type=match_type,
                similarity_value=similarity_value,
                matched_device_name=matched_device_name
            ),
            local_cache=self.correspondence_id___correspondence_instance,
        )

        return new_entity
