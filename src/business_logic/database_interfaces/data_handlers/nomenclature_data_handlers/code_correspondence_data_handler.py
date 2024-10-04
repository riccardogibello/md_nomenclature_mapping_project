from typing import Optional, Any

from src.business_logic.database_interfaces.data_handlers.abstract_data_handler import AbstractDataHandler
from src.business_logic.database_interfaces.table_handlers.nomenclature_table_handlers.emdn_correspondence_table_handler import \
    EmdnCorrespondenceTableHandler
from src.business_logic.database_interfaces.table_handlers.nomenclature_table_handlers.gmdn_correspondence_table_handler import \
    GmdnCorrespondenceTableHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.abstract_entity import AbstractEntity
from src.data_model.nomenclature_codes.emdn_correspondence import EmdnCorrespondence
from src.data_model.nomenclature_codes.gmdn_correspondence import GmdnCorrespondence


class CodeCorrespondenceDataHandler(AbstractDataHandler):

    def __init__(
            self,
            table_handler: GmdnCorrespondenceTableHandler | EmdnCorrespondenceTableHandler
    ):
        super().__init__(
            table_handler=table_handler
        )

        self.correspondence_id___correspondence_instance: dict[
            int, GmdnCorrespondence | EmdnCorrespondence
        ] = {}

    def fetch_correspondences(
            self,
            correspondence_table_handler: GmdnCorrespondenceTableHandler | EmdnCorrespondenceTableHandler,
            clean_device_ids: Optional[list[int]] = None,
            correspondence_ids: Optional[list[int]] = None
    ) -> Optional[dict[int, GmdnCorrespondence | EmdnCorrespondence]]:
        if clean_device_ids is not None and correspondence_ids is not None:
            raise Exception("Both clean device ids and correspondence ids cannot be provided at the same time.")
        else:
            field_name__value_list = None
            if clean_device_ids is not None or correspondence_ids is not None:
                if clean_device_ids is not None:
                    field_name = TableColumnName.CLEAN_DEVICE_ID.value
                    values = clean_device_ids
                else:
                    field_name = TableColumnName.IDENTIFIER.value
                    values = correspondence_ids
                field_name__value_list = {
                    field_name: values,
                }

            fetched_instances: Any = correspondence_table_handler.fetch(
                filters=field_name__value_list,
                instance_builder=lambda _values: EmdnCorrespondence(
                    *_values
                ),
            )

            if fetched_instances is not None:
                if type(fetched_instances) is AbstractEntity:
                    self.correspondence_id___correspondence_instance[fetched_instances.identifier] = fetched_instances
                else:
                    # Update the local cache with the new dictionary
                    self.correspondence_id___correspondence_instance.update(fetched_instances)

                return fetched_instances.copy()
            else:
                return None
