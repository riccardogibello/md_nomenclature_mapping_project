from typing import Optional

from src.data_model.nomenclature_codes.gmdn_correspondence import GmdnCorrespondence
from src.__constants import UNDEFINED_STRING


class EmdnCorrespondence(GmdnCorrespondence):

    def __init__(
            self,
            correspondence_identifier: int,
            emdn_code_id: int,
            clean_device_id: int,
            match_type: str,
            similarity_value: Optional[float] = 0.0,
            matched_name: Optional[str] = UNDEFINED_STRING,
    ):
        super().__init__(
            correspondence_identifier=correspondence_identifier,
            code_id=emdn_code_id,
            clean_device_id=clean_device_id,
            match_type=match_type,
            similarity_value=similarity_value,
            matched_name=matched_name
        )
