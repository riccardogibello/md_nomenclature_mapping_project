from typing import Optional

from src.data_model.abstract_entity import AbstractEntity
from src.data_model.enums import MatchType
from src.__constants import UNDEFINED_STRING


class GmdnCorrespondence(AbstractEntity):

    def __init__(
            self,
            correspondence_identifier: int,
            code_id: int,
            clean_device_id: int,
            match_type: str,
            similarity_value: Optional[float] = 0.0,
            matched_name: Optional[str] = UNDEFINED_STRING,
    ):
        if similarity_value is None:
            similarity_value = 0.0
        if matched_name is None:
            matched_name = UNDEFINED_STRING

        super().__init__(
            identifier=correspondence_identifier
        )

        match_type: MatchType = MatchType.get_enum_from_value(
            match_type,
            enum_class=MatchType,
        )

        self.code_id: int = int(code_id)
        self.clean_device_id: int = int(clean_device_id)
        self.match_type: MatchType = match_type
        self.similarity_value: Optional[float] = float(similarity_value)
        self.matched_name = str(matched_name)
