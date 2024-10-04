from typing import Optional

from src.data_model.abstract_entity import AbstractEntity
from src.data_model.enums import AbstractEnum
from src.__constants import UNDEFINED_VALUE, UNDEFINED_STRING


class SubmissionType(AbstractEnum):
    PRE_AMENDMENT = 1
    IDE = 2
    FOR_EXPORT_ONLY = 3
    UNKNOWN = 4
    GUIDANCE_UNDER_DEVELOPMENT = 5
    ENFORCEMENT_DISCRETION = 6
    NOT_FDA_REGULATED = 7

    def __str__(self):
        return self.name.replace("_", " ").title()


class FdaCode(AbstractEntity):

    def __init__(
            self,
            identifier: int,
            product_code: str,
            device_name: str,
            device_class: int,
            review_panel: Optional[str] = UNDEFINED_STRING,
            medical_specialty: Optional[str] = UNDEFINED_STRING,
            submission_type_id: Optional[int] = UNDEFINED_STRING,
            definition: Optional[int] = UNDEFINED_VALUE,
            physical_state: Optional[int] = UNDEFINED_VALUE,
            technical_method: Optional[int] = UNDEFINED_VALUE,
            target_area: Optional[int] = UNDEFINED_VALUE,
            is_implant: Optional[bool] = None,
            is_life_sustaining: Optional[bool] = None,
    ):
        super().__init__(identifier)

        self.review_panel: Optional[str] = str(review_panel)
        self.medical_specialty: Optional[str] = str(medical_specialty)
        self.product_code: str = str(product_code)
        self.device_name: str = str(device_name)
        self.device_class: int = int(device_class)
        if submission_type_id is not None:
            self.submission_type: Optional[SubmissionType] = SubmissionType.get_enum_from_value(
                value=submission_type_id,
                enum_class=SubmissionType,
            )
        else:
            self.submission_type: Optional[SubmissionType] = None
        self.definition: Optional[str] = str(definition)
        self.physical_state: Optional[str] = str(physical_state)
        self.technical_method: Optional[str] = str(technical_method)
        self.target_area: Optional[str] = str(target_area)
        self.is_implant: Optional[bool] = is_implant
        self.is_life_sustaining: Optional[bool] = is_life_sustaining
