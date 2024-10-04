from typing import Optional

from src.data_model.devices.standardized_device import StandardizedDevice
from src.__constants import UNDEFINED_VALUE


class CleanDevice(StandardizedDevice):

    def __init__(
            self,
            identifier: int,
            original_name: str,
            clean_manufacturer_id: int,
            catalogue_code: str,
            clean_name: str,
            device_type: str,
            high_risk_medical_device_type_string: Optional[str] = None,
            ppc: Optional[str] = None,
            standardized_device_id: Optional[int] = UNDEFINED_VALUE,
    ):
        super().__init__(
            identifier,
            original_name,
            clean_manufacturer_id,
            catalogue_code,
            device_type,
            high_risk_medical_device_type_string,
            ppc,
        )

        # The name of the medical device after cleaning.
        self.clean_name = str(clean_name)

        if standardized_device_id is None:
            standardized_device_id = UNDEFINED_VALUE
        # The standardized device id related to the current cleaned device instance.
        self.standardized_device = int(standardized_device_id)
