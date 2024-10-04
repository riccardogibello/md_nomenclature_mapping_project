from typing import Optional, Callable, Any

from src.data_model.abstract_entity import AbstractEntity
from src.data_model.enums import HighRiskMedicalDeviceType
from src.__constants import UNDEFINED_STRING


class StandardizedDevice(AbstractEntity):

    def __init__(
            self,
            identifier: int,
            name: str,
            clean_manufacturer_id: int,
            catalogue_code: str,
            device_type: str,
            high_risk_medical_device_type_string: Optional[str] = UNDEFINED_STRING,
            ppc: Optional[str] = UNDEFINED_STRING,
    ):
        if high_risk_medical_device_type_string is None:
            high_risk_medical_device_type: HighRiskMedicalDeviceType = HighRiskMedicalDeviceType.NOT_SPECIFIED
        else:
            high_risk_medical_device_type: HighRiskMedicalDeviceType = HighRiskMedicalDeviceType.get_enum_from_value(
                high_risk_medical_device_type_string,
                enum_class=HighRiskMedicalDeviceType,
            )
        if ppc is None:
            ppc = UNDEFINED_STRING

        super().__init__(identifier)

        # The standardized name of the medical device
        self.name: str = str(name)

        # The manufacturer ID of the medical device
        self.manufacturer_id: int = int(clean_manufacturer_id)

        # The catalogue code of the medical device
        self.catalogue_code: str = str(catalogue_code)

        # The type of the medical device
        self.device_type: str = str(device_type)

        # The high risk medical device type of the medical device
        self.high_risk_medical_device_type: HighRiskMedicalDeviceType = high_risk_medical_device_type

        # The FDA product code of the medical device
        self.ppc = str(ppc)

    def to_json(
            self,
            key__lambda: Optional[dict[str, Callable[[Any], Any]]] = None,
            excluded_keys: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        This method converts the object to a JSON object. It is used to avoid useless data to be sent to the client
        (in particular, the original device name is not sent).

        :param key__lambda: A dictionary, in which each key is a field of the class and each value is a lambda
        function that takes the value of the field and returns the value to be added to the dictionary.
        :param excluded_keys: A list of keys that should be excluded from the dictionary.

        :return: A dictionary that can be converted to JSON.
        """
        return super().to_json(
            key__lambda={
                'high_risk_medical_device_type': lambda x: str(x.value),
            },
            excluded_keys=[
                str(self.name),
            ]
        )
