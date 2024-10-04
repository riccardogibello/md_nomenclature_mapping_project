from typing import Any, Optional, Callable

from src.__constants import UNDEFINED_VALUE
from src.data_model.manufacturers.standardized_manufacturer import StandardizedManufacturer


class CleanManufacturer(StandardizedManufacturer):

    def __init__(
            self,
            identifier: int,
            name: str,
            clean_name: str,
            standardized_manufacturer_id: Optional[int] = UNDEFINED_VALUE,
            original_state_identifier: int = UNDEFINED_VALUE,
    ):
        super().__init__(
            identifier,
            name
        )

        self.clean_name: str = str(clean_name)
        self.original_state_identifier: int = int(original_state_identifier)
        if standardized_manufacturer_id is None:
            standardized_manufacturer_id = UNDEFINED_VALUE
        self.standardized_manufacturer_id: int = int(standardized_manufacturer_id)

    def to_json(
            self,
            key__lambda: Optional[dict[str, Callable[[Any], Any]]] = None,
            excluded_keys: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        It converts the current CleanManufacturer instance to a JSON dictionary. In particular, it excludes the
        original name of the manufacturer, since it is not needed by the client.

        :param key__lambda: A dictionary that maps the keys of the JSON dictionary to a lambda function that
        transforms the value associated to the key.
        :param excluded_keys: A list of keys to be excluded from the JSON dictionary.

        :return: A JSON dictionary representing the current CleanManufacturer instance.
        """
        json_dictionary: dict[str, Any] = super().to_json(
            key__lambda=key__lambda,
            excluded_keys=[
                str(self.name)
            ]
        )

        return json_dictionary
