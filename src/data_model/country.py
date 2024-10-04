from typing import Optional, Any, Callable

from src.data_model.abstract_entity import AbstractEntity
from src.data_model.enums import CountryName


class Country(AbstractEntity):
    name: CountryName
    is_american_state: bool

    def __init__(
            self,
            identifier: int,
            name: str,
            is_american_state: bool
    ):
        super().__init__(identifier)
        self.name = CountryName.get_enum_from_value(
            value=name,
            enum_class=CountryName,
        )
        self.is_american_state = is_american_state

    def to_json(
            self,
            key__lambda: Optional[dict[str, Callable[[Any], Any]]] = None,
            excluded_keys: Optional[list[str]] = None
    ):
        return super().to_json(
            key__lambda={
                **(key__lambda if key__lambda is not None else {}),
                'name': lambda name: name.value,
            },
            excluded_keys=excluded_keys,
        )
