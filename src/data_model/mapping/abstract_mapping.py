from typing import Optional, Callable, Any

from src.data_model.abstract_entity import AbstractEntity


class AbstractMapping(AbstractEntity):

    def __init__(
            self,
            identifier: int,
            emdn_id: int,
            gmdn_id: int,
    ):
        super().__init__(identifier)

        # The identifier of the EMDN code involved in the mapping.
        self.emdn_id: int = int(emdn_id)

        # The identifier of the GMDN code involved in the mapping.
        self.gmdn_id: int = int(gmdn_id)

    def to_json(
            self,
            key__lambda: Optional[dict[str, Callable[[Any], Any]]] = None,
            excluded_keys: Optional[list[str]] = None,
    ) -> dict[str, any]:
        return super().to_json(
            key__lambda={
                **(key__lambda if key__lambda is not None else {})
            },
            excluded_keys=excluded_keys,
        )

    def copy_from(
            self,
            other_instance: "AbstractMapping"
    ) -> None:
        self.identifier = other_instance.identifier
        self.emdn_id = other_instance.emdn_id
        self.gmdn_id = other_instance.gmdn_id
