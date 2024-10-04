from typing import Optional, Callable, Any

from src.data_model.mapping.abstract_mapping import AbstractMapping


class NomenclatureMapping(AbstractMapping):

    def __init__(
            self,
            identifier: int,
            emdn_id: int,
            gmdn_id: int,
    ):
        super().__init__(
            identifier=identifier,
            emdn_id=emdn_id,
            gmdn_id=gmdn_id,
        )

    def to_json(
            self,
            key__lambda: Optional[dict[str, Callable[[Any], Any]]] = None,
            excluded_keys: Optional[list[str]] = None,
    ) -> dict[str, any]:
        return super().to_json(
            key__lambda={
                **(key__lambda if key__lambda is not None else {}),
            },
            excluded_keys=excluded_keys,
        )

    def copy_from(
            self,
            other_instance: "NomenclatureMapping"
    ) -> None:
        super().copy_from(other_instance)
