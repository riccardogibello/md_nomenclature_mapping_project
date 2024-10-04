from src.data_model.abstract_entity import AbstractEntity


class StandardizedManufacturer(AbstractEntity):

    def __init__(
            self,
            manufacturer_id: int,
            name: str,
    ):
        super().__init__(
            manufacturer_id
        )

        self.name = str(name)
