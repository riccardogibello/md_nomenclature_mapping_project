from src.data_model.abstract_entity import AbstractEntity


class DeviceMapping(AbstractEntity):

    def __init__(
            self,
            identifier: int,
            mapping_id: int,
            first_device_id: int,
            second_device_id: int,
            company_name_similarity: float,
            similarity: float,
    ):
        super().__init__(identifier)

        self.mapping_id = int(mapping_id)
        self.first_device_id = int(first_device_id)
        self.second_device_id = int(second_device_id)
        self.company_name_similarity = float(company_name_similarity)
        self.similarity = float(similarity)
