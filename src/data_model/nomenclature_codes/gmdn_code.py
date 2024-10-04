from src.data_model.nomenclature_codes.code import Code
from src.data_model.nomenclature_codes.sentence_data import SentenceData


class GmdnCode(Code):
    definition_data: SentenceData

    def __init__(
            self,
            identifier: int,
            term_name: str,
            definition: str,
    ):
        super().__init__(
            identifier,
            description_data=SentenceData(
                sentence=term_name,
            )
        )

        self.definition_data = SentenceData(
            sentence=definition,
        )
