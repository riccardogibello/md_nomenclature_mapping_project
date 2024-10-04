from src.data_model.abstract_entity import AbstractEntity
from src.data_model.nomenclature_codes.sentence_data import SentenceData


class Code(AbstractEntity):
    """
    Class that stores all the data related to a specific medical device src.
    Every src consists of (at least) a name (that is the official src) and a synthetic data.
    """

    def __init__(
            self,
            identifier: int,
            description_data: SentenceData
    ):
        """
        Initializes the class.

        :param identifier: The unique identifier used to identify the code in the database.
        :param description_data: It is a data structure which contains all the data related to the main string which
                                 identifies the code (e.g., code description for EMDN, Term Name for GMDN).
        """

        super().__init__(
            identifier=identifier
        )

        # The container for all the data related to the main string which identifies the code
        # (e.g., code description for EMDN, Term Name for GMDN).
        self.description_data: SentenceData = description_data
