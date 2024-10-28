from typing import Optional

import pandas as pd
from torch import Tensor

from src.business_logic.code_mappers.model_mappers.emdn_category_predictors.neural_model_mapper.emdn_category_predictor import \
    EmdnCategoryPredictor
from src.business_logic.code_mappers.model_mappers.emdn_code_predictor import EmdnCodePredictor


class DataDrivenEmdnCodePredictor(EmdnCodePredictor):
    emdn_category_predictor: EmdnCategoryPredictor = None
    fda_distribution_over_emdn: pd.DataFrame = None

    def __init__(
            self,
            emdn_nomenclature_path: str,
            _emdn_category_predictor: Optional[EmdnCategoryPredictor] = None,
            emdn_code_to_embedding: Optional[dict[str, Tensor]] = None
    ):
        if self.emdn_category_predictor is None:
            if _emdn_category_predictor is not None:
                self.emdn_category_predictor = _emdn_category_predictor
            else:
                raise ValueError("An EmdnCategoryPredictor must be provided.")

        super().__init__(
            pretrained_model=self.emdn_category_predictor.model.embedding_layer,
            emdn_nomenclature_path=emdn_nomenclature_path,
            emdn_code_to_embedding=emdn_code_to_embedding,
            train_emdn_categories=list(self.emdn_category_predictor.labels_dictionary.values())
        )

    def predict(
            self,
            gmdn_term_names: list[str] | str,
            gmdn_emdn_category_probabilities: Optional[dict[str, float]] = None
    ) -> dict[str, list[tuple[str, float]]]:
        """
        This method returns a dictionary in which every GMDN Term Name is associated to a list of tuples. Every tuple
        contain an EMDN code and the related probability of being the correct code for the GMDN Term Name. The list
        is ordered from the most probable to the least probable EMDN code.

        :param gmdn_term_names:
        :param gmdn_emdn_category_probabilities:
        :return:
        """
        # If only one GMDN term name is provided, convert it to a list
        if type(gmdn_term_names) is str:
            gmdn_term_names = [gmdn_term_names]
        # Identify the probabilities of belonging to each EMDN category
        predicted_emdn_category_probabilities = self.emdn_category_predictor.predict(
            input_texts=gmdn_term_names
        )
        gmdn_emdn_probabilities: dict[str, dict[str, float]] = {}
        for _gmdn_term_name, _predicted_emdn_category in zip(
                gmdn_term_names,
                predicted_emdn_category_probabilities
        ):
            # Get the map containing the EMDN categories and the related probabilities
            gmdn_predicted_emdn_categories = _predicted_emdn_category
            gmdn_emdn_probabilities[_gmdn_term_name] = gmdn_predicted_emdn_categories

        return super().predict(
            gmdn_term_names=gmdn_term_names,
            gmdn_emdn_category_probabilities=gmdn_emdn_probabilities
        )
