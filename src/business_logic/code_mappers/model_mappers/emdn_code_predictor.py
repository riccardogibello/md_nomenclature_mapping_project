from typing import Optional

import pandas as pd
from sentence_transformers import SentenceTransformer
from torch import Tensor, stack, argsort

from src.business_logic.utilities.os_utilities import convert_to_device
from src.business_logic.code_mappers.model_mappers.clean_code_text import clean_text
from src.__directory_paths import SOURCE_DATA_DIRECTORY_PATH


class EmdnCodePredictor:
    emdn_nomenclature: pd.DataFrame = None
    pretrained_model: SentenceTransformer = None
    emdn_code_to_embedding: dict[str, Tensor] = None

    def __init__(
            self,
            train_emdn_categories: list[str],
            emdn_nomenclature_path: str,
            pretrained_model: Optional[SentenceTransformer] = None,
            emdn_code_to_embedding: Optional[dict[str, Tensor]] = None
    ):
        if self.pretrained_model is None and pretrained_model is None:
            raise ValueError("A SentenceTransformer model must be provided.")
        if self.pretrained_model is None:
            self.pretrained_model: SentenceTransformer = pretrained_model

        # Load the file containing the entire EMDN list of codes
        _file_df: pd.DataFrame = pd.read_csv(
            emdn_nomenclature_path,
            sep=';',
        )
        # Keep the columns corresponding to the EMDN category, EMDN code,
        # and EMDN description
        _file_df = _file_df.iloc[:, [1, 2, -3]]
        # Drop any missing value
        _file_df.dropna(inplace=True)
        # Clean all the EMDN descriptions by using the clean_text function
        _file_df.iloc[:, -1] = _file_df.iloc[:, -1].apply(
            clean_text,
            perform_classical_cleaning=False
        )
        _file_df.columns = ['category', 'code', 'description']
        self.emdn_nomenclature: pd.DataFrame = _file_df.copy()
        # Keep only the rows in which the 'category' value is contained in the training list
        self.emdn_nomenclature = self.emdn_nomenclature[
            self.emdn_nomenclature['category'].isin(train_emdn_categories)
        ]

        # Sort the EMDN nomenclature by the EMDN code from A to Z
        self.emdn_nomenclature.sort_values(
            by='code',
            inplace=True
        )

        if emdn_code_to_embedding is not None:
            self.emdn_code_to_embedding = emdn_code_to_embedding
        else:
            # Build a reference between the EMDN code and the description embedding
            self.emdn_code_to_embedding: dict[str, Tensor] = {
                code: self.pretrained_model.encode(
                    clean_text(description, perform_classical_cleaning=False),
                    convert_to_tensor=True,
                    device='cuda'
                )
                for code, description in zip(
                    self.emdn_nomenclature['code'],
                    self.emdn_nomenclature['description']
                )
            }

    def predict(
            self,
            gmdn_term_names: list[str] | str,
            gmdn_emdn_category_probabilities: Optional[dict[str, dict[str, float]]] = None
    ) -> dict[str, list[tuple[str, float]]]:
        if type(gmdn_term_names) is str:
            gmdn_term_names = [gmdn_term_names]

        self.pretrained_model = convert_to_device(
            self.pretrained_model,
            'cuda'
        )

        # Embed the given GMDN term name
        gmdn_tensors = self.pretrained_model.encode(
            [
                clean_text(_gmdn_term_name, perform_classical_cleaning=False)
                for _gmdn_term_name in gmdn_term_names
            ],
            convert_to_tensor=True,
            device='cuda'
        )

        _emdn_codes = list(self.emdn_code_to_embedding.keys())
        emdn_codes_tensors = stack(
            list(self.emdn_code_to_embedding.values())
        )
        # Compute the cosine similarity between the GMDN term name and each EMDN code
        similarity_matrix: Tensor = self.pretrained_model.similarity(
            gmdn_tensors,
            emdn_codes_tensors,
        )
        gmdn_mappings: dict[str, list[tuple[str, float]]] = {}
        # For each mapped GMDN code
        for _row, _gmdn_term_name in zip(similarity_matrix, gmdn_term_names):
            # Keep a dictionary in which the keys are the EMDN categories and the values are the
            # tuples containing the best EMDN code (with the related score)
            gmdn_emdn_code_scores: dict[str, list[tuple[str, float]]] = {}
            # Reorder the EMDN codes by the similarity with the GMDN term name
            sorted_indices = argsort(
                _row,
                descending=True
            )
            for _index in sorted_indices:
                code = _emdn_codes[_index]
                category = code[0]
                if gmdn_emdn_category_probabilities is not None:
                    category_probability = gmdn_emdn_category_probabilities.get(
                        _gmdn_term_name,
                        {}
                    ).get(
                        category,
                        0
                    )
                    similarity = _row[_index].item()
                    actual_score = similarity * category_probability
                else:
                    actual_score = _row[_index].item()
                if gmdn_emdn_code_scores.keys().__contains__(category):
                    # Append the current code and similarity to the list of codes
                    gmdn_emdn_code_scores[category].append((code, actual_score))
                else:
                    gmdn_emdn_code_scores[category] = [(code, actual_score)]

            # Extract a list of all the code - score correspondences for all the categories
            emdn_code_score_list: list[tuple[str, float]] = []
            for category, emdn_code_similarities in gmdn_emdn_code_scores.items():
                emdn_code_score_list.extend(emdn_code_similarities)
            # Reorder the codes based on their score
            reordered_emdn_codes: list[tuple[str, float]] = sorted(
                emdn_code_score_list,
                key=lambda item: item[1],
                reverse=True
            )
            # Save the ordered list in the map
            gmdn_mappings[_gmdn_term_name] = reordered_emdn_codes

        return gmdn_mappings
