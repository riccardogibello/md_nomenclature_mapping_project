import re

import numpy as np
from nltk import corpus


def clean_text(
        original_text: str | list[str] | np.ndarray,
        perform_classical_cleaning: bool = True
) -> str:
    """
    Clean the text by lowering it, removing non-ASCII characters, removing any substring between parentheses,
    removing multiple blank spaces and removing extra space at the beginning and end of the string.
    If perform_classical_cleaning is True, it also removes any digits, all the characters that are not letters,
    hyphens or spaces, removes any - character that is either at the beginning/end of the string,
    or is followed/preceded by a space, removes any string in the common_strings list.

    :param original_text: The text to clean.
    :param perform_classical_cleaning: Whether to perform the classical cleaning or not.

    :return: The cleaned text.
    """
    if type(original_text) is list:
        original_text = ' '.join(original_text)
    elif type(original_text) is np.ndarray:
        original_text = ' '.join(original_text.ravel())

    # Lower the text, to make it easier to process
    entire_text_lowered = original_text.lower()
    # Remove all the non-ASCII characters
    cleaned_string = str(entire_text_lowered.encode('ascii', errors='ignore').decode())

    # Remove any substring between parentheses
    cleaned_string = re.sub(r'\([^)]*\)', '', cleaned_string)

    if perform_classical_cleaning:
        # Remove any digits (due to the high number of chemical components in some names)
        # and all the characters that are not letters, hyphens or spaces
        cleaned_string = re.sub(r'[^a-z- ]', ' ', cleaned_string)
        # Remove any - character that is either at the beginning/end of the string,
        # or is followed/preceded by a space
        cleaned_string = re.sub(r" -", " ", cleaned_string)
        cleaned_string = re.sub(r"- ", " ", cleaned_string)
        cleaned_string = re.sub(r"^-", "", cleaned_string)
        cleaned_string = re.sub(r"-$", "", cleaned_string)

        common_strings = [
            'single-use',
            'other',
        ]
        # Get all the english stopwords
        common_strings.extend(corpus.stopwords.words('english'))

        # Replace any string in the common_strings list with an empty string
        cleaned_string = ' '.join([word for word in cleaned_string.split() if word not in common_strings])

    # Remove any extra spaces
    cleaned_string = re.sub(r" +", " ", cleaned_string)

    return cleaned_string.strip()
