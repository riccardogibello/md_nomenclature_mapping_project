from typing import Union, Callable, Optional

from src.data_model.nomenclature_codes.emdn_code import EmdnCode
from src.data_model.nomenclature_codes.gmdn_code import GmdnCode


def fetch_codes(
        searched_values: list[Union[int, str]],
        returned_codes: dict[int, Union[EmdnCode, GmdnCode]],
        fetch_codes_callback: Callable[
            [Union[list[int], list[str]], bool],
            Optional[GmdnCode | EmdnCode | dict[int, EmdnCode | GmdnCode]]
        ] | None,
        store_in_cache: Callable[[Union[EmdnCode, GmdnCode]], None],
) -> Union[None, Union[EmdnCode, GmdnCode], dict[int, Union[EmdnCode, GmdnCode]]]:
    # If the searched values are empty, or the first value is int (identifier)
    if searched_values is None or (len(searched_values) > 0 and type(searched_values[0]) is int):
        # Find a list of the cached code identifiers
        retrieved_values = list([code.identifier for code in returned_codes.values()])
    else:
        # Find a list of the cached code sentence data (EMDN description or GMDN term name)
        retrieved_values = list([code.description_data.sentence for code in returned_codes.values()])

    # Find a list of the missing code identifiers
    missing = list(set(searched_values) - set(retrieved_values))
    if fetch_codes_callback is not None:
        # Fetch the missing EMDN codes from the database
        missing_gmdn_codes = fetch_codes_callback(
            # If the method is trying to fetch all the codes, then set the identifiers to the ones
            # already in cache (to be excluded); otherwise, set the missing identifiers to be fetched
            retrieved_values if len(searched_values) == 0 else missing,
            # If the method is trying to fetch all the codes, then set the is_search_set=False to exclude
            # the already fetched codes; otherwise, set the is_search_set=True to fetch the missing codes;
            False if len(searched_values) == 0 else True,
        )
    else:
        missing_gmdn_codes = {}

    # Extend the initial cached list of codes with the missing ones
    # and add each of them to the local cache
    if type(missing_gmdn_codes) is GmdnCode or type(missing_gmdn_codes) is EmdnCode:
        store_in_cache(missing_gmdn_codes)
        returned_codes[missing_gmdn_codes.identifier] = missing_gmdn_codes
    elif type(missing_gmdn_codes) is dict:
        for code in missing_gmdn_codes.values():
            store_in_cache(code)
            returned_codes[code.identifier] = code

    if len(returned_codes) == 0:
        return None
    elif len(returned_codes) == 1:
        return list(returned_codes.values())[0]
    else:
        return returned_codes
