from enum import Enum
from typing import Optional, Any, Union


class SafeMapAdditionOutcome(Enum):
    ADDED = 0,
    NEW_KEY_ADDED = 1,
    ALREADY_PRESENT = 2,
    REPLACED = 3


def safe_add_to_dictionary(
        _dict: dict[object, Union[Any, list[Any]]],
        key: object,
        value_to_add: Any,
        is_value_list: Optional[bool] = True,
        avoid_duplicates: Optional[bool] = True,
) -> SafeMapAdditionOutcome:
    """
    This function adds a value to a dictionary, either as a single value or as a list of values. The addition of the
    new value is dependent on the value of the parameters is_value_list and avoid_duplicates.
    If the given key is not present in the dictionary, a new key-value pair is added. If the key is present and the
    value is not a list, the existing value is replaced with the new one. If the key is present and the value is a list,
    the new value is appended to the list, unless the value is already present in the list and the duplicates are to be
    avoided.

    :param _dict: The dictionary to which the value is to be added.
    :param key: The key to which the value is to be added.
    :param value_to_add: The value to be added to the dictionary.
    :param is_value_list: A boolean indicating whether the value is a list or not.
    :param avoid_duplicates: A boolean indicating whether the duplicates are to be avoided or not, if the value is a
    list.

    :return: An enum indicating the outcome of the addition operation.
    """
    # If the dictionary contains the given key
    if _dict.keys().__contains__(key):
        # If the value is a list
        if is_value_list:
            # If the list already contains the value to add and the duplicates are to be avoided
            if _dict[key].__contains__(value_to_add) and avoid_duplicates:
                return SafeMapAdditionOutcome.ALREADY_PRESENT
            else:
                _dict[key].append(value_to_add)
                return SafeMapAdditionOutcome.ADDED
        else:
            # Replace the existing value with the new one
            _dict[key] = value_to_add
            return SafeMapAdditionOutcome.REPLACED
    else:
        # Add a new key-value pair
        if is_value_list:
            _dict[key] = [value_to_add]
        else:
            _dict[key] = value_to_add

        return SafeMapAdditionOutcome.NEW_KEY_ADDED


def safe_remove_from_dictionary(
        _dict: dict[object, Union[Any, list[Any]]],
        key: object,
        value_to_remove: Optional[Any] = None,
        is_value_list: Optional[bool] = True,
        remove_list_if_empty: Optional[bool] = True,
) -> Any:
    """
    This method safely removes a value from a dictionary. If the dictionary contains the given key, the value is removed
    from the list corresponding to the key, if the value is a list. If the value is not a list, the key-value pair is
    removed. If the list corresponding to the key is empty after the removal of the value, the key-value pair is removed
    from the dictionary.

    :param _dict: The dictionary from which the value is to be removed.
    :param key: The key from which the value is to be removed.
    :param value_to_remove: The value to be removed from the dictionary.
    :param is_value_list: A boolean indicating whether the value is a list or not.
    :param remove_list_if_empty: A boolean indicating whether the list corresponding to the key is to be removed
    if it is empty after the removal of the value.

    :return: The removed instance.
    """
    removed_instance: Any = None
    if is_value_list and value_to_remove is None:
        raise Exception()

    # if the dictionary contains the given key
    if _dict.keys().__contains__(key):
        # if the value is a list
        if is_value_list:
            _list = _dict[key]
            i = 0
            # iterate over all the elements of the list
            while i < len(_list):
                tmp_instance = _list[i]

                # if the tmp_instance is equal to the one to be removed
                if tmp_instance == value_to_remove:
                    _list.remove(tmp_instance)
                    removed_instance = tmp_instance
                else:
                    i += 1

            # if the list corresponding to the key is empty,
            if remove_list_if_empty and len(_list) == 0:
                _dict.pop(key)
        else:
            # otherwise, remove the key-value entry
            removed_instance = _dict.pop(key)

    return removed_instance
