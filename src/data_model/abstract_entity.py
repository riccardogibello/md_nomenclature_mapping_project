from typing import Optional, Callable, Any


class AbstractEntity:
    identifier: int = -1

    def __init__(
            self,
            identifier: int,
    ):
        self.identifier = int(identifier)

    def to_json(
            self,
            key__lambda: Optional[dict[str, Callable[[Any], Any]]] = None,
            excluded_keys: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        It converts the current object to a dictionary that can be converted to JSON.

        :param key__lambda: A dictionary, in which each key is a field of the class and each value is a lambda
                            function that takes the value of the field and returns the value to be added to the
                            dictionary.
        :param excluded_keys: A list of keys that should be excluded from the dictionary.

        :return: A dictionary that can be converted to JSON.
        """
        json_dictionary: dict[str, Any] = {}

        # for each field of the class
        for key in self.__dict__.keys():
            if excluded_keys is None or not excluded_keys.__contains__(key):
                # if the field has a specific lambda to extract the corresponding value
                if key__lambda is not None and key in key__lambda.keys():
                    # extract the value and add it to the dictionary
                    json_dictionary[key] = key__lambda[key](self.__dict__[key])
                else:
                    # else, add the value to the dictionary
                    json_dictionary[key] = self.__dict__[key]

        return json_dictionary
