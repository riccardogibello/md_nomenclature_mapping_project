from typing import Optional, Tuple, Union, Callable

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.abstract_data_handler import AbstractDataHandler, get_instances
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.fetch_codes import fetch_codes
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.nomenclature_table_handlers.emdn_code_table_handler import \
    EmdnCodeTableHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.business_logic.model_handlers.emdn_tree_handler import EmdnTreeHandler
from src.data_model.nomenclature_codes.emdn_code import EmdnCode


class EmdnCodeDataHandler(AbstractDataHandler):

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            init_cache: Optional[bool] = False,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            table_handler=EmdnCodeTableHandler(
                connection_handler=connection_handler,
                reset_table=reset_table,
            )
        )

        # Instantiate the table handler for retrieving data from the EMDN codes table
        self.table_handler: EmdnCodeTableHandler = self.table_handler

        # Instantiate the handler of the EMDN tree structure (cache)
        self.emdn_tree_handler: EmdnTreeHandler = EmdnTreeHandler()

        # If the cache must be initialized,
        if init_cache:
            # Fetch all the EMDN codes from the database
            self.get_emdn_codes()

    def get_code(
            self,
            emdn_code: Optional[str] = None,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> Tuple[bool, Optional[EmdnCode]]:
        # Try to get the searched EMDN code from the local tree
        emdn_code_instance: Optional[EmdnCode] = self.emdn_tree_handler.get_code(
            alphanumeric_code=emdn_code
        )
        if emdn_code_instance is not None:
            return True, emdn_code_instance
        else:
            # Try to search in the database
            result = get_instances(
                fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                    instance_builder=lambda values: EmdnCode(
                        *values
                    ),
                    filters=_filters,
                ),
                sql_filters=[
                    SqlQueryComponent(
                        column_name=TableColumnName.CODE.value,
                        column_values=[emdn_code],
                    )
                ],
                local_cache=None,
                perform_fetch_on_database=perform_fetch_on_database,
            )

            if len(result) > 0:
                emdn_code_instance = list(result.values())[0]
                # Add the instance to the cache
                self.emdn_tree_handler.add_to_tree(emdn_code_instance)

                return True, emdn_code_instance
            else:
                return False, None

    def get_emdn_codes(
            self,
            alphanumeric_codes: Optional[Union[str, list[str]]] = None,
            identifiers: Optional[Union[int, list[int]]] = None,
            fetch_from_database: Optional[bool] = True,
    ) -> Union[None, EmdnCode, dict[int, EmdnCode]]:
        """
        This method gets the EMDN codes from the cache and the database, if required, either
        by their alphanumeric codes or by their identifiers.

        :param alphanumeric_codes: The EMDN alphanumeric codes to be fetched.
        :param identifiers: The EMDN identifiers to be fetched.
        :param fetch_from_database: A boolean indicating whether the EMDN codes should be fetched from the database
                                    if not already in the cache.

        :return: The searched EMDN codes, if found, either in a single instance or in a dictionary. None, otherwise.
        """
        # Prepare the list of GMDN identifiers to search for (either the ids
        # or the term names)
        searched_values: list[int | str]
        # If the search is by EMDN identifier
        if identifiers is not None:
            # Set the function callback to retrieve the codes from the cache
            cache_search_function: Callable = lambda identifier: self.emdn_tree_handler.get_code(
                alphanumeric_code=None,
                identifier=identifier
            )
            # Set the searched identifiers
            searched_values = identifiers if type(identifiers) is list else [identifiers]
        # If the search is by EMDN alphanumeric code
        elif alphanumeric_codes is not None:
            cache_search_function: Callable = lambda alpha_code: self.emdn_tree_handler.get_code(
                alphanumeric_code=alpha_code,
                identifier=None,
            )
            searched_values = alphanumeric_codes if type(alphanumeric_codes) is list else [alphanumeric_codes]
        # If all the EMDN codes must be fetched
        else:
            # Fetch all the EMDN identifiers from the database
            searched_values = []
            # Set the function callback to retrieve the codes from the cache
            # to a function that always returns None
            cache_search_function: Callable = lambda _: None

        # Prepare a map in which to store the returned GMDN codes, either
        # fetched from the cache and the database
        returned_codes: dict[int, EmdnCode] = {}
        returned_identifiers: list[int | str] = []
        # If all the EMDN codes must be fetched
        if len(searched_values) == 0:
            # Get all the codes stored in the cache
            returned_codes, returned_identifiers = self.emdn_tree_handler.get_codes()
        else:
            # For each searched value
            for searched_value in searched_values:
                # Get the EMDN code from the cache with the right callback
                emdn_code = cache_search_function(
                    searched_value
                )
                # If the EMDN code was found in the cache
                if emdn_code is not None:
                    # Add it to the returned codes
                    returned_codes[emdn_code.identifier] = emdn_code
                    # Add the identifier to the list of returned identifiers
                    if type(searched_value) is int:
                        returned_identifiers.append(emdn_code.identifier)
                    else:
                        returned_identifiers.append(emdn_code.emdn_code)

        # Set the fetch codes callback to fetch the GMDN codes from the database if it is required,
        # otherwise set it to None so that the fetch_codes method does not try to fetch the codes
        # from the database
        fetch_codes_callback = self.table_handler.fetch_emdn_codes if fetch_from_database else None
        are_searched_codes_given = len(searched_values) > 0
        are_searched_codes_found = are_searched_codes_given and list(searched_values) == list(returned_identifiers)
        if are_searched_codes_found:
            if len(returned_codes) == 0:
                return None
            elif len(returned_codes) == 1:
                return list(returned_codes.values())[0]
            else:
                return returned_codes
        else:
            return fetch_codes(
                searched_values=searched_values,
                returned_codes=returned_codes,
                store_in_cache=self.emdn_tree_handler.add_to_tree,
                fetch_codes_callback=fetch_codes_callback,
            )

    def add_code(
            self,
            emdn_code_string: str,
            emdn_description: str,
            is_leaf,
    ) -> EmdnCode:
        # Add the new emdn code to the database, if not already in the database (but not to the cache)
        added_emdn_code = self.add_data_to_database(
            are_data_already_present=lambda fetch_from_db: self.get_code(
                emdn_code=emdn_code_string,
            ),
            add_data_to_database_callback=lambda _: self.table_handler.add_code(
                emdn_code_string=emdn_code_string,
                emdn_description=emdn_description,
                is_leaf=is_leaf,
            ),
        )

        self.emdn_tree_handler.add_to_tree(added_emdn_code)

        return added_emdn_code
