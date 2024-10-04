from typing import Optional, Tuple, Union

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.abstract_data_handler import AbstractDataHandler, get_instances
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.fetch_codes import fetch_codes
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.nomenclature_table_handlers.gmdn_code_table_handler import \
    GmdnCodeTableHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.nomenclature_codes.gmdn_code import GmdnCode


class GmdnCodeDataHandler(AbstractDataHandler):

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
            init_cache: Optional[bool] = False,
    ):
        super().__init__(
            table_handler=GmdnCodeTableHandler(
                connection_handler=connection_handler,
                reset_table=reset_table
            )
        )

        self.gmdn_code_identifier___gmdn_code_instance: dict[int, GmdnCode] = {}

        # Set the GMDN table handler
        self.table_handler: GmdnCodeTableHandler = self.table_handler

        # If the cache must be initialized
        if init_cache:
            # Perform a query to fetch all the GMDN codes
            self.get_gmdn_codes()

    def get_cached_code(
            self,
            identifier: Optional[int] = None,
            gmdn_term_name: Optional[str] = None,
    ) -> Optional[Union[GmdnCode, list[GmdnCode]]]:
        """
        This method returns the list of GMDN src entities stored in the current data structures.

        :return: The list of GMDN src entities stored in the current data structures.
        """
        if identifier is not None:
            try:
                return self.gmdn_code_identifier___gmdn_code_instance[identifier]
            except KeyError:
                return None
        elif gmdn_term_name is not None:
            for gmdn_code in self.gmdn_code_identifier___gmdn_code_instance.values():
                if gmdn_code.description_data.sentence == gmdn_term_name:
                    return gmdn_code
            return None
        else:
            return list(self.gmdn_code_identifier___gmdn_code_instance.values())

    def store_in_cache(
            self,
            added_code: GmdnCode
    ) -> None:
        self.gmdn_code_identifier___gmdn_code_instance[added_code.identifier] = added_code

    def is_gmdn_code_already_existing(
            self,
            gmdn_term_name: str,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> Tuple[bool, Optional[GmdnCode]]:
        returned_value = get_instances(
            fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                filters=_filters,
                instance_builder=lambda values: GmdnCode(
                    *values
                ),
            ),
            sql_filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.TERM_NAME.value,
                    column_values=[gmdn_term_name],
                )
            ],
            local_cache=self.gmdn_code_identifier___gmdn_code_instance,
            perform_fetch_on_database=perform_fetch_on_database
        )

        if len(returned_value) == 1:
            return True, list(returned_value.values())[0]
        else:
            return False, None

    def add_gmdn_code(
            self,
            gmdn_term_name: str,
            gmdn_definition: str,
    ) -> GmdnCode:
        return self.table_handler.add_gmdn_code(
            term_name=gmdn_term_name,
            definition=gmdn_definition,
        )

    def get_gmdn_codes(
            self,
            identifiers: Optional[Union[int, list[int]]] = None,
            gmdn_term_names: Optional[Union[str, list[str]]] = None,
            fetch_from_database: Optional[bool] = True,
    ) -> Union[None, GmdnCode, dict[int, GmdnCode]]:
        """
        This method gets the GMDN codes from the cache and the database, if required, either
        by their GMDN Term Names or by their identifiers.

        :param identifiers: The identifiers of the GMDN codes to be fetched.
        :param gmdn_term_names: The term names of the GMDN codes to be fetched.
        :param fetch_from_database: A flag indicating whether the GMDN codes must be fetched from the database.

        :return:    The searched GMDN codes, either in a single instance or in a dictionary of instances.
                    None, if the searched GMDN codes do not exist.
        """
        # Prepare a map in which to store the returned GMDN codes, either
        # fetched from the cache and the database
        returned_codes: dict[int, GmdnCode] = {}

        # Prepare the list of GMDN identifiers to search for (either the ids
        # or the term names)
        searched_values: list[int | str]
        # If the GMDN identifiers are given
        if identifiers is not None:
            # Set the searched identifiers
            searched_values = identifiers if type(identifiers) is list else [identifiers]
        elif gmdn_term_names is not None:
            # Set the searched term names
            searched_values = gmdn_term_names if type(gmdn_term_names) is list else [gmdn_term_names]
        else:
            # Leave the searched values empty, so that all the codes are fetched
            # from the database
            searched_values = []

        returned_ids: list[int | str] = []
        # If all the GMDN codes must be fetched
        if len(searched_values) == 0:
            # Add all the cached GMDN codes to the returned codes
            returned_codes = self.gmdn_code_identifier___gmdn_code_instance.copy()
            returned_ids = list(returned_codes.keys())
        else:
            # Search for the cached GMDN codes
            for searched_value in searched_values:
                if type(searched_value) is int:
                    code = self.get_cached_code(
                        identifier=searched_value
                    )
                else:
                    code = self.get_cached_code(
                        gmdn_term_name=searched_value
                    )
                # If the GMDN code exists in the cache
                if code is not None:
                    # Add the GMDN code to the returned codes
                    returned_codes[code.identifier] = code
                    if type(searched_value) is int:
                        returned_ids.append(code.identifier)
                    else:
                        returned_ids.append(code.description_data.sentence)

        # Set the fetch codes callback to fetch the GMDN codes from the database if it is required,
        # otherwise set it to None so that the fetch_codes method does not try to fetch the codes
        # from the database
        fetch_codes_callback = self.table_handler.fetch_gmdn_codes if fetch_from_database else None

        are_searched_codes_given = len(searched_values) > 0
        are_searched_codes_found = are_searched_codes_given and list(searched_values) == list(returned_ids)
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
                store_in_cache=self.store_in_cache,
                fetch_codes_callback=fetch_codes_callback,
            )
