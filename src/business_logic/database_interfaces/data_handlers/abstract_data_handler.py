from abc import ABC
from enum import Enum
from typing import Optional, Callable, Any, Tuple, Union, List

from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent, \
    AbstractTableHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.abstract_entity import AbstractEntity


class TableDataHandlerType(Enum):
    MAPPING_ERROR_NOTIFICATION = 0


def get_instances(
        fetch_information_from_database: Callable[[List[SqlQueryComponent] | None], AbstractEntity | dict | None],
        local_cache: Optional[dict[int, AbstractEntity]] = None,
        sql_filters: Optional[List[SqlQueryComponent]] = None,
        perform_fetch_on_database: Optional[bool] = True,
) -> dict:
    """
    This method is used to check if the searched information already exists in the cache. It checks also the
    database, if it is required to and the information is not present in the cache.

    :param fetch_information_from_database: The lambda function to fetch the information from the database
                                            interacting with a specific table handler.
    :param sql_filters: A list of filters to make the query on the database.
    :param local_cache: The local cache used to index the information in the specific Data Handler.
                        This cache indexes are the identifiers of the indexed instances.
    :param perform_fetch_on_database:   A boolean value to indicate whether the fetch must be performed on the database
                                        if the information is not present in the cache.

    :return: True and the class containing the information, if it exists. False and None otherwise.
    """
    actual_filters = None
    returned_map = {}
    if sql_filters is not None:
        # If the filter is unique, the search is made by the identifier, and the
        # filter is for inclusion
        if (
                len(sql_filters) == 1 and
                sql_filters[0].column_name == TableColumnName.IDENTIFIER.value and
                sql_filters[0].must_be_comprised is True
        ):
            missing_identifiers = []

            # Take the searched identifiers
            searched_identifiers: List[int] = sql_filters[0].column_values
            # For each identifier that is searched
            for _id in searched_identifiers:
                # If the local cache does not contain it
                if local_cache is None or not local_cache.keys().__contains__(int(_id)):
                    missing_identifiers.append(int(_id))
                else:
                    # Add the instance related to the identifier to the returned results
                    returned_map[int(_id)] = local_cache[int(_id)]
        else:
            # In this case, the cache is not considered and the database fetch must be performed
            # with the given filters
            actual_filters = sql_filters
    else:
        # Set all the cached instances as returned
        if local_cache is not None:
            returned_map = local_cache
            cached_identifiers = list(local_cache.keys())
            if len(cached_identifiers) > 0:
                # Set the identifiers not to be fetched in the filters
                actual_filters = [
                    SqlQueryComponent(
                        column_name=TableColumnName.IDENTIFIER.value,
                        column_values=cached_identifiers,
                        must_be_comprised=False
                    )
                ]
            else:
                actual_filters = sql_filters

    # If the filters are not given, or there is at least one filter
    if perform_fetch_on_database and (actual_filters is None or len(actual_filters) > 0):
        # Fetch the missing instances from the database by passing the computed filters
        new_information_instance = fetch_information_from_database(
            actual_filters
        )
        # If the returned information is not None
        if new_information_instance is not None:
            # If there are multiple instances
            if type(new_information_instance) is dict:
                # If the local cache must be updated
                if local_cache is not None:
                    # Update the cache with the new instances
                    local_cache.update(new_information_instance)
                # Update the returned map with the new instances
                returned_map.update(new_information_instance)
            else:
                # If the local cache must be updated
                if local_cache is not None:
                    # Update the cache with the new instance
                    local_cache[new_information_instance.identifier] = new_information_instance
                # Update the returned map with the new instance
                returned_map[new_information_instance.identifier] = new_information_instance

    return returned_map


class AbstractDataHandler(ABC):
    """
    This is an Abstract class of the Data Handlers, used in the MVC pattern to build a bridge between the GUI and the
    database. This class type interacts with the Table Handlers to retrieve the information from the database.
    """

    def __init__(
            self,
            table_handler: Any | AbstractTableHandler
    ):
        self.table_handler = table_handler

    @staticmethod
    def add_data_to_database(
            are_data_already_present: Callable[[bool], Tuple[bool, AbstractEntity]],
            add_data_to_database_callback: Callable[[None], AbstractEntity],
            local_cache: Optional[dict[int, AbstractEntity]] = None,
    ) -> Any:
        """
        This method is used to add specific data to the database and to the cache.

        :param are_data_already_present:    A lambda function to verify whether the given data
                                            are already in the database or not.
        :param add_data_to_database_callback:   A lambda function to add the given data to the database through a
                                                Table Handler method.
        :param local_cache: The local cache that contains all the information retrieved from the database so far.

        :return:    The instance containing the given data, which is a descendant of the AbstractEntity and is added
                    to the database and cache if not already present.
        """
        # Try to find whether the information is already present in the cache
        # or database
        response, information_instance = are_data_already_present(
            True
        )
        # if the information is present in the database
        if response:
            # Return the information
            return information_instance
        else:
            # Add the information to the database and in the current cache
            information_instance = add_data_to_database_callback(None)
            if local_cache is not None:
                local_cache[information_instance.identifier] = information_instance

            # Return the information
            return information_instance

    @staticmethod
    def get_information_given_ids(
            instance_ids: Union[int, list[int]],
            fetch_information_from_database: Callable[
                [Union[int, list[int]]],
                Union[None, AbstractEntity, dict[int, AbstractEntity]]
            ],
            local_cache: Optional[dict[int, AbstractEntity]] = None,
    ) -> Union[Any, dict[int, Any]]:
        # if the given instance identifier is an integer
        if type(instance_ids) is int:
            instance_id: int = instance_ids
            # if the local cache contains already the corresponding instance
            if local_cache is not None and local_cache.keys().__contains__(instance_id):
                return local_cache[instance_id]
            else:
                # fetch the instance from the database
                return fetch_information_from_database(
                    instance_id
                )
        else:
            # if a list of instances must be retrieved
            returned_device_map: dict[int, AbstractEntity] = {}
            to_be_fetched_instance_ids: Optional[list[int]] = None

            # if some filters are given, then do not fetch all the instances
            if len(instance_ids) > 0:
                # remove the duplicates
                instance_ids = list(set(instance_ids))
                to_be_fetched_instance_ids = []
                # for each instance id to be fetched,
                for instance_id in instance_ids:
                    # if the instance identifier is in the cache
                    if local_cache is not None and local_cache.keys().__contains__(instance_id):
                        # the instance is added to the returned map
                        returned_device_map[instance_id] = local_cache[instance_id]
                    else:
                        # else, the instance identifier is added to the list of the instances to be fetched
                        to_be_fetched_instance_ids.append(instance_id)

            # if there are instances to be fetched or all the instances must be fetched
            if to_be_fetched_instance_ids is None or len(to_be_fetched_instance_ids) > 0:
                if to_be_fetched_instance_ids is None:
                    to_be_fetched_instance_ids = []
                fetched_instances: Union[AbstractEntity, dict[int, AbstractEntity]] = fetch_information_from_database(
                    to_be_fetched_instance_ids
                )

                if fetched_instances is not None:
                    # if the returned type is a dictionary
                    if type(fetched_instances) is dict:
                        # update the returned map
                        returned_device_map.update(fetched_instances)
                    else:
                        # else, the returned type is a single instance
                        # update the returned map
                        returned_device_map[fetched_instances.identifier] = fetched_instances

            if len(returned_device_map) == 0:
                return None
            else:
                return returned_device_map
