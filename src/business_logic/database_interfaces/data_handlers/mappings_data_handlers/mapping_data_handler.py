import multiprocessing
import threading
from typing import Optional, Any

import numpy as np

from src.business_logic.database_interfaces.data_handlers.abstract_data_handler import AbstractDataHandler
from src.data_model.mapping.nomenclature_mapping import NomenclatureMapping
from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.mappings_data_handlers.__key_creation import \
    concatenate_emdn_gmdn_identifiers
from src.business_logic.database_interfaces.table_handlers.mappings_table_handlers.mapping_table_handler import \
    MappingTableHandler
from src.business_logic.utilities.dictionary_utilities import safe_add_to_dictionary, safe_remove_from_dictionary
from src.data_model.enums import TranslationDirection
from src.data_model.nomenclature_codes.emdn_code import EmdnCode


class MappingDataHandler(AbstractDataHandler):
    """
    This is the handler to manage the information related to mappings between GMDN and EMDN codes.
    """

    # __emdn_code__similarity_mappings = a dictionary in which every key is an EMDN src, to which a list of
    #                                    mappings is associated.
    __emdn_id__similarity_mappings: dict[int, list[NomenclatureMapping]]

    # __gmdn_code__similarity_mappings = a dictionary in which every key is a GMDN src, to which a list of
    #                                    mappings is associated.
    __gmdn_id__similarity_mappings: dict[int, list[NomenclatureMapping]]

    # __emdn_gmdn__correspondence__map = a map in which the key is the composition of EMDN-GMDN and the value is an
    #                                    instance of mapping.
    __emdn_gmdn__correspondence__map: dict[str, NomenclatureMapping]

    # mapping_table_handler = a table handler in order to use proper methods to update a given table.
    mapping_table_handler: MappingTableHandler

    mapping_table_thread_lock: threading.Lock = threading.Lock()
    mapping_table_process_lock: multiprocessing.Lock = multiprocessing.Lock()

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
            init_cache: Optional[bool] = False,
    ):
        super().__init__(
            table_handler=MappingTableHandler(
                connection_handler=connection_handler,
                reset_table=reset_table,
            )
        )
        # Initialize the table of GMDN-EMDN mappings
        self.table_handler: MappingTableHandler = self.table_handler

        # initialize the caches to keep the mappings
        self.__emdn_id__similarity_mappings: dict[int, list[NomenclatureMapping]] = {}
        self.__gmdn_id__similarity_mappings: dict[int, list[NomenclatureMapping]] = {}
        self.__emdn_gmdn__correspondence__map: dict[str, NomenclatureMapping] = {}

        self.mapping_table_handler = MappingTableHandler(
            connection_handler=connection_handler,
        )

        # If the cache must be initialized
        if init_cache:
            # Force the update of the cache by fetching all the mappings from the database
            self.get_mappings(
                translation_direction=TranslationDirection.FROM_GMDN_TO_EMDN,
                force_refresh=True
            )

    def get_total_mappings_number(
            self,
            enforce_cache_update: Optional[bool] = False,
    ) -> int:
        """
        This method returns the total number of mappings currently present in the cache.

        :param enforce_cache_update: Indicates whether the updates from the database is needed before computing the
                                     searched number.

        :return: The total number of translations currently present in the cache.
        """
        raise NotImplementedError("This method is not implemented yet.")

    def get_mappings(
            self,
            translation_direction: TranslationDirection,
            force_refresh: Optional[bool] = False,
    ) -> dict[int, list[NomenclatureMapping]]:
        """
        This method returns a shallow copy the cache which keeps the mappings between GMDN-EMDN, in the specified
        direction of indexing by translation_direction. If enforce_cache_update is set to True, an update
        of the cache is performed before giving the results back.

        :param translation_direction: Indicates the right cache map to be returned.
        :param force_refresh:   Indicates whether the cache must be updated from the database before returning
                                the results.

        :return: A shallow copy the cache which keeps the mappings between GMDN-EMDN, in the specified
        direction of indexing by translation_direction.
        """
        # If a query must be performed on the database to update the cache
        if force_refresh:
            # Fetch all the mappings from the database
            fetched_mapping_list = self.mapping_table_handler.fetch_mappings()
            # Update the cache, if necessary
            for fetched_mapping in fetched_mapping_list:
                self._add_mapping_into_cache(
                    fetched_mapping
                )

        # If the key must be the GMDN identifier
        if translation_direction == TranslationDirection.FROM_GMDN_TO_EMDN:
            mapping_couples: dict[int, list[NomenclatureMapping]] = self.__gmdn_id__similarity_mappings.copy()
        # If the key must be the EMDN identifier
        else:
            mapping_couples: dict[int, list[NomenclatureMapping]] = self.__emdn_id__similarity_mappings.copy()

        return mapping_couples

    def fetch_correspondences_given_gmdn(
            self,
            gmdn_id: int,
    ) -> list[NomenclatureMapping]:
        """
        This method searches either in the whitelisted and in the mapping codes for the given GMDN code.

        :param gmdn_id: The identifier of the GMDN code for which the related EMDN codes must be found.

        :return: A list of EMDN identifiers that are related to the given GMDN code.
        """
        # Otherwise, search for a mapping for the given GMDN identifier in the cache of mappings
        if self.__gmdn_id__similarity_mappings.keys().__contains__(gmdn_id):
            # Get all the EMDN identifiers
            mappings = [
                mapping
                for mapping in self.__gmdn_id__similarity_mappings[gmdn_id]
            ]

            return mappings
        else:
            return []

    def fetch_correspondences_given_emdn(
            self,
            emdn_code_entity: EmdnCode,
    ) -> list[NomenclatureMapping]:
        """
        This method returns a list of mappings, containing the EMDN src corresponding to
        emdn_code_entity, a related GMDN src and other metadata.

        :param emdn_code_entity: The EMDN src entity for which a GMDN translation must be found.

        :return: A list of mappings, containing the EMDN src corresponding to emdn_code_entity, a related GMDN
                 src and other metadata.
                 If no correspondence is found, then an empty list is returned.
        """
        # Enforce a fetch from the database of all the EMDN-GMDN mappings
        self.get_mappings(
            translation_direction=TranslationDirection.FROM_EMDN_TO_GMDN
        )

        # If the cache contains the identifier of the searched EMDN code
        if self.__emdn_id__similarity_mappings.keys().__contains__(emdn_code_entity.identifier):
            # Return all the GMDN identifiers that are not blacklisted and that are related to the given EMDN code
            return [
                mapping
                for mapping in self.__emdn_id__similarity_mappings[emdn_code_entity.identifier]
            ]
        else:
            return []

    def is_mapping_already_cached(
            self,
            gmdn_id: int,
            emdn_id: int,
    ) -> bool:
        """
        This method verifies whether the given GMDN-EMDN mapping is already locally stored.

        :param gmdn_id: The GMDN identifier involved in the correspondence.
        :param emdn_id: The EMDN identifier involved in the correspondence.

        :return: True, if the mapping is already present. False, otherwise.
        """
        key: str = concatenate_emdn_gmdn_identifiers(
            emdn_id=emdn_id,
            gmdn_id=gmdn_id,
        )

        # if the mapping is present in the cache
        if self.__emdn_gmdn__correspondence__map.keys().__contains__(key):
            return True
        else:
            # otherwise,
            return False

    def fetch_mapping(
            self,
            emdn_id: int,
            gmdn_id: int,
            fetch_from_database: Optional[str] = False,
    ) -> Optional[NomenclatureMapping]:
        """
        This method fetches a given mapping's data from the local cache.

        If the data are not present in the cache, it looks up in the database and returns a proper value, if any.

        If the data are in cache, but fetch_from_database is set to True, a call to the database is performed.
        If the data are still present, they are updated in the cache. If they are not, they are removed in the cache.

        :param emdn_id: The EMDN src string involved in whitelisting.
        :param gmdn_id: The GMDN src string involved in whitelisting.
        :param fetch_from_database: Indicates whether the local mapping instance from the cache must be updated
                                    by doing a query on the database.

        :return: A mapping instance, if it is present at least in the database. None, if it is non-existent or if it
                 has been deleted.
        """
        key: str = concatenate_emdn_gmdn_identifiers(
            emdn_id=emdn_id,
            gmdn_id=gmdn_id
        )

        # if the given mapping is already in cache
        if self.__emdn_gmdn__correspondence__map.keys().__contains__(key):
            # get the cached instance of the mapping
            cached_mapping = self.__emdn_gmdn__correspondence__map[key]
            # if the database must be queried in order to update the local cache
            if fetch_from_database:
                # get the mapping from the database
                database_mapping = self.mapping_table_handler.get_mapping(
                    emdn_id=emdn_id,
                    gmdn_id=gmdn_id
                )
                # if the mapping has been deleted from the database
                if database_mapping is None:
                    # remove the mapping from the cache
                    self.remove_mapping(
                        emdn_id=emdn_id,
                        gmdn_id=gmdn_id,
                    )
                    return None
                else:
                    # update the values in cache
                    cached_mapping.copy_from(database_mapping)

            return cached_mapping
        else:
            # try to retrieve the mapping from the database
            database_mapping = self.mapping_table_handler.get_mapping(
                emdn_id=emdn_id,
                gmdn_id=gmdn_id
            )
            # if the mapping is not present
            if database_mapping is None:
                return None
            else:
                # otherwise, add it to the cache
                self._add_mapping_into_cache(database_mapping)
                return database_mapping

    def _add_mapping_into_cache(
            self,
            mapping: NomenclatureMapping,
    ) -> None:
        """
        This method, given a proper mapping, adds it to the local caches.

        :param mapping: The mapping to be added to the local caches.
        """
        emdn_id: int = mapping.emdn_id
        gmdn_id: int = mapping.gmdn_id
        key = concatenate_emdn_gmdn_identifiers(emdn_id, gmdn_id)

        # add the mapping to the three different dictionaries
        safe_add_to_dictionary(
            _dict=self.__emdn_id__similarity_mappings,
            key=emdn_id,
            value_to_add=mapping
        )
        safe_add_to_dictionary(
            _dict=self.__gmdn_id__similarity_mappings,
            key=gmdn_id,
            value_to_add=mapping
        )
        safe_add_to_dictionary(
            _dict=self.__emdn_gmdn__correspondence__map,
            key=key,
            value_to_add=mapping,
            is_value_list=False,
        )

    def remove_mapping(
            self,
            emdn_id: int,
            gmdn_id: int,
    ) -> None:
        """
        This method is used to clean all the data structures when removing a particular GMDN-EMDN mapping.
        It removes every data structure present in the caches (if any) and calls the proper method on
        the mapping_table_handler to remove the mapping from the database too.

        :param emdn_id: The EMDN src string involved in the mapping.
        :param gmdn_id: The GMDN src string involved in the mapping.
        """
        # Acquire the locks over the mapping table
        with self.mapping_table_thread_lock:
            with self.mapping_table_process_lock:
                # get the combined key used to index the following cache
                key = concatenate_emdn_gmdn_identifiers(emdn_id, gmdn_id)
                # remove the couple from the local caches (if any)
                removed_mapping: NomenclatureMapping = safe_remove_from_dictionary(
                    _dict=self.__emdn_gmdn__correspondence__map,
                    key=key,
                    value_to_remove=None,
                    is_value_list=False,
                )

                # if the given mapping was found in the first cache and removed,
                if removed_mapping is not None:
                    # remove it from the other caches too
                    safe_remove_from_dictionary(
                        _dict=self.__emdn_id__similarity_mappings,
                        key=emdn_id,
                        value_to_remove=removed_mapping,
                    )
                    safe_remove_from_dictionary(
                        _dict=self.__gmdn_id__similarity_mappings,
                        key=gmdn_id,
                        value_to_remove=removed_mapping,
                    )

                # remove the entry from the database too (if any)
                self.mapping_table_handler.remove_mapping_from_database(
                    gmdn_id=gmdn_id,
                    emdn_id=emdn_id
                )

    def create_mapping(
            self,
            emdn_id: int,
            gmdn_id: int,
    ) -> Optional[NomenclatureMapping]:
        """
        Given a GMDN, an EMDN src and a proper is_from_exact_match flag, this method adds/updates
        the mapping in the database. Then, the same addition/update is reflected in the cache.

        This can be performed only if the mapping is not blacklisted.

        :param emdn_id: The EMDN alphanumerical src.
        :param gmdn_id: The GMDN src given by the application and unique for the whole GMDN vocabulary.

        :return: True if the mapping with the given data, at the end of the method, is present both in the database
                 and in the caches.

                 False, if it was not possible to add/update it (due to blacklisting or because the is_exact_match had
                 to be set to False, but in the database it presented a True value).
        """
        with self.mapping_table_thread_lock:
            with self.mapping_table_process_lock:
                # check whether the mapping is already cached in the local data structures
                created_mapping = self.fetch_mapping(
                    gmdn_id=gmdn_id,
                    emdn_id=emdn_id,
                )

                # if the mapping is non-existent both in cache and database
                if created_mapping is None:
                    # then, create it
                    created_mapping = self.mapping_table_handler.add_mapping(
                        emdn_gmdn_ids=[(emdn_id, gmdn_id)]
                    )
                    # then, add it also locally
                    self._add_mapping_into_cache(
                        mapping=created_mapping,
                    )

        return created_mapping

    def add_batched_mappings(
            self,
            mappings_batch: np.ndarray[Any, np.dtype],
    ) -> Optional[dict[str, NomenclatureMapping]]:
        """
        This method, given a list of tuples, adds the data in the mapping table by calling the create_mapping method.

        :param mappings_batch: A list of tuples, where every tuple contains a GMDN and EMDN code identifier.
        """
        new_mapping_list = {}
        for gmdn_emdn in mappings_batch:
            emdn_code = gmdn_emdn[0]
            gmdn_code = gmdn_emdn[1]

            new_mapping = self.create_mapping(
                emdn_id=emdn_code,
                gmdn_id=gmdn_code,
            )
            if new_mapping is not None:
                key: str = str(emdn_code) + "$" + str(gmdn_code)
                new_mapping_list[key] = new_mapping

        if len(new_mapping_list) == 0:
            return None
        else:
            return new_mapping_list
