from typing import Optional, Tuple, Union, List

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.abstract_data_handler import AbstractDataHandler, \
    get_instances
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import SqlQueryComponent
from src.business_logic.database_interfaces.table_handlers.device_table_handler.clean_device_table_handler import \
    CleanDeviceTableHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.data_model.devices.clean_device import CleanDevice


class CleanDeviceDataHandler(AbstractDataHandler):

    def __init__(
            self,
            connection_handler: ConnectionHandler,
            init_cache: Optional[bool] = False,
            reset_table: Optional[bool] = False,
    ):
        super().__init__(
            table_handler=CleanDeviceTableHandler(
                connection_handler=connection_handler,
                reset_table=reset_table,
            ),
        )

        # device_id__device_instance maps a device id to a CleanDevice instance.
        self.table_handler: CleanDeviceTableHandler = self.table_handler
        self._device_id__device_instance: dict[int, CleanDevice] = {}

        # clean_device_handler is used to interact with the clean_device table.
        self._device_id__device_instance: dict[int, CleanDevice]

        if init_cache:
            self.get_devices_by_ids()

    def get_by_manufacturer_id(
            self,
            manufacturer_id: int,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> dict[int, CleanDevice]:
        """
        This method fetches all the devices that are produced by a specific manufacturer.

        :param manufacturer_id: The identifier of the manufacturer.
        :param perform_fetch_on_database:   If set to True, then the method will fetch the data from the database
                                            if not found.

        :return: A list of devices that are produced by the manufacturer.
        """
        # If the data should be fetched from the database if not found
        if perform_fetch_on_database:
            # Call anyway this method, without the cache because it is indexed by the device id
            returned_value = get_instances(
                fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                    filters=_filters,
                    instance_builder=lambda values: CleanDevice(
                        *values
                    ),
                ),
                sql_filters=[
                    SqlQueryComponent(
                        column_name=TableColumnName.CLEAN_MANUFACTURER_ID.value,
                        column_values=[manufacturer_id],
                    ),
                ],
                local_cache=self._device_id__device_instance,
                perform_fetch_on_database=perform_fetch_on_database,
            )

            return returned_value
        else:
            # Iterate over the cached devices and check whether the manufacturer id is the same
            return {
                device_id: device_instance
                for device_id, device_instance in self._device_id__device_instance.items()
                if device_instance.manufacturer_id == manufacturer_id
            }

    def get_devices_by_ids(
            self,
            device_ids: Optional[Union[list[int], int]] = None,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> Union[None, CleanDevice, dict[int, CleanDevice]]:
        """
        This method returns a dictionary that maps a device id to a CleanDevice instance.

        :param device_ids: The identifiers of the devices to be fetched.
        :param perform_fetch_on_database:   If set to True, then the method will fetch the data from the database
                                            if not found.

        :return: A dictionary that maps a device id to a CleanDevice instance.
        """
        filters: Optional[List[SqlQueryComponent]] = None
        if device_ids is not None:
            if isinstance(device_ids, int):
                device_ids = [device_ids]
            filters = [
                SqlQueryComponent(
                    column_name=TableColumnName.IDENTIFIER.value,
                    column_values=device_ids,
                ),
            ]

        return get_instances(
            fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                filters=_filters,
                instance_builder=lambda values: CleanDevice(
                    *values
                ),
            ),
            sql_filters=filters,
            local_cache=self._device_id__device_instance,
            perform_fetch_on_database=perform_fetch_on_database,
        )

    def get_device(
            self,
            original_device_name: str,
            device_model: str,
            clean_manufacturer_id: int,
            perform_fetch_on_database: Optional[bool] = True,
    ) -> Tuple[bool, Optional[CleanDevice]]:
        """
        This method checks if a device is already existing in the database.
        The key used to check if the device is (original_device_name, clean_manufacturer_id).
        The method firstly tries in the cache, then in the database (if fetch_from_db is set to True).

        :param original_device_name: The original name of the device.
        :param device_model: The model of the device.
        :param clean_manufacturer_id: The identifier of the manufacturer of the device in the clean_manufacturer table.
        :param perform_fetch_on_database:   If set to True, then the method will fetch the data from the database
                                            if not found.

        :return:    If the device is already existing, then the method returns True and the device instance.
                    If the device is not existing, then the method returns False and None.
        """
        returned_value = get_instances(
            fetch_information_from_database=lambda _filters: self.table_handler.fetch(
                filters=_filters,
                instance_builder=lambda values: CleanDevice(
                    *values
                ),
            ),
            sql_filters=[
                SqlQueryComponent(
                    column_name=TableColumnName.NAME.value,
                    column_values=[original_device_name],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.CATALOGUE_CODE.value,
                    column_values=[device_model],
                ),
                SqlQueryComponent(
                    column_name=TableColumnName.CLEAN_MANUFACTURER_ID.value,
                    column_values=[clean_manufacturer_id],
                ),
            ],
            local_cache=self._device_id__device_instance,
            perform_fetch_on_database=perform_fetch_on_database,
        )

        if returned_value is not None and len(returned_value) > 0:
            return True, list(returned_value.values())[0]
        else:
            return False, None

    def add_clean_device(
            self,
            original_device_name: str,
            catalogue_code: str,
            clean_manufacturer_id: int,
            clean_name: str,
            device_type: str,
            ppc: Optional[str] = None,
            high_risk_device_type: Optional[str] = None,
            standardized_device_id: Optional[int] = None,
    ) -> CleanDevice:
        """
        This method adds a device to the database.

        :param original_device_name: The original name of the device.
        :param catalogue_code: The model of the device.
        :param clean_manufacturer_id: The identifier of the manufacturer of the device in the clean_manufacturer table.
        :param clean_name: The clean name of the device.
        :param device_type: The type of the device (MD, IVD...).
        :param ppc: The FDA product code classification of the medical device.
        :param high_risk_device_type: The high risk device type.
        :param standardized_device_id: The identifier of the standardized device in the standardized_device table.

        :return: The medical device instance containing the information of the device.
        """
        return self.table_handler.add_clean_device(
            name=original_device_name,
            clean_manufacturer_id=clean_manufacturer_id,
            catalogue_code=catalogue_code,
            clean_name=clean_name,
            device_type=device_type,
            high_risk_device_type=high_risk_device_type,
            product_code=ppc,
            standardized_device_id=standardized_device_id,
        )
