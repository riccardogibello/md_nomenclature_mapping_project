from typing import Optional, Any

from psycopg2 import OperationalError

from src.business_logic.database_interfaces.connection_handler import DatabaseInformation
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler
from src.__directory_paths import SQL_CSV_TABLES_DIRECTORY_PATH, SOURCE_DATA_DIRECTORY_PATH

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler, local_database_information
from src.business_logic.database_interfaces.data_handlers.country_data_handler import CountryDataHandler
from src.business_logic.database_interfaces.data_handlers.device_data_handler.clean_device_data_handler import \
    CleanDeviceDataHandler
from src.business_logic.database_interfaces.data_handlers.manufacturer_data_handler.clean_manufacturer_data_handler import \
    CleanManufacturerDataHandler
from src.business_logic.database_interfaces.data_handlers.mappings_data_handlers.device_mapping_data_handler import \
    DeviceMappingDataHandler
from src.business_logic.database_interfaces.data_handlers.mappings_data_handlers.mapping_data_handler import \
    MappingDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.emdn_code_data_handler import \
    EmdnCodeDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.emdn_correspondence_data_handler import \
    EmdnCorrespondenceDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.fda_code_data_handler import \
    FdaCodeDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.gmdn_code_data_handler import \
    GmdnCodeDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.gmdn_correspondence_data_handler import \
    GmdnCorrespondenceDataHandler


def import_tables(
        data_handler_list: Optional[list[Any]] = None,
        database_information: Optional[DatabaseInformation] = local_database_information,
        base_path: Optional[str] = SQL_CSV_TABLES_DIRECTORY_PATH
) -> None:
    """
    This method imports all the given tables from the given base path through the given data handlers.

    :param data_handler_list: The data handlers to be used for importing the tables.
    :param database_information: The database information to be used for the connection.
    :param base_path: The base path from which the tables will be imported.
    """
    if data_handler_list is None:
        data_handler_list = [
            CleanManufacturerDataHandler,
            EmdnCodeDataHandler,
            DeviceMappingDataHandler,
            MappingDataHandler,
            EmdnCorrespondenceDataHandler,
            FdaCodeDataHandler,
            GmdnCodeDataHandler,
            GmdnCorrespondenceDataHandler,
            CleanDeviceDataHandler,
            CountryDataHandler,
        ]

    connection_handler = ConnectionHandler(
        database_information=database_information
    )
    for data_handler_constructor in data_handler_list:
        # Instantiate the table handler
        data_handler = data_handler_constructor(
            connection_handler=connection_handler,
            reset_table=True,
        )
        data_handler.table_handler.import_table(
            base_path=base_path
        )


def export_tables(
        data_handler_list: Optional[list[Any]] = None,
        database_information: Optional[DatabaseInformation] = local_database_information,
) -> None:
    """
    This method exports all the given tables through the given data handlers.

    :param data_handler_list: The data handlers to be used for exporting the tables.
    :param database_information: The database information to be used for the connection.
    """
    if data_handler_list is None:
        data_handler_list = [
            CleanManufacturerDataHandler,
            EmdnCodeDataHandler,
            DeviceMappingDataHandler,
            MappingDataHandler,
            EmdnCorrespondenceDataHandler,
            FdaCodeDataHandler,
            GmdnCodeDataHandler,
            GmdnCorrespondenceDataHandler,
            CleanDeviceDataHandler,
            CountryDataHandler,
        ]

    connection_handler = ConnectionHandler(
        database_information=database_information
    )
    for data_handler_constructor in data_handler_list:
        # Instantiate the table handler
        data_handler = data_handler_constructor(
            connection_handler=connection_handler
        )
        data_handler.table_handler.export_table()


def execute_file(
        file_path: str,
        additional_sql_commands: Optional[list[str]] = None,
        database_information: Optional[DatabaseInformation] = local_database_information
) -> None:
    """
    This method executes all the SQL commands from the given file path.

    :param file_path: The path of the file containing the SQL commands.
    :param additional_sql_commands: Additional SQL commands to be executed after the ones from the file.
    :param database_information: The database information to be used for the connection.
    """
    connection_handler = ConnectionHandler(
        database_information=database_information
    )

    # Open the SQL file
    # Open and read the file as a single buffer
    fd = open(file_path, 'r')
    sql_file = fd.read()
    fd.close()

    # Remove any row which contains a -- comment
    sql_commands: list[str] = []
    tmp_sql_command = ''
    is_end_of_command = False
    for row in sql_file.split('\n'):
        # If the previous row was the end of a command, reset the tmp_sql_command
        if is_end_of_command:
            tmp_sql_command = ''
            is_end_of_command = False
        # If the row contains a -- comment, pass it
        if '--' in row or '/*' in row or '*/' in row:
            continue
        else:
            # Update the tmp_sql_command with the row
            tmp_sql_command += row + ' '
            # If the row contains a ";", it is the end of the command
            if ';' in row:
                sql_commands.append(tmp_sql_command)
                is_end_of_command = True

    # If the sql_commands list is not empty, execute the commands too
    if additional_sql_commands is not None:
        sql_commands.extend(additional_sql_commands)

    # Execute every command from the input file
    for command in sql_commands:
        # This will skip and report errors
        # For example, if the tables do not yet exist, this will skip over
        # the DROP TABLE commands
        try:
            connection_handler.cursor.execute(command)
        except OperationalError:
            pass
    connection_handler.connection.commit()


def build_and_export_table(
        table_name: str,
        columns_order: list[str],
        executable_name: str,
        database_information: Optional[DatabaseInformation] = local_database_information,
        drop_table: bool = False,
) -> None:
    """
    This method executes the SQL commands from the input file and exports the built table.

    :param table_name: The name of the table to be exported.
    :param columns_order: The order of the columns in the table.
    :param executable_name: The name of the file containing the SQL commands.
    :param database_information: The database information to be used for the connection.
    :param drop_table: Whether to drop the table before exporting it.
    """
    # Execute all the SQL commands from the input file
    execute_file(
        SOURCE_DATA_DIRECTORY_PATH + executable_name,
    )
    # Export the built table
    AbstractTableHandler.export_table_static(
        base_path=SQL_CSV_TABLES_DIRECTORY_PATH,
        table_name=table_name,
        columns=columns_order,
        connection_handler=ConnectionHandler(
            database_information=database_information
        ),
        drop_table=drop_table
    )
