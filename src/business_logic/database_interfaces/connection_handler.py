from datetime import datetime
from typing import Tuple, Union, Optional, Any, List

import numpy as np
import psycopg2
from psycopg2 import extras

from src.business_logic.database_interfaces.__table_constants import TableColumnName


class DatabaseInformation:

    def __init__(
            self,
            user: str,
            password: str,
            host: Optional[str] = 'localhost',
            port: Optional[int] = 3306,
            database_name: Optional[str] = '',
    ):
        """
        Sets the proper values for the database credentials and information.

        :param user: The user that must be supplied to the database in order to log in.
        :param password: The password that must be supplied to the database in order to log in.
        :param host: The host containing the database to which the connection must be established.
        :param port: The port used to connect to the database.
        :param database_name: The name of the database to which the CursorHandler must be connected to.
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database_name = database_name


local_database_information = DatabaseInformation(
    database_name='nomenclature_mapping_project',
    user='test',
    password='test',
    host='localhost',
    port=5432,
)


class ConnectionHandler:
    """
    This is the custom implementation of the cursor dealing with the specific SQLite technology. If the underlying
    database technology is changed, then this is the only class to be modified.
    """

    def __init__(
            self,
            database_information: DatabaseInformation
    ):
        """
        This method creates an instance of CursorHandler to be used to manage the connection to the database.

        :param database_information: An instance that contains all the information to connect to the database.
        """
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.cursor: Optional[psycopg2.extensions.cursor] = None

        self.last_access_to_cursor: Optional[datetime] = None
        self.database_information: Optional[DatabaseInformation] = None

        # If the database connection is not yet established or the current process/thread is different from the one
        # that created the connection
        if self.connection is None:
            # Connect to the database machine
            try:
                # =======================================================================
                # Required initialization pipeline of the database:

                # CREATE USER [user_name] WITH PASSWORD '[password]';
                # CREATE database [database_name];
                # GRANT ALL PRIVILEGES ON DATABASE [database_name] TO [user_name];
                # \c nomenclature_mapping_project;
                # CREATE EXTENSION vector;
                # GRANT ALL ON SCHEMA public TO gb11;
                # =======================================================================
                connection = psycopg2.connect(
                    host=database_information.host,
                    port=database_information.port,
                    user=database_information.user,
                    password=database_information.password,
                    dbname=database_information.database_name,
                )
                self.connection = connection
            # If the db does not exist yet
            except psycopg2.OperationalError:
                raise ValueError('The database does not exist yet. Please create it first.')

        # Store the database information
        self.database_information = database_information
        # Initialize the cursor
        self.cursor = self.connection.cursor()
        self.last_access_to_cursor = datetime.now()

    def _get_cursor(self):
        # If the last time that the thread accessed to the database was more than 5 minutes ago
        is_time_expired = (
                self.last_access_to_cursor is None or
                ((datetime.now() - self.last_access_to_cursor).total_seconds() > 300)
        )

        if is_time_expired:
            self.__init__(self.database_information)

        # Return the cursor created by the current connection
        return self.cursor

    def add_batched(
            self,
            column_names: list[str],
            table_name: str,
            values: list[list],
            id_column_name: Optional[str] = TableColumnName.IDENTIFIER.value,
    ) -> List[tuple]:
        """
        This method adds a batch of rows to a specific table and returns the primary keys of the inserted rows.

        :param column_names: The column names of the table to which the rows must be added.
        :param table_name: The name of the table to which the rows must be added.
        :param values: The values of the rows to be added.
        :param id_column_name: The name of the column that contains the primary keys of the table.

        :return: The identifiers of each of the added rows.
        """
        # Get a new cursor
        cursor = self._get_cursor()

        # Create the SQL command that must be executed
        sql_string = f"""
            INSERT INTO {table_name}({','.join(column_names)}) 
            VALUES %s
            RETURNING {id_column_name}
        """
        # Execute the SQL command
        ids = extras.execute_values(
            cursor,
            sql_string,
            values,
            page_size=10000,
            fetch=True,
        )
        # Commit the statement
        self.connection.commit()

        return ids

    def execute(
            self,
            sql_string: str,
            parameters: Optional[Tuple] = (),
            commit: Optional[bool] = True,
    ) -> None:
        """
        This method executes the given SQL command with the given parameters. It commits the action, if not specified
        differently.

        :param sql_string: The SQL command to be executed.
        :param parameters: The named parameters to be added to the sql_string.
        :param commit: Indicates whether the SQL command must be committed or not.
        """
        # sanitize parameters so that to remove nan values and replace them with None
        cleaned_parameters = []
        for parameter in parameters:
            if parameter is np.nan:
                cleaned_parameters.append(None)
            else:
                cleaned_parameters.append(parameter)

        # Get a new cursor
        cursor = self._get_cursor()
        # Execute the SQL command
        cursor.execute(
            sql_string,
            cleaned_parameters,
        )

        # If the current execution of the statement requires that the
        # transaction ends
        if commit:
            # Commit the SQL statement and end the transaction
            self.connection.commit()

    def fetch(
            self,
            sql_string: str,
            parameters: Optional[Tuple] = None,
            fetch_only_one: Optional[bool] = True,
    ) -> Union[list, list[list]]:
        """
        This method fetches from a specific table a row / a set of rows.

        :param sql_string: The SQL command to be executed.
        :param parameters: The named parameters to be added to the sql_string.
        :param fetch_only_one: Indicates whether the command must fetch only one row, or multiple ones.

        :return: A list or rows, if fetch_only_one is False. A single row, otherwise.
        """
        # if either the given parameters are None or empty
        if parameters is None or len(parameters) == 0:
            parameters = ()
        else:
            # otherwise,
            all_empty = True
            index = 0
            # for each parameter
            for parameter in parameters:
                # if the parameter is None
                if parameter is None or parameter == '' or parameter == np.nan:
                    if parameter is None:
                        # then replace it with an empty string
                        parameters = parameters[:index] + ('',) + parameters[index + 1:]
                else:
                    # otherwise, at least one parameter is valid
                    all_empty = False
                index += 1
            # if all the given parameters are None or '' or np.nan
            if all_empty:
                # then set the parameters to be empty
                parameters = ()

        # if the method must fetch only one row
        if fetch_only_one:
            self.cursor.execute(
                sql_string,
                parameters
            )
            returned_value = self.cursor.fetchone()
        else:
            self.cursor.execute(
                sql_string,
                parameters
            )
            returned_value = self.cursor.fetchall()

        # Commit the SQL statement and end the transaction
        self.connection.commit()

        # If the fetch query did not produce any results
        if returned_value is None:
            return []
        else:
            return list(returned_value)

    def import_table(
            self,
            file_path: str,
            table_name: str,
    ) -> None:
        # Open the CSV file for reading
        with open(file_path, 'r') as csvfile:
            # Read the first line, that is the header of the table
            header = csvfile.readline()
            header = [header_column.replace('\n', '') for header_column in header.split(',')]

            csvfile: Any = csvfile
            self.cursor.copy_expert(
                sql=f"COPY {table_name}({','.join(header)}) FROM STDIN WITH (FORMAT CSV, HEADER)",
                file=csvfile,
            )

        self.connection.commit()

    def retrieve_last_inserted_row(
            self,
    ) -> list[list[Any]]:
        """
        This method retrieves the last inserted table identifiers, given that the SQL query included
        the "RETURNING" statement.

        :return: The last inserted IDs.
        """
        fetched_pks: Any = self.cursor.fetchall()
        self.connection.commit()

        return fetched_pks
