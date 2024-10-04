import csv
import os
from abc import ABC
from enum import Enum
from typing import Optional, Callable, Any, Union, Tuple, List

import psycopg2

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.__table_constants import TableColumnName
from src.business_logic.utilities.os_utilities import create_directory
from src.__directory_paths import SQL_CSV_TABLES_DIRECTORY_PATH


class SqlOperator(Enum):
    EQUAL = '='
    GREATER_THAN = '>'
    GREATER_THAN_OR_EQUAL = '>='
    LESS_THAN = '<'
    LESS_THAN_OR_EQUAL = '<='
    NOT_EQUAL = '!='
    LIKE = 'LIKE'
    IN = 'IN'
    NOT_IN = 'NOT IN'
    IS_NULL = 'IS NULL'
    IS_NOT_NULL = 'IS NOT NULL'
    BETWEEN = 'BETWEEN'
    NOT_BETWEEN = 'NOT BETWEEN'
    AND = 'AND'
    OR = 'OR'


class SqlQueryComponent:
    column_name: str
    column_values: list[Any]
    next_operator: SqlOperator
    must_be_comprised: bool

    def __init__(
            self,
            column_name: str,
            column_values: list[Any],
            must_be_comprised: Optional[bool] = True,
            next_operator: Optional[SqlOperator] = SqlOperator.AND,
    ):
        self.column_name = column_name
        self.column_values = column_values
        self.next_operator = next_operator
        self.must_be_comprised = must_be_comprised

    def to_sql_string(
            self,
    ) -> Tuple[str, list[Any]]:
        """
        This method creates an SQL query that, for the given column name, checks if the column value is in the given
        list of values. It also returns the list of values to be set into the query.

        :return:    This method returns an SQL query that, for the given column name, checks if the column value is in
                    the given list of values. It also returns the list of values to be set into the query.
        """
        # create the prefix of the SQL query
        sql_string = ""
        # create the list of values to be set into the query
        parameters = []
        # if there is only one value to filter the column
        if len(self.column_values) == 1 and self.must_be_comprised:
            # set the equality sql string
            sql_string = f"""
                {self.column_name} = %s
            """
            # add the value to the list of values
            parameters.append(self.column_values[0])
        # if there are more than one value to filter the column
        elif len(self.column_values) > 1 or not self.must_be_comprised:
            if self.must_be_comprised:
                # set the IN sql string
                sql_string = f"""
                    {self.column_name} IN (
                """
            else:
                # set the NOT IN sql string
                sql_string = f"""
                    {self.column_name} NOT IN (
                """
            # for each index of the list of values
            for column_value_index in range(len(self.column_values) - 1):
                # add the placeholder in the string
                sql_string += f"%s, "
                # add the value to the list of values
                parameters.append(self.column_values[column_value_index])
            # add the last placeholder in the string
            sql_string += f"%s)"
            # add the last value to the list of values
            parameters.append(self.column_values[-1])

        return sql_string, parameters


class SqlQuery:
    sql_query_components: list[SqlQueryComponent]

    def __init__(
            self,
            sql_query_components: list[SqlQueryComponent],
    ):
        self.sql_query_components = sql_query_components

    def to_sql_string(
            self,
    ) -> Tuple[str, list[Any]]:
        """
        This method builds the SQL string that represents the query and creates a list of all the parameters to be
        set into it.

        :return:    This method returns the SQL string that represents the query and creates a list of all the
                    parameters to be set into it.
        """
        parameters = []
        # get the sql string and parameters for the first filter
        previous_query_component = self.sql_query_components[0]
        sql_string, filter_parameters = previous_query_component.to_sql_string()
        sql_string = f"""
            WHERE {sql_string}
        """
        # add the given filter values to the list of parameters
        parameters.extend(filter_parameters)
        # for each extra given filter, add an AND clause to the SQL string
        for sql_query_component in self.sql_query_components[1:]:
            # get the sql string and parameters for the current filter
            tmp_sql_string, tmp_filter_parameters = sql_query_component.to_sql_string()
            sql_string += f"""
                {previous_query_component.next_operator.value} {tmp_sql_string}
            """
            # add the given filter values to the list of parameters
            parameters.extend(tmp_filter_parameters)

            # update the value of the previous filter to the current one
            previous_query_component = sql_query_component

        return sql_string, parameters


class AbstractTableHandler(ABC):
    """
    This is an abstract class that is used to manage information stored in SQLite tables.
    """

    def __init__(
            self,
            table_name: str,
            connection_handler: ConnectionHandler,
            reset_table: Optional[bool] = False,
            initializing_table_statement: Callable[[str], str] = None,
    ):
        """
        Sets the current class' fields.
        """
        # Instantiate the field containing the database table name
        self.table_name = table_name

        # Initialize the cursor handler to interact with the database
        self.connection_handler = connection_handler

        # If the table must be reset, and it exits, drop it
        if reset_table:
            self._reset_table()
        # If the statement to initialize the table has been given
        if initializing_table_statement is not None:
            try:
                # Create the table if it does not exist
                self.connection_handler.execute(
                    sql_string=initializing_table_statement(self.table_name),
                )
            except psycopg2.errors.UniqueViolation:
                # If the insert of the table has been already done
                # by another thread/process, then rollback the current transaction
                self.connection_handler.connection.rollback()

    def _reset_table(
            self
    ) -> None:
        """
        This method resets the table by dropping it.
        """
        # Drop the table
        self.connection_handler.execute(
            f"DROP TABLE IF EXISTS {self.table_name};"
        )
        # Drop the sequence that creates the identifiers
        self.connection_handler.execute(
            f"DROP SEQUENCE IF EXISTS {self.table_name}_id_seq;"
        )

    def get_table_column_names(
            self
    ) -> list[str]:
        """
        This method returns the column names of the table.

        :return:    This method returns the column names of the table.
        """
        result = self.connection_handler.fetch(
            sql_string=f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{self.table_name}';
            """,
            fetch_only_one=False,
        )

        return [
            column_name[0]
            for column_name in result
        ]

    @staticmethod
    def export_table_static(
            table_name: str,
            columns: List[str],
            base_path: Optional[str] = SQL_CSV_TABLES_DIRECTORY_PATH,
            connection_handler: Optional[ConnectionHandler] = None,
            rows: Optional[List[List[Any]]] = None,
            drop_table: Optional[bool] = False,
    ):
        # Create the output directory if it does not exist
        create_directory(base_path)

        if rows is None and connection_handler is None:
            raise ValueError("Either the connection handler or the rows must be given")
        if rows is None:
            if columns is not None:
                # Fetch the rows from the table
                select_string: str = ', '.join(columns)
            else:
                select_string = '*'

            rows = connection_handler.fetch(
                sql_string=f"""
                    SELECT {select_string}
                    FROM {table_name}
                """,
                fetch_only_one=False,
            )

        # Create the name of the table file as the name of the table
        filename = table_name + ".csv"
        file_path = os.path.join(base_path, filename)

        # Open the output file and write the rows to it
        with open(file_path, 'w', newline='', encoding='utf-8', errors='ignore') as csvfile:
            # Create a CSV writer object
            writer = csv.writer(csvfile)
            # Write the column names to the file
            writer.writerow(columns)
            # For each data row
            for row in rows:
                # Write the row to the file
                writer.writerow(row)

        if drop_table:
            connection_handler.execute(
                f"DROP TABLE IF EXISTS {table_name};"
            )
            connection_handler.execute(
                f"DROP SEQUENCE IF EXISTS {table_name}_id_seq;"
            )

    def export_table(
            self,
            table_name: Optional[str] = None,
    ) -> None:
        """
        This method exports the table to a CSV file.

        :param table_name: The name of the table to export.
        """
        # If the table name is not given, use the current table name
        if table_name is None:
            table_name = self.table_name

        # Fetch the names of the table columns
        column_names = self.get_table_column_names()
        # Perform a complete fetch of the rows in the table handled
        # by the handler
        rows = self.fetch(
            columns_order=column_names
        )
        if rows is None:
            rows = []

        AbstractTableHandler.export_table_static(
            table_name=table_name,
            rows=rows,
            columns=column_names,
        )

    def import_table(
            self,
            table_name: Optional[str] = None,
            table_file_name: Optional[str] = None,
            base_path: Optional[str] = SQL_CSV_TABLES_DIRECTORY_PATH,
    ) -> None:
        """
        This method imports the table from the CSV file into the database.

        :param table_name: The name of the table to import.
        :param table_file_name: The name of the CSV file to import.
        :param base_path: The base path where the CSV file is stored.
        """
        # If the table name is not given, use the current table name
        if table_name is None:
            table_name = self.table_name
        if table_file_name is None:
            table_file_name = table_name

        # Create the name of the table file as the name of the table
        filename = table_file_name + ".csv"
        # Compute the full path to the SQL CSV table file
        file_path = os.path.join(base_path, filename)

        # If the file exists
        if os.path.exists(file_path):
            # Copy the file into the table
            self.connection_handler.import_table(
                file_path=file_path,
                table_name=table_name,
            )
        else:
            raise FileNotFoundError(f"The file {file_path} does not exist")

    def fetch(
            self,
            filters: Optional[List[SqlQueryComponent]] = None,
            instance_builder: Optional[Callable[[List[Any]], Any]] = None,
            columns_order: Optional[List[str]] = None,
    ) -> Union[None, Any, dict[int, Any]]:
        """
        This method fetches a set of rows from the database by using the given filters. It translates every row
        in an instance, if the instance builder is given.

        :param filters: The list of filters to apply to the query.
        :param instance_builder: The function that builds the instance from the row.
        :param columns_order: The order of the columns to fetch.

        :return: The list of instances built from the rows fetched, if the instance builder is given.
        The list of rows, otherwise.
        """
        # If there is at least one query filter
        if filters is not None and len(filters) > 0:
            # Set up the where clause for the SQL query
            where_clause = SqlQuery(
                sql_query_components=filters
            ).to_sql_string()
        else:
            # If there are no query filters, leave the where clause empty
            where_clause = ("", [])

        # Compose the string to select all the columns
        select_substring = '*'
        # If the columns order is given
        if columns_order is not None:
            # Select only the columns in the given order
            select_substring = ', '.join(columns_order)

        # Run the complete fetch query
        results = self.connection_handler.fetch(
            sql_string=f"""
                SELECT {select_substring}
                FROM {self.table_name}
                {where_clause[0]}
            """,
            parameters=tuple(
                where_clause[1]
            ),
            fetch_only_one=False,
        )

        # If no results have been found
        if len(results) == 0:
            return None
        else:
            # If the instance builder is not given
            if instance_builder is None:
                # Return the list of rows as fetched from the database
                return results
            else:
                # If there is only one result
                if len(results) == 1:
                    # Return the instance built from the row
                    return instance_builder(
                        results[0]
                    )
                else:
                    # Return a dictionary that maps the identifier of the instance
                    # to the instance itself
                    return {
                        result[0]: instance_builder(
                            result
                        )
                        for result in results
                    }

    def update(
            self,
            where_clause: SqlQueryComponent,
            values_to_update: list[SqlQueryComponent]
    ) -> None:
        """
        This method updates the table by using the where clause and all the new values to update.

        :param where_clause: The where clause parameters of the update query.
        :param values_to_update: The list of column name-value pairs to update.
        """
        sql_string = f"UPDATE {self.table_name} SET"

        if len(values_to_update) == 0:
            raise ValueError("At least one value to update must be given")
        elif len(where_clause.column_values) == 0:
            raise ValueError("At least one value to filter the rows to update must be given")
        else:
            parameters = []
            # For each SQL column - value pair
            for value_to_update in values_to_update[:-1]:
                if len(value_to_update.column_values) > 1:
                    raise ValueError("The value to update must be a single value")
                else:
                    # Build the SQL string and the list of parameters to set the new value in the table column
                    tmp_sql_string, tmp_parameters = value_to_update.to_sql_string()
                    # Update the SQL string
                    sql_string += f"""
                        {tmp_sql_string},
                    """
                    # Update the list of parameters
                    parameters.extend(tmp_parameters)

            # Set the last column - value pair in the SQL string
            tmp_sql_string, tmp_parameters = values_to_update[-1].to_sql_string()
            sql_string += f"""
                {tmp_sql_string}
            """
            parameters.extend(tmp_parameters)

            tmp_sql_string, tmp_parameters = where_clause.to_sql_string()
            sql_string += f"""
                WHERE {tmp_sql_string}
            """
            parameters.extend(tmp_parameters)

            self.connection_handler.execute(
                sql_string=sql_string,
                parameters=tuple(
                    parameters
                )
            )

    def add(
            self,
            filters: List[SqlQueryComponent] | List[List[SqlQueryComponent]],
            instance_builder: Callable[[list[Any]], Any],
            pk_column_names: Optional[list[str]] = None,
            id_column_name: Optional[TableColumnName] = TableColumnName.IDENTIFIER.value,
    ) -> Any:
        """
        This method adds a new row to the table by using the given values. It returns the instance built from the
        values.

        :param filters: The filters used to populate the column values in the new row.
        :param instance_builder: The function that builds the instance from the row.
        :param pk_column_names: The name of the column(s) that contain the primary key of each row.
        :param id_column_name: The name of the column that contains the identifier of the row.
        """
        if pk_column_names is None:
            pk_column_names = [TableColumnName.IDENTIFIER.value]

        if len(filters) == 0:
            raise ValueError("At least one row to insert must be given")
        else:
            # If only one row must be added
            if not isinstance(filters[0], list):
                # Create a list of 1 element (i.e., the list of values for the unique
                # row to be added)
                filters = [filters]

            # Get, from the first set of filters, all the column names
            column_names = [
                filter_.column_name
                for filter_ in filters[0]
            ]

            pk_filters: Optional[list[list[SqlQueryComponent]]] = None
            values: Optional[list[list[Any]]] = None
            try:
                # For each of the rows to be inserted
                for i in range(len(filters)):
                    tmp_values = []
                    tmp_pk_filters = []
                    # Take the current list of values to be inserted
                    for filter_ in filters[i]:
                        tmp_value = filter_.column_values[0]
                        # If the current column is the primary key column
                        if filter_.column_name in pk_column_names:
                            # Append the current value to the list of primary keys
                            tmp_pk_filters.append(
                                SqlQueryComponent(
                                    column_name=filter_.column_name,
                                    column_values=[tmp_value]
                                )
                            )
                        # Append the current value to the list of values
                        tmp_values.append(tmp_value)
                    if pk_filters is None:
                        pk_filters = []
                    # Add to the list the current filter to get the row by PK
                    pk_filters.append(tmp_pk_filters)
                    if values is None:
                        values = []
                    # Add the current list of values to the list of values to be inserted
                    values.append(tmp_values)

                # Perform the batch insertion and get the identifiers of the new rows
                inserted_ids: list[tuple] = self.connection_handler.add_batched(
                    column_names=column_names,
                    table_name=self.table_name,
                    values=values,
                    id_column_name=id_column_name,
                )

                # Prepare the list of instances to be returned
                instances = []
                # Set, in each of the values, the identifier of the new row
                for i in range(len(values)):
                    # Get the list of values for the current row
                    tmp_values = values[i]
                    # Add the identifier of the new row at the beginning of the list
                    tmp_values.insert(0, inserted_ids[i][0])
                    # Create the instance from the values
                    instances.append(
                        instance_builder(
                            tmp_values
                        )
                    )

                # If only one instance has been created, return it
                if len(instances) == 1:
                    return instances[0]
                else:
                    return instances
            except (psycopg2.errors.UniqueViolation, psycopg2.IntegrityError):
                # If the insert of the table has been already done, then rollback the current transaction
                self.connection_handler.connection.rollback()
                if pk_filters is not None:
                    returned_instances = []
                    for pk_filter in pk_filters:
                        # Get the instance fetched from the primary key
                        tmp_instance = self.fetch(
                            filters=pk_filter,
                            instance_builder=instance_builder,
                        )
                        # If the instance has been found
                        if tmp_instance is not None:
                            returned_instances.append(tmp_instance)
                    if len(returned_instances) == 1:
                        return returned_instances[0]
                    else:
                        return returned_instances
                else:
                    raise ValueError("The primary key column name must be given")
