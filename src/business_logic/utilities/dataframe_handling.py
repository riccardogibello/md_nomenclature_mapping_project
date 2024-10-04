from typing import Type, Optional, Any

import pandas as pd
from pandas import DataFrame, Series
import numpy as np


def get_row_value(
        row: Series | np.ndarray,
        column_name: int | str | list[str] | list[int],
        casting_type: Optional[Type] | list[Optional[Type]] = None
) -> Any | list[Any]:
    """
    Given a row and a column name, it returns the value of the cell that is in the intersection of the row and the
    column.

    :param row: The row from which the value(s) must be extracted.
    :param column_name: The name(s) of the column(s) from which the value(s) must be extracted.
    :param casting_type: The type(s) to which the value(s) must be cast. It must have the same length as column_name.

    :return: The value(s) of the required cell(s).
    """
    if isinstance(column_name, list):
        assert len(column_name) == len(casting_type), "The length of column_name and casting_type is different."
        returned_values = []
        column_names: list[str] = column_name
        casting_types: list[Optional[Type]] = casting_type
        for column_name, casting_type in zip(column_names, casting_types):
            value = row[column_name] if not pd.isna(row[column_name]) else None
            if casting_type is not None and value is not None:
                returned_values.append(casting_type(value))
            else:
                returned_values.append(value)
        return returned_values
    else:
        value = row[column_name] if not pd.isna(row[column_name]) else None
        if casting_type is not None and value is not None:
            return casting_type(value)
        else:
            return value


def clean_dataframe_from_null_values(
        dataframe: DataFrame,
) -> None:
    """
    Given a dataframe as input, it modifies it by replacing the empty strings with NaN. It removes every NaN value,
    and it resets the index.

    :param dataframe: The dataframe to be cleaned.

    :return: The dataframe instance passed in as a parameter, after cleaning.
    """
    dataframe.replace(
        '',
        np.nan,
        inplace=True
    )
    dataframe.dropna(
        axis=0,
        inplace=True
    )

    dataframe.reset_index(
        inplace=True,
        drop=True
    )
