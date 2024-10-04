from enum import Enum
from typing import Optional

from src.business_logic.database_interfaces.__table_constants import TableColumnName

DEVICE_SUFFIX = '_d'
MD_SUFFIX = '_md'


class DataFrameColumnName(Enum):
    MANUFACTURER_NAME_COLUMN = ''.join([TableColumnName.NAME.value + MD_SUFFIX])
    MANUFACTURER_CLEAN_NAME_COLUMN = ''.join([TableColumnName.CLEAN_NAME.value + MD_SUFFIX])
    STANDARDIZED_MANUFACTURER_ID_COLUMN = ''.join([TableColumnName.STANDARDIZED_MANUFACTURER_ID.value + MD_SUFFIX])
    ORIGINAL_MANUFACTURER_STATE_ID_COLUMN = ''.join([TableColumnName.ORIGINAL_STATE_ID.value + MD_SUFFIX])

    CLEAN_DEVICE_ID_COLUMN = ''.join([TableColumnName.CLEAN_DEVICE_ID.value, DEVICE_SUFFIX])
    DEVICE_NAME_COLUMN = ''.join([TableColumnName.NAME.value + DEVICE_SUFFIX])
    DEVICE_CLEAN_NAME_COLUMN = ''.join([TableColumnName.CLEAN_NAME.value + DEVICE_SUFFIX])
    CLEAN_MANUFACTURER_ID_COLUMN = ''.join([TableColumnName.CLEAN_MANUFACTURER_ID.value + DEVICE_SUFFIX])
    CATALOGUE_CODE_COLUMN = ''.join([TableColumnName.CATALOGUE_CODE.value + DEVICE_SUFFIX])
    DEVICE_TYPE_COLUMN = ''.join([TableColumnName.DEVICE_TYPE.value + DEVICE_SUFFIX])
    HIGH_RISK_DEVICE_TYPE_COLUMN = ''.join([TableColumnName.HIGH_RISK_DEVICE_TYPE.value + DEVICE_SUFFIX])
    PRODUCT_CODE_COLUMN = ''.join([TableColumnName.PRODUCT_CODE.value + DEVICE_SUFFIX])
    PANEL_COLUMN = ''.join([TableColumnName.PANEL.value + DEVICE_SUFFIX])
    STANDARDIZED_DEVICE_ID_COLUMN = ''.join([TableColumnName.STANDARDIZED_DEVICE_ID.value + DEVICE_SUFFIX])

    EMDN_ID_COLUMN = ''.join([TableColumnName.EMDN_ID.value + DEVICE_SUFFIX])
    EMDN_CODE_COLUMN = ''.join([TableColumnName.CODE.value + DEVICE_SUFFIX])

    GMDN_ID_COLUMN = ''.join([TableColumnName.GMDN_ID.value + DEVICE_SUFFIX])
    TERM_NAME_COLUMN = ''.join([TableColumnName.NAME.value + '_g'])
    DEFINITION_COLUMN = ''.join([TableColumnName.DEFINITION.value + '_g'])


COMMON_COLUMNS = [
    DataFrameColumnName.MANUFACTURER_NAME_COLUMN.value,
    DataFrameColumnName.MANUFACTURER_CLEAN_NAME_COLUMN.value,
    DataFrameColumnName.STANDARDIZED_MANUFACTURER_ID_COLUMN.value,
    DataFrameColumnName.ORIGINAL_MANUFACTURER_STATE_ID_COLUMN.value,
    DataFrameColumnName.CLEAN_DEVICE_ID_COLUMN.value,
    DataFrameColumnName.DEVICE_NAME_COLUMN.value,
    DataFrameColumnName.DEVICE_CLEAN_NAME_COLUMN.value,
    DataFrameColumnName.CLEAN_MANUFACTURER_ID_COLUMN.value,
    DataFrameColumnName.CATALOGUE_CODE_COLUMN.value,
    DataFrameColumnName.DEVICE_TYPE_COLUMN.value,
    DataFrameColumnName.HIGH_RISK_DEVICE_TYPE_COLUMN.value,
    DataFrameColumnName.PRODUCT_CODE_COLUMN.value,
    DataFrameColumnName.STANDARDIZED_DEVICE_ID_COLUMN.value,
]
ITALIAN_DATAFRAME_COLUMNS = COMMON_COLUMNS.copy()
ITALIAN_DATAFRAME_COLUMNS.extend([
    DataFrameColumnName.EMDN_ID_COLUMN.value,
    DataFrameColumnName.EMDN_CODE_COLUMN.value,
])


def get_italian_empty_rows() -> list[tuple[
    str,
    str,
    Optional[int],
    int,
    Optional[int],
    str,
    str,
    Optional[int],
    str,
    str,
    Optional[str],
    Optional[str],
    Optional[int],
    Optional[int],
    Optional[str],
]]:
    return []


AMERICAN_DATAFRAME_COLUMNS = COMMON_COLUMNS.copy()
AMERICAN_DATAFRAME_COLUMNS.extend([
    DataFrameColumnName.GMDN_ID_COLUMN.value,
    DataFrameColumnName.TERM_NAME_COLUMN.value,
    DataFrameColumnName.DEFINITION_COLUMN.value,
])


def get_american_empty_rows() -> list[tuple[
    str,
    str,
    Optional[int],
    int,
    Optional[int],
    str,
    str,
    Optional[int],
    str,
    str,
    Optional[str],
    Optional[str],
    Optional[int],
    Optional[int],
    Optional[str],
    Optional[str],
]]:
    return []
