from typing import Callable

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.emdn_code_data_handler import \
    EmdnCodeDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.gmdn_code_data_handler import \
    GmdnCodeDataHandler
from src.business_logic.database_interfaces.table_handlers.abstract_table_handler import AbstractTableHandler


class AbstractNomenclatureTableHandler(AbstractTableHandler):
    """
    This is an abstract table handler, used to give access to the GMDN and EMDN nomenclatures.
    """
    # emdn_code_data_handler = the handler to access the EMDN codes' data.
    emdn_code_data_handler: EmdnCodeDataHandler

    # gmdn_code_data_handler = the handler used to retrieve specific GMDN codes.
    gmdn_code_data_handler: GmdnCodeDataHandler

    def __init__(
            self,
            emdn_code_data_handler: EmdnCodeDataHandler,
            gmdn_code_data_handler: GmdnCodeDataHandler,
            table_name: str,
            reset_table: bool,
            initializing_table_statement: Callable[[str], str],
            connection_handler: ConnectionHandler,
    ):
        super().__init__(
            connection_handler=connection_handler,
            table_name=table_name,
            reset_table=reset_table,
            initializing_table_statement=initializing_table_statement,
        )

        self.emdn_code_data_handler = emdn_code_data_handler
        self.gmdn_code_data_handler = gmdn_code_data_handler
