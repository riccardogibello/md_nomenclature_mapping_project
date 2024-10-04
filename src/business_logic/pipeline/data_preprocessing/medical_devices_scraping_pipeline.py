from src.business_logic.database_interfaces.connection_handler import DatabaseInformation
from src.business_logic.database_interfaces.tables_portability import build_and_export_table
from src.business_logic.pipeline.data_preprocessing.american_data_loading import load_american_medical_device_data
from src.business_logic.pipeline.data_preprocessing.italian_data_preprocessing import load_italian_medical_device_data
from src.data_model.enums import CountryName
from src.__directory_paths import SOURCE_DATA_DIRECTORY_PATH
from src.__file_paths import OLD_ITALIAN_FULL_PATH


def load_medical_device_data_into_database(
        database_information: DatabaseInformation,
        countries: list[CountryName],
) -> None:
    """
    This method fires the pipelines in order to clean, if needed, the files containing the Italian and American
    medical devices' company data. Then, it exports the table containing the GMDN-FDA code correspondences.

    :param database_information: The database information to connect to the database.
    :param countries: The countries from which the medical devices' data will be loaded.
    """
    for country in countries:
        if country == CountryName.ITALY:
            # Call the method to load the Italian medical devices' data
            # from the CSV repository into the database
            load_italian_medical_device_data(
                old_csv_file_path=OLD_ITALIAN_FULL_PATH,
                database_information=database_information,
            )
        elif country == CountryName.USA:
            # Call the method to load the American medical devices' data
            # from the XML repository into the database
            load_american_medical_device_data(
                _source_directory_path=SOURCE_DATA_DIRECTORY_PATH,
                database_information=database_information,
            )
        else:
            raise ValueError('Invalid country name')

    # Build a CSV file containing the correspondences
    # between GMDN and FDA codes
    build_and_export_table(
        table_name='gmdn_fda',
        database_information=database_information,
        columns_order=[
            'gmdn_id',
            'gmdn_term_name',
            'product_code',
            'panel',
            'medical_specialty',
            'device_name',
        ],
        executable_name='create_gmdn_fda_table.sql',
        drop_table=True
    )
