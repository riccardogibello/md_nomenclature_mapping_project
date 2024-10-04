import os

import pandas as pd

from src.business_logic.database_interfaces.connection_handler import DatabaseInformation, ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.fda_code_data_handler import \
    FdaCodeDataHandler
from src.business_logic.utilities.os_utilities import unzip_folder
from src.data_model.nomenclature_codes.fda_code import SubmissionType
from src.__directory_paths import SOURCE_DATA_DIRECTORY_PATH


def clean_and_store_fda_codes(
        database_information: DatabaseInformation
) -> None:
    """
    This method cleans and stores in a database table all the FDA classification codes.

    :param database_information: The information of the database to connect to.
    """
    connection_handler = ConnectionHandler(
        database_information=database_information
    )
    # Instantiate the handler of the FDA codes
    fda_data_handler = FdaCodeDataHandler(
        connection_handler=connection_handler
    )
    fda_file_txt_path = SOURCE_DATA_DIRECTORY_PATH + 'foiclass.txt'
    fda_file_zip_path = SOURCE_DATA_DIRECTORY_PATH + 'foiclass.zip'

    if not os.path.exists(fda_file_txt_path):
        # Unzip the FDA file containing all the nomenclature of FDA codes
        unzip_folder(
            destination_folder_path=SOURCE_DATA_DIRECTORY_PATH,
            zip_file_path=fda_file_zip_path,
        )

    if os.path.exists(fda_file_txt_path):
        # Open the file as a dataframe, with | as a separator
        fda_dataframe = pd.read_csv(
            fda_file_txt_path,
            sep='|',
            encoding_errors='ignore',
        )
        # Identify columns with float64 dtype
        float_columns = fda_dataframe.select_dtypes(include=['float64']).columns

        # Cast these columns to string before filling NaN values
        fda_dataframe[float_columns] = fda_dataframe[float_columns].astype(str)

        # Replace any NaN value with an empty string
        fda_dataframe.fillna('', inplace=True)

        # Drop the unnecessary columns
        fda_dataframe.drop(
            columns=[
                "UNCLASSIFIED_REASON",
                "GMPEXEMPTFLAG",
                "THIRDPARTYFLAG",
                "REVIEWCODE",
                "REGULATIONNUMBER",
                "SummaryMalfunctionReporting",
            ],
            inplace=True
        )

        fda_dataframe.rename(
            columns={
                "REVIEW_PANEL": "review_panel",
                "MEDICALSPECIALTY": "medical_specialty",
                "PRODUCTCODE": "product_code",
                "DEVICENAME": "device_name",
                "DEVICECLASS": "device_class",
                "SUBMISSION_TYPE_ID": "submission_type_id",
                "DEFINITION": "definition",
                "PHYSICALSTATE": "physical_state",
                "TECHNICALMETHOD": "technical_method",
                "TARGETAREA": "target_area",
                "Implant_Flag": "is_implant",
                "Life_Sustain_support_flag": "is_life_sustaining",
            },
            inplace=True
        )

        # For each row, corresponding to an FDA code, add it to the database
        for index, row in fda_dataframe.iterrows():
            # Get the information about the FDA code
            review_panel = str(row['review_panel'])
            medical_specialty = str(row['medical_specialty'])
            product_code = str(row['product_code'])
            device_name = str(row['device_name'])
            try:
                device_class = int(row['device_class'])
            except ValueError:
                device_class = -1
            try:
                submission_type_id = int(row['submission_type_id'])
                if 0 <= submission_type_id <= 7:
                    submission_type = SubmissionType.get_enum_from_value(
                        submission_type_id,
                        enum_class=SubmissionType,
                    )
                else:
                    submission_type = None
            except ValueError:
                submission_type = None

            definition = str(row['definition'])
            physical_state = str(row['physical_state'])
            technical_method = str(row['technical_method'])
            target_area = str(row['target_area'])
            is_implant = bool(row['is_implant'])
            is_life_sustaining = bool(row['is_life_sustaining'])

            # Add the FDA code to the database
            fda_data_handler.add_fda_code(
                review_panel=review_panel,
                medical_specialty=medical_specialty,
                product_code=product_code,
                device_name=device_name,
                device_class=device_class,
                submission_type_id=submission_type if submission_type is None else submission_type.value,
                definition=definition,
                physical_state=physical_state,
                technical_method=technical_method,
                target_area=target_area,
                is_implant=is_implant,
                is_life_sustaining=is_life_sustaining,
            )

        # Delete the FDA file
        os.remove(fda_file_txt_path)
