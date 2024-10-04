import os
import pstats
import time
from typing import Optional

from src.business_logic.database_interfaces.connection_handler import DatabaseInformation, local_database_information
from src.business_logic.database_interfaces.tables_portability import (
    export_tables, build_and_export_table
)
from src.business_logic.pipeline.data_preprocessing.nomenclature_mappings_cleaning import clean_nomenclature_mappings
from src.business_logic.pipeline.emdn_database_loading import clean_and_store_emdn_codes
from src.business_logic.pipeline.fda_database_loading import clean_and_store_fda_codes
from src.business_logic.pipeline.data_preprocessing.medical_devices_scraping_pipeline import \
    load_medical_device_data_into_database
from src.business_logic.pipeline.market_cross_checking.cross_check_markets import cross_check_markets
from src.business_logic.pipeline.model_training.datasets_handling import build_train_test_datasets
from src.business_logic.pipeline.model_training.train import train_and_test
from src.business_logic.utilities.os_utilities import create_directory
from src.data_model.enums import AbstractEnum, CountryName
from src.__constants import RANDOM_SEED_INT
from src.__directory_paths import PIPELINE_COMPUTATION_DIRECTORY_PATH
from src.__file_paths import EMDN_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH

import cProfile


class PipelinePhase(AbstractEnum):
    LOAD_EMDN_DATA = 'LoadEmdnData'
    LOAD_FDA_DATA = 'LoadFdaData'
    LOAD_MEDICAL_DEVICE_DATA = 'LoadMedicalDeviceData'
    START_MAPPING_SCRAPING = 'StartMappingScraping'
    TRAIN_MODEL = 'TrainModel'


def _execute_pipeline_phase(
        **kwargs,
) -> None:
    """
    This method starts the given phase of the pipeline process.

    The ServerState instance must be brand-new, due to some CursorHandler pickling errors passing it around
    different processes. Therefore, this does not correspond to the state instance of the main process.

    :param phase: The pipeline phase which has to be performed.
    """
    # Verify that in the kwargs there is the database information
    # and the name of the phase
    if 'database_information' not in kwargs:
        raise Exception("The database information must be passed to the pipeline phase execution function.")
    if 'phase' not in kwargs:
        raise Exception("The phase of the pipeline must be passed to the pipeline phase execution function.")
    else:
        phase = kwargs['phase']

    time.sleep(2)

    if phase == PipelinePhase.LOAD_EMDN_DATA:
        # Load all the EMDN codes into the database from the initial
        # CSV file
        clean_and_store_emdn_codes(
            csv_emdn_file_path=EMDN_FILE_PATH,
            database_information=kwargs['database_information'],
        )

    elif phase == PipelinePhase.LOAD_FDA_DATA:
        # Load all the FDA codes into the database from the initial
        # CSV file
        clean_and_store_fda_codes(
            database_information=kwargs['database_information']
        )

    elif phase == PipelinePhase.LOAD_MEDICAL_DEVICE_DATA:
        if "countries" not in kwargs:
            raise Exception("The countries must be passed to the pipeline phase execution function.")
        # Call the method to load all the medical device data into the database
        # from the starting data sources
        load_medical_device_data_into_database(
            database_information=kwargs['database_information'],
            countries=kwargs['countries'],
        )

    elif phase == PipelinePhase.START_MAPPING_SCRAPING:
        # Cross-check the European and American markets to find
        # correspondences between EMDN, GMDN, and FDA specialty codes
        cross_check_markets(
            company_name_threshold=85,
        )
        # Build a CSV file containing the correspondences
        # between EMDN, GMDN and FDA codes
        build_and_export_table(
            table_name='emdn_gmdn_fda',
            database_information=kwargs['database_information'],
            columns_order=[
                'original_device_mapping_id',
                'emdn_id',
                'emdn_code',
                'emdn_description',
                'emdn_category',
                'gmdn_id',
                'gmdn_term_name',
                'product_code',
                'device_name',
                'medical_specialty',
                'panel',
            ],
            executable_name='create_emdn_gmdn_fda_table.sql',
            drop_table=True
        )
        # Clean the created mappings between EMDN-GMDN-FDA codes
        # and store them in a separated CSV file
        clean_nomenclature_mappings()

    elif phase == PipelinePhase.TRAIN_MODEL:
        # Build the datasets containing the train and test samples of
        # correspondences between GMDN Term Names and EMDN categories
        # starting from the cleaned file containing EMDN-GMDN-FDA
        # correspondences
        build_train_test_datasets(
            random_seed=RANDOM_SEED_INT,
            include_emdn_in_training=False,
            _train_file_path=TRAIN_FILE_PATH,
            _test_file_path=TEST_FILE_PATH,
            outliers=['W']
        )
        # Train and test the EMDN category predictor
        train_and_test()

    else:
        raise Exception("The pipeline phase is not recognized.")


def _run_and_profile_phase(
        phase: PipelinePhase,
        database_information: DatabaseInformation,
        **kwargs,
) -> None:
    """
    This method runs the pipeline phase and profiles it, storing the results in a file under the
    statistics folder.

    :param phase: The name of the phase to be executed.
    :param database_information: The information to connect to the database.
    :param kwargs: The arguments to pass to the pipeline phase.
    """
    base_statistics_folder = os.path.join(PIPELINE_COMPUTATION_DIRECTORY_PATH, phase.name + '/')
    create_directory(base_statistics_folder)

    # Instantiate the profiler
    prof = cProfile.Profile()
    # Run the pipeline phase
    prof.runctx(
        cmd=f"_execute_pipeline_phase("
            f"  **kwargs, "
            f"  database_information = database_information, "
            f"  phase = phase"
            f")",
        globals=globals(),
        locals=locals(),
    )

    # Create a unique string containing the date and the seconds
    # to avoid overwriting the files
    date_string = time.strftime("%Y_%m_%d__%H_%M_%S")

    prof_file_path = base_statistics_folder + phase.name.lower() + '_' + date_string + '.prof'
    txt_file_path = base_statistics_folder + phase.name.lower() + '_' + date_string + '.txt'
    prof.dump_stats(prof_file_path)
    stream = open(txt_file_path, 'w')
    stats = pstats.Stats(prof_file_path, stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()
    stream.close()
    # Delete the prof file
    os.remove(prof_file_path)


def execute_pipeline(
        database_information: Optional[DatabaseInformation] = None,
) -> None:
    """
    This method, provided with the server_state_singleton instance, starts the whole pipeline process, which:
        1.  extracts the EMDN codes from the CSV file and tokenize them by using the MetaMap Lite tool;
        2.  extracts, cleans and stores the CSV files for the American and Italian markets of medical devices;
        2.  extracts and tokenizes all the GMDN codes (Term Names + Definitions) found to be related to at least
            one medical device in the American market;
        3.  builds a vocabulary of translations between GMDN and EMDN codes by intersecting the information included
            in the two newly created CSVs;
        4.  loads from disk a pretrained language model in order to evaluate similarity between medical strings;
            cleans its vocabulary by eliminating the strings which are not of interest in the medical field (by
            using the MetaMapLite executable);
        5.  tests the language model algorithm on a validation set;
    """
    if database_information is None:
        database_information = local_database_information

    pipeline_phase__arguments: dict[PipelinePhase, dict | None] = {
        PipelinePhase.LOAD_EMDN_DATA: None,
        PipelinePhase.LOAD_FDA_DATA: None,
        PipelinePhase.LOAD_MEDICAL_DEVICE_DATA: {
            'countries': [
                CountryName.USA,
                CountryName.ITALY,
            ]
        },
        PipelinePhase.START_MAPPING_SCRAPING: None,
        PipelinePhase.TRAIN_MODEL: None,
    }

    for phase, arguments in pipeline_phase__arguments.items():
        # If the current phase is the training one
        if phase == PipelinePhase.TRAIN_MODEL:
            # Export all the tables, in case they are needed
            export_tables(
                database_information=database_information
            )

        if arguments is not None and type(arguments) is dict:
            _run_and_profile_phase(
                **arguments,
                phase=phase,
                database_information=database_information,
            )
        else:
            _run_and_profile_phase(
                phase=phase,
                database_information=database_information,
            )
