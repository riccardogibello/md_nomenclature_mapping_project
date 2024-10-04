import os
from pathlib import Path

from src.business_logic.pipeline.execute_pipeline import execute_pipeline
from src.business_logic.utilities.emissions_tracker import tracked_function
from src.__directory_paths import LOGGING_DIRECTORY_PATH, DIRECTORIES, SOURCE_DATA_DIRECTORY_PATH
from src.business_logic.utilities.os_utilities import create_directory

if __name__ == '__main__':
    # Setup each directory in the DIRECTORIES list
    for directory_path in DIRECTORIES:
        if directory_path == SOURCE_DATA_DIRECTORY_PATH:
            if not os.path.exists(SOURCE_DATA_DIRECTORY_PATH):
                raise FileNotFoundError(
                    f'Load the "SourceData" folder in the "OutputData" folder before running the pipeline.'
                )
        else:
            create_directory(directory_path)
    # Run the pipeline to refactor the public data sources,
    # extract the relevant data for the model training, and
    # train the baseline model for predicting the EMDN code
    # given a specific GMDN Term Name
    tracked_function(
        _command=execute_pipeline,
        _output_dir=Path(LOGGING_DIRECTORY_PATH),
        _interval=60,
    )
