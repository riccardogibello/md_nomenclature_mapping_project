import random

import pandas as pd
from matplotlib import pyplot as plt, patches as mpatches

from src.business_logic.database_interfaces.connection_handler import ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.mappings_data_handlers.device_mapping_data_handler import \
    DeviceMappingDataHandler
from src.business_logic.database_interfaces.data_handlers.mappings_data_handlers.mapping_data_handler import \
    MappingDataHandler
from src.business_logic.database_interfaces.data_handlers.nomenclature_data_handlers.emdn_code_data_handler import \
    EmdnCodeDataHandler
from src.data_model.enums import TranslationDirection


def store_device_and_nomenclature_correspondences(
        base_output_path: str,
        dataframe: pd.DataFrame,
        connection_handler: ConnectionHandler,
):
    """
    This method stores the device and nomenclature correspondences that are contained in the dataframe stored in the
    given file_path. The nomenclature correspondences are stored in a separate file, not to repeat too many times
    the same strings due to the same nomenclature matches in different device matches.

    :param base_output_path: The base path to store the statistics of the GMDN-EMDN correspondences.
    :param dataframe: The dataframe containing the device and nomenclature correspondences.
    :param connection_handler: The information to access the database.
    """
    # The dataframe must contain the following columns:
    # - DEVICE_ID_DF1
    # - DEVICE_ID_DF2
    # - EMDN_ID
    # - GMDN_ID
    # - COMPANY_NAME_SIMILARITY (if any)
    # - SIMILARITY (either on the device name or the catalogue code)

    # Store the GMDN and EMDN correspondences in a separate file
    _store_nomenclature_correspondences(
        base_output_path=base_output_path,
        gmdn_emdn_correspondences=dataframe.drop_duplicates(),
        connection_handler=connection_handler,
    )


def _store_nomenclature_correspondences(
        base_output_path: str,
        gmdn_emdn_correspondences: pd.DataFrame,
        connection_handler: ConnectionHandler,
) -> None:
    """
    This method stores the nomenclature correspondences in the database and computes the statistics over the
    correspondences.

    :param base_output_path: The folder path to store the statistics of the GMDN-EMDN correspondences.
    :param gmdn_emdn_correspondences: The dataframe containing the GMDN-EMDN correspondences.
    :param connection_handler: The information to access the database.
    """
    # Create the handler to store the mapping data
    device_mapping_data_handler = DeviceMappingDataHandler(
        connection_handler=connection_handler,
    )
    # Add all the matches between devices and nomenclature codes in the proper tables
    device_mapping_data_handler.add_batched_device_mappings(
        mappings=gmdn_emdn_correspondences,
    )

    # Compute the statistics over the found GMDN-EMDN correspondences
    _build_stats(
        base_output_path,
        mapping_data_handler=device_mapping_data_handler.mapping_data_handler,
        new_file_name='emdn_frequencies.png',
        connection_handler=connection_handler,
    )


def _build_stats(
        base_path: str,
        mapping_data_handler: MappingDataHandler,
        new_file_name: str,
        connection_handler: ConnectionHandler,
) -> None:
    """
    This method builds the statistics over the found GMDN-EMDN correspondences. These consist of the frequencies of the
    EMDN categories in the found correspondences. The output is a bar plot with the frequencies of the EMDN categories.

    :param base_path: The base path to store the output of this method.
    :param mapping_data_handler: The handler to access the mappings between GMDN and EMDN codes.
    :param new_file_name: The name of the file to store the statistics.
    :param connection_handler: The information to access the database.
    """
    # Instantiate the handler for the EMDN codes
    emdn_code_data_handler = EmdnCodeDataHandler(
        connection_handler=connection_handler,
    )

    # Load all the current mappings
    emdn_id__mapping = mapping_data_handler.get_mappings(
        translation_direction=TranslationDirection.FROM_EMDN_TO_GMDN,
        force_refresh=True,
    )
    # Get a list of all the unique emdn ids
    emdn_ids = list(emdn_id__mapping.keys())
    # Fetch all the EMDN codes
    emdn_codes = emdn_code_data_handler.get_emdn_codes(
        identifiers=emdn_ids,
    )
    if emdn_codes is None:
        return
    else:
        if type(emdn_codes) is not dict:
            emdn_codes = {
                emdn_codes.identifier: emdn_codes
            }

        row_list = []
        for emdn_id, mapping_list in emdn_id__mapping.items():
            # Get the instance of the EMDN code
            emdn_code = emdn_codes[emdn_id]
            for mapping in mapping_list:
                row_list.append({
                    'EMDN_CLASSIFICATION': emdn_code.emdn_code,
                    'GMDN_CLASSIFICATION': mapping.gmdn_id,
                })

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(
            columns=['EMDN_CLASSIFICATION', 'GMDN_CLASSIFICATION'],
            data=row_list
        )
        # Get all the EMDN_CLASSIFICATION strings with duplicates
        emdn_classification_duplicates = df['EMDN_CLASSIFICATION'].value_counts()
        emdn_categories_counts = {}
        total_count = 0
        for emdn_classification, count in emdn_classification_duplicates.items():
            # Get the EMDN category
            emdn_category = emdn_classification[0]
            # Add the count to the category
            if emdn_category in emdn_categories_counts:
                emdn_categories_counts[emdn_category] += count
            else:
                emdn_categories_counts[emdn_category] = count
            total_count += count

        # Plot the EMDN categories
        categories = [
            category
            for category in emdn_categories_counts.keys()
        ]
        # Order them in alphabetical order
        categories.sort()
        counts = [
            emdn_categories_counts[category]
            for category in categories
        ]

        # Setting a larger figure size
        plt.figure(figsize=(12, 6))
        random.seed(7)
        # Build a list of random colors for each category, with a random distribution between 0.5 and 1
        colors = [
            (
                random.uniform(0.2, 1),
                random.uniform(0.2, 1),
                random.uniform(0.2, 1),
                0.75
            )
            for _ in range(len(categories))
        ]
        # Create a bar plot with a color for each category
        plt.bar(categories, counts, color=colors)
        # plt.xscale('log')  # Set x-axis to log scale
        plt.yscale('log')  # Set y-axis to log scale
        plt.xlabel('EMDN Category')
        plt.ylabel('Frequency')
        plt.title('GMDN-EMDN Vocabulary Category Frequencies')
        # Add a legend on the right of the figure
        # for each EMDN category to report the related frequency
        # Create legend handles manually
        legend_handles = [
            mpatches.Patch(
                color=color,
                label=(
                        label + '= ' +
                        str(emdn_categories_counts[label]) + " " +
                        "(" + str(round(emdn_categories_counts[label] / total_count * 100, 2)) + "%)"
                )
            )
            for color, label in zip(colors, categories)
        ]
        plt.legend(
            handles=legend_handles,
            title='Category Frequencies (vocabulary entries: ' + str(total_count) + ')',
            bbox_to_anchor=(0.5, -0.15),
            ncol=6,
            # Set the legend under the title
            loc='upper center',
        )
        # Remove the top and bottom space of the plot
        plt.subplots_adjust(top=0.95, bottom=0.3)
        plt.subplots_adjust(left=0.06)
        # Remove space on the right of the plot
        plt.subplots_adjust(right=0.98)
        # Save the plot as an image file (e.g., PNG)
        plt.savefig(base_path + new_file_name)
