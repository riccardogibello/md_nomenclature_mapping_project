import json
import os.path
import random
import shutil
import xml
from typing import List
from xml.etree import ElementTree

import pandas as pd

from src.business_logic.database_interfaces.connection_handler import DatabaseInformation, ConnectionHandler
from src.business_logic.database_interfaces.data_handlers.country_data_handler import CountryDataHandler
from src.business_logic.pipeline.data_preprocessing.data_cleaning import clean_md_catalogue_code, \
    clean_company_name, clean_string, clean_extra_blank_spaces
from src.business_logic.pipeline.data_preprocessing.dataframe_column_names import get_american_empty_rows, \
    AMERICAN_DATAFRAME_COLUMNS
from src.business_logic.pipeline.data_preprocessing.store_batched_data import store_batched_american_data
from src.business_logic.utilities.concurrentifier import perform_parallelization
from src.business_logic.utilities.os_utilities import get_file_names_into_zip, unzip_folder
from src.data_model.enums import CountryName
from src.__constants import GUDID_ZIP_FILE_NAME, GUDID_FOLDER_NAME


def _find_nested_element(
        element: ElementTree.Element,
        tag_paths: str | list[str],
) -> ElementTree.Element | None:
    """
    This method safely finds a nested element in an XML tree, given the path to the element.

    :param element: The root element from which to start the search.
    :param tag_paths: The path to the element to be found.

    :return: The found element, if any, otherwise None.
    """
    child = element
    for tag in tag_paths:
        child = child.find(tag)
        if child is None:
            break

    return child


def _safe_text_access(
        element: ElementTree.Element
) -> str:
    """
    This method safely accesses the text of an element, if it is not None.

    :param element: The element from which to extract the text.

    :return: The text of the element, if it is not None, otherwise an empty string.
    """
    if element is not None and element.text is not None:
        return element.text
    else:
        return ''


def _extract_ordering_from_file_name(
        file_path: str
) -> int:
    """
    This method extracts the ordering numbers contained in the names of the FDA XML files. These are used to sort the
    files in the correct order.

    :param file_path: The path of the file from which to extract the ordering number.

    :return: The ordering number extracted from the file name.
    """
    splits = file_path.split('_')
    for split in splits:
        if split.__contains__('Part'):
            # Remove the 'Part' string
            split = split.replace('Part', '')
            return int(split)

    # Return a random number
    return random.randint(0, 1000)


def _parse_american_xml_file(
        zip_file_members: list[str],
        source_directory_path: str,
        usa_identifier: int,
) -> str:
    """
    This method extracts from the main zip file a list of files and parses them to retrieve all the medical devices'
    data that they contain. The retrieved data are stored in proper rows that are returned at the end of the
    parsing to the caller.

    :param zip_file_members: A list of the XML files' paths which contain data about medical devices sold in the
    American market.
    :param source_directory_path: The directory where all the sources of data are stored.
    :param usa_identifier: The identifier of the USA entity in the database.

    :return: A string representing the JSON representation of the dataframe containing the extracted data.
    """
    # Sort the zip file members
    zip_file_members.sort()
    # Get the first and last file names
    first_file_name = zip_file_members[0]
    first_value = _extract_ordering_from_file_name(first_file_name)
    last_file_name = zip_file_members[-1]
    last_value = _extract_ordering_from_file_name(last_file_name)
    folder_suffix = f"{first_value}_{last_value}"

    # Build the path for the zip file containing the GUDID data
    zip_file_path = source_directory_path + GUDID_ZIP_FILE_NAME
    # Verify that the zip file exists
    if not os.path.exists(zip_file_path):
        raise Exception(f"The file {zip_file_path} does not exist.")
    else:
        # Prepare the columns that will be used to store the data in a dataframe
        american_dataframe_columns = AMERICAN_DATAFRAME_COLUMNS
        # Get an empty list of rows to be filled with the data extracted from the XML files
        american_dataframe_rows = get_american_empty_rows()

        # Build the path to the folder where the files must be extracted
        zip_output_folder = source_directory_path + GUDID_FOLDER_NAME + '/' + folder_suffix + '/'

        # Unzip the given to the output folder
        unzip_folder(
            destination_folder_path=zip_output_folder,
            zip_file_path=zip_file_path,
            zip_member_paths=zip_file_members,
        )
        # Extract the file names from the list of members
        file_paths = [
            zip_output_folder + file_name.split('/')[-1]
            for file_name in zip_file_members
        ]

        try:
            # For each XML file to be parsed
            for xml_file_path in file_paths:
                try:
                    # Load the file containing a subset of the GUDID database
                    tree = ElementTree.parse(xml_file_path)
                    # Get the root of the XML tree
                    root = tree.getroot()

                    # NOTE: it is necessary to append the namespace before the tag, enclosed between {} brackets
                    # Get all the elements containing data related to medical devices
                    children = list(root.findall('{http://www.fda.gov/cdrh/gudid}device'))
                    # Iterate over the medical device data
                    for device_element in children:
                        # Find the tag containing the name of the device name
                        element = _find_nested_element(
                            device_element,
                            ['{http://www.fda.gov/cdrh/gudid}brandName'],
                        )
                        # Set the found original medical device name
                        original_device_name: str = _safe_text_access(element)
                        # Clean the medical device name from extra blank spaces
                        medical_device_name = clean_extra_blank_spaces(
                            clean_string(original_device_name)
                        ).upper()

                        # Find the tag containing the catalogue code
                        element = _find_nested_element(
                            device_element,
                            ['{http://www.fda.gov/cdrh/gudid}versionModelNumber'],
                        )
                        # Set the original medical catalogue code value
                        original_medical_catalogue_code: str = _safe_text_access(element)
                        # Clean the medical catalogue code
                        md_catalogue_code = clean_md_catalogue_code(
                            clean_string(original_medical_catalogue_code),
                            is_italian_code=False,
                        )

                        # Find the tag containing the company name
                        element = _find_nested_element(
                            device_element,
                            ['{http://www.fda.gov/cdrh/gudid}companyName'],
                        )
                        # Set the original company name
                        original_company_name: str = _safe_text_access(element)
                        # Clean the company name
                        company_name = clean_company_name(
                            original_company_name
                        )

                        # Try to find the GMDN Term Name
                        element = _find_nested_element(
                            device_element,
                            [
                                '{http://www.fda.gov/cdrh/gudid}gmdnTerms',
                                '{http://www.fda.gov/cdrh/gudid}gmdn',
                                '{http://www.fda.gov/cdrh/gudid}gmdnPTName'
                            ],
                        )
                        # Set the found GMDN term name
                        gmdn_term: str = clean_extra_blank_spaces(
                            _safe_text_access(element)
                        )
                        # Try to find the element containing the GMDN definition
                        element = _find_nested_element(
                            device_element,
                            [
                                '{http://www.fda.gov/cdrh/gudid}gmdnTerms',
                                '{http://www.fda.gov/cdrh/gudid}gmdn',
                                '{http://www.fda.gov/cdrh/gudid}gmdnPTDefinition'
                            ],
                        )
                        # Set the found GMDN definition
                        gmdn_definition: str = clean_extra_blank_spaces(
                            _safe_text_access(element)
                        )

                        # Find the element containing the FDA product code
                        element = _find_nested_element(
                            device_element,
                            [
                                '{http://www.fda.gov/cdrh/gudid}productCodes',
                                '{http://www.fda.gov/cdrh/gudid}fdaProductCode',
                                '{http://www.fda.gov/cdrh/gudid}productCode'
                            ],
                        )
                        # Set the value for the FDA product code
                        fda_product_code: str = clean_extra_blank_spaces(
                            _safe_text_access(element)
                        )

                        # If either original values of the company name, the medical device name or the catalogue code
                        # are missing, skip the row
                        if original_company_name == '' or original_device_name == '' or md_catalogue_code == '':
                            continue
                        else:
                            vales = [
                                original_company_name,
                                original_device_name,
                                md_catalogue_code,
                            ]
                            if None in vales or '' in vales:
                                raise Exception(f"Missing values: {vales}")

                            american_dataframe_rows.append(
                                (
                                    original_company_name,
                                    company_name,
                                    None,
                                    usa_identifier,
                                    None,
                                    original_device_name,
                                    medical_device_name,
                                    None,
                                    md_catalogue_code,
                                    'MD',
                                    None,
                                    fda_product_code,
                                    None,
                                    None,
                                    gmdn_term,
                                    gmdn_definition,
                                )
                            )

                    # Delete the file from which the data have been extracted
                    os.remove(xml_file_path)
                except xml.etree.ElementTree.ParseError:
                    # If an error occurred while parsing the XML file, print the error and continue with the next file
                    continue

            # Create a dataframe from the rows representing the extracted data
            american_dataframe = pd.DataFrame(
                data=american_dataframe_rows,
                columns=american_dataframe_columns,
            )

            # Remove the output folder
            shutil.rmtree(zip_output_folder)

            # Return the string of the JSON representation of the dataframe
            return json.dumps(american_dataframe.to_json())
        except Exception as e:
            print(f"Error while parsing the XML files: {e}")
            # Remove the output folder
            shutil.rmtree(zip_output_folder)


def load_american_medical_device_data(
        _source_directory_path: str,
        database_information: DatabaseInformation,
) -> None:
    """
    This method performs the pipeline, if it was not done yet, to process the XML files contained in the dataset
    provided by the FDA. This consists of loading all the files, parsing and extracting useful information and
    storing it in proper data structures. All the extracted data are then stored in the database.

    :param _source_directory_path: The directory where all the sources of data are stored.
    :param database_information: The information needed to connect to the database.
    """
    # Build the connection handler to connect to the database
    connection_handler = ConnectionHandler(
        database_information
    )
    # Instantiate the handler to add the USA as a country, if not already added
    country_data_handler = CountryDataHandler(
        connection_handler=connection_handler,
        init_cache=True,
    )
    usa_entity = country_data_handler.add_country(
        country_name=CountryName.USA.value,
        is_american_state=False,
    )

    # Build the path to the zip output folder and the zip file
    zip_file_path = _source_directory_path + GUDID_ZIP_FILE_NAME
    zip_output_folder = _source_directory_path + GUDID_FOLDER_NAME + '/'
    # Delete, if any, the output folder for the zip content
    if os.path.exists(zip_output_folder):
        shutil.rmtree(zip_output_folder)
        # Recreate the folder
        os.mkdir(zip_output_folder)

    # Verify that the input zip file is present
    if os.path.exists(zip_file_path):
        # Get a list of all the files contained into it
        zip_file_members = [
            file_name
            for file_name in get_file_names_into_zip(
                zip_file_path=zip_file_path
            )
            if not file_name.__contains__('.zip')
        ]

        cleaned_dataframe_strings: List[str] = perform_parallelization(
            input_sized_element=zip_file_members,
            processing_batch_callback=_parse_american_xml_file,
            additional_batch_callback_parameters=[
                _source_directory_path,
                usa_entity.identifier,
            ],
        )

        # Store all the american data in the database at once
        store_batched_american_data(
            dataframe_strings=cleaned_dataframe_strings,
            connection_handler=connection_handler,
        )

    if os.path.exists(zip_output_folder):
        # Remove the output folder
        shutil.rmtree(zip_output_folder)
