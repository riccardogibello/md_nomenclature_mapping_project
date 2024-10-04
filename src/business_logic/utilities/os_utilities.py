import os
from typing import Optional, Any
from zipfile import ZipFile
import torch


def convert_to_device(
        element: Any,
        device_name: str
) -> Any:
    """
    This method moves the given element to the given device_name, if possible.

    :param element: The element to be moved to the given device_name.
    :param device_name: The device_name to which the element must be moved.

    :return: The element moved to the given device_name.
    """
    if not hasattr(element, 'to'):
        raise TypeError("The element does not support the 'to' method for device conversion.")

    is_cuda_available = torch.cuda.is_available()

    if device_name == 'cuda' and is_cuda_available:
        try:
            current_device = element.device.type
        except AttributeError:
            current_device = 'cpu'

        if current_device == 'cuda':
            return element
        else:
            return element.to('cuda')
    elif device_name == 'cpu':
        return element.to('cpu')
    else:
        raise ValueError(f"Unsupported device_name: {device_name}. Supported values are 'cuda' and 'cpu'.")


def _format_path(
        given_path: str,
        is_file_path: Optional[bool] = True,
) -> str:
    """
    This method formats any given_path to a slash-separated one.

    :param given_path: The full path to be formatted.
    :param is_file_path: Indicates whether a trailing '/' must be added, if not already present.

    :return: The formatted path.
    """
    given_path = given_path.replace(os.path.sep, '/')
    if os.path.altsep:
        given_path = given_path.replace(os.path.altsep, '/')

    if not is_file_path and not given_path.endswith('/'):
        given_path += '/'

    return given_path


def create_directory(
        directory_path: str,
) -> None:
    """
    This method, given a full directory_path, creates the full path of directories if it does not already
    exist.

    :param directory_path: The full path of the directory to be created.
    """
    directory_path = _format_path(
        directory_path,
        is_file_path=False,
    )

    split = directory_path.split('/')
    max_value: Any = len(split)
    for i in range(max_value):
        # take a sublist with length i of the path to the directory
        sub_list = split[:i + 1]
        sub_path = '/'.join(sub_list)
        if sub_path != '' and not os.path.isdir(sub_path):
            os.makedirs(sub_path)


def get_file_names_into_zip(
        zip_file_path: str,
) -> list[str]:
    """
    This method returns the names of all the files in the given zip_file_path.

    :param zip_file_path: The path to the zip file.

    :return: The list of all the files in the given zip file.
    """
    if os.path.isfile(zip_file_path):
        zip_file = ZipFile(
            zip_file_path,
        )
        return zip_file.namelist()
    else:
        return []


def unzip_folder(
        destination_folder_path: str,
        zip_file_path: str,
        zip_member_paths: Optional[list[str]] = None,
) -> None:
    """
    This method unzips the given zip folder, referenced by zip_folder_path, in the given destination_folder_path.

    :param destination_folder_path: The destination where the folder must be unzipped.
    :param zip_file_path: The folder to be unzipped.
    :param zip_member_paths: All the paths in the zip file that must be unzipped.
    """
    if os.path.isfile(zip_file_path):
        zip_file = ZipFile(
            zip_file_path,
        )
        zip_file.extractall(
            path=destination_folder_path,
            members=zip_member_paths,
            pwd=None
        )


def find_files_containing_string(
        base_folder_path: str,
        search_substring: str
) -> list[str]:
    """
    This method finds all the files containing the searched substring in any subfolder of the given base folder path.

    :param base_folder_path: The initial folder path to start the search.
    :param search_substring: The substring to be searched in the file names.

    :return: The list of all the file paths containing the searched substring.
    """
    matching_files = []
    for dir_path, dir_names, filenames in os.walk(base_folder_path):
        for filename in filenames:
            if search_substring in filename:
                full_path = _format_path(os.path.join(dir_path, filename))
                matching_files.append(full_path)
        for dir_name in dir_names:
            matching_files.extend(
                find_files_containing_string(
                    _format_path(os.path.join(dir_path, dir_name)),
                    search_substring
                )
            )

    return list(set(matching_files))
