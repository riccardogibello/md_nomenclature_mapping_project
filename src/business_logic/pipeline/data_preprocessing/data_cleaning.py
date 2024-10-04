import html
import re
import string
from typing import Optional


def clean_md_catalogue_code(
        medical_device_code: str,
        is_italian_code: bool = True,
) -> str:
    """
    This method standardizes both American and Italian catalogue codes of medical devices.

    :param is_italian_code: Indicates whether to apply some cleaning rules specific to the Italian codes.
    :param medical_device_code: The code to be standardized.

    :return: The standardized code.
    """
    # clean the string in case of None or other malformed types
    medical_device_code = clean_string(medical_device_code)

    # remove DA(L) - A(L) pattern
    medical_device_code = re.sub(pattern=r"[" + string.punctuation + " ]*DA[ ]", string=medical_device_code, repl=',')
    medical_device_code = re.sub(pattern=r"[" + string.punctuation + " ]*DAL[ ]", string=medical_device_code, repl=',')
    medical_device_code = re.sub(pattern=r"[" + string.punctuation + " ]+A[ ]", string=medical_device_code, repl=',')
    medical_device_code = re.sub(pattern=r"[" + string.punctuation + " ]+AL[ ]", string=medical_device_code, repl=',')

    # replace the '-', '/' and ';' separators with a comma
    medical_device_code = medical_device_code.replace(' - ', ',')
    medical_device_code = medical_device_code.replace('/', ',')
    medical_device_code = medical_device_code.replace(';', ',')

    # remove every blank space from the src
    medical_device_code = re.sub(pattern=r"[ ]+", string=medical_device_code, repl='')

    # transform the src in capital letters
    medical_device_code = medical_device_code.upper()

    if is_italian_code:
        # replace every X (or x) holder (even repeated) in the src with a single asterisk character
        medical_device_code = re.sub(pattern=r"[X]+", string=medical_device_code, repl='*')

        # replace possible subsequent asterisk characters with a single one
        medical_device_code = re.sub(pattern=r"[*]+", string=medical_device_code, repl='*')

    # remove any character between () and []
    medical_device_code = re.sub(pattern=r"[(].*[)]", string=medical_device_code, repl='')
    medical_device_code = re.sub(pattern=r"[\[].*[\]]", string=medical_device_code, repl='')

    # remove any parenthesis left in the src
    medical_device_code = medical_device_code.replace('(', '')
    medical_device_code = medical_device_code.replace(')', '')
    medical_device_code = medical_device_code.replace('[', '')
    medical_device_code = medical_device_code.replace(']', '')
    # Remove any escape character
    medical_device_code = medical_device_code.replace('\\', '')

    return clean_extra_blank_spaces(medical_device_code)


def clean_company_name(
        company_name: str
) -> str:
    """
    This method cleans the company_name by removing any special character, parenthesis, and any pattern enclosed between
    "&#" and ";". It also removes any pattern enclosed between "(" and ")". It splits the string based on the ":"
    character and keeps the second split if it is not empty, otherwise, it keeps the first split. Finally, it removes
    any exceeding blank spaces.

    :param company_name: The company name to be cleaned.

    :return: The cleaned company name.
    """
    if company_name is None:
        return ''
    lowered_name = html.unescape(company_name)
    # + ( ) / to be left in the string
    lowered_name = lowered_name.upper().encode('ascii', 'ignore').decode('ascii')
    # Remove any match to double quote ("), period (.), comma (,), backslash (), single quote ('), hyphen (-),
    # asterisk (*), question mark (?), and at symbol (@)
    lowered_name = re.sub(r'[".,\\\'\-*?@]', ' ', lowered_name)
    # Remove any pattern enclosed between "&#" and ";"
    lowered_name = re.sub(r'#\d+;', '', lowered_name)

    # Remove any pattern enclosed between "(" and ")"
    lowered_name = re.sub(r'\([^)]*\)', '', lowered_name)
    lowered_name = re.sub(r'\[[^)]*]', '', lowered_name)
    # Remove any type of parenthesis char
    lowered_name = re.sub(r'[()\[\]{}]', '', lowered_name)
    lowered_name = re.sub(r'[&+/;!]', ' ', lowered_name)

    # Remove anything that comes after INC, CORP, LTD, SRL, SPA, SNC, SAS, SA, GMBH, AG, KG, BV, NV, SL, CV, CO,
    # LLP, PLC, LLC
    lowered_name = re.sub(
        r'\b(?:inc|a/s|s/a|m/s|c/o|corp|ltd|srl|spa|snc|sas|sa|gmbh|ag|kg|bv|nv|sl|cv|co|llp|plc|llc)\b.*',
        '',
        lowered_name
    )

    # Split the string based on the ":" character
    splits = lowered_name.split(":")
    # If the second split is empty
    if len(splits) > 1:
        if splits[1] == '':
            # Keep the first split
            lowered_name = splits[0]
        else:
            # Keep the second split
            lowered_name = splits[1]
    else:
        lowered_name = splits[0]

    # Remove any exceeding blank spaces
    return clean_extra_blank_spaces(lowered_name)


def clean_string(
        string_to_be_cleaned: Optional[str] = None,
        encoding: Optional[str] = 'utf-8',
) -> str:
    """
    This method cleans a given string_to_be_cleaned by setting it to the empty string if the given one is None or
    does not correspond to a str or int type.

    Then, it encodes and decodes the string in the given encoding.

    :param string_to_be_cleaned: The string that must be cleaned.
    :param encoding: The encoding used for the cleaned string.

    :return: The cleaned string.
    """
    if string_to_be_cleaned is None or not (type(string_to_be_cleaned) is str or type(string_to_be_cleaned) is int):
        return ''
    else:
        if type(string_to_be_cleaned) is int:
            string_to_be_cleaned = str(string_to_be_cleaned)

        return string_to_be_cleaned.encode(encoding, errors='ignore').decode(encoding)


def clean_extra_blank_spaces(
        given_string: str
) -> str:
    """
    This method removes from the given_string any extra blank character, both internal, leading or trailing.

    :param given_string: The string to be cleaned.

    :return: The given_string cleaned from any extra blank character, both internal, leading or trailing.
    """
    # if the string is None, then it is replaced with a blank character
    if given_string is None:
        return ''
    else:
        # clean from potential internal spaces
        cleaned_string = re.sub(pattern=r"[ ]+", string=given_string, repl=' ')
        # remove leading and trailing blank spaces
        return cleaned_string.strip()
