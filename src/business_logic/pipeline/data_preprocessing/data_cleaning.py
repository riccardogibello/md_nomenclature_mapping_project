import html
import re
import string
from typing import Optional


def clean_md_catalogue_code(
        medical_device_code: str,
        is_italian_code: bool = True,
) -> list[str]:
    """
    This method standardizes both American and Italian catalogue codes of medical devices.

    :param is_italian_code: Indicates whether to apply some cleaning rules specific to the Italian codes.
    :param medical_device_code: The code to be standardized.

    :return: A list containing all the codes extracted from the given medical device code.
    """
    # If the symbol contains an & symbol, return an empty string
    if medical_device_code.__contains__('&'):
        return []
    else:
        # Use the following symbols as separators: ' - ', ';'
        # Don't use these:
        #   '-' (there are cases in which the catalogue code contains a hyphen, e.g. 123-456)
        #   '/' (there are cases in which the catalogue code contains a slash, e.g. 123/456)
        medical_device_code = re.sub(
            pattern=r" - |;",
            string=medical_device_code,
            repl=','
        )
        # If the code is italian
        if is_italian_code:
            patterns = [
                r"[" + string.punctuation + " ]*DA[ ]",
                r"[" + string.punctuation + " ]*DAL[ ]",
                r"[" + string.punctuation + " ]+A[ ]",
                r"[" + string.punctuation + " ]+AL[ ]",
            ]
            for pattern in patterns:
                medical_device_code = re.sub(
                    pattern=pattern,
                    string=medical_device_code,
                    repl=','
                )
        # If the code contains a comma
        if medical_device_code.__contains__(','):
            # Split the code into multiple codes using the comma as a separator and perform the cleaning
            # on each of them
            codes = medical_device_code.split(',')
            cleaned_codes = []
            for code in codes:
                cleaned_code = clean_md_catalogue_code(code, is_italian_code)
                if len(cleaned_code) > 0:
                    cleaned_codes.extend(cleaned_code)
            return cleaned_codes
        else:
            # Remove:
            patterns = [
                # Any character between () and []
                r"\[.*?\]|\(.*?\)",
                # Any unpaired round or square bracket,
                # any escape character,
                # any " and ' symbol,
                # any # symbol
                # any . symbol
                # any - symbol
                # any of "–=°Ⅱ@_™" symbols
                r"[\(\)\[\]\\]|['\"#.]|[\–\=°Ⅱ@_™]",
            ]
            for pattern in patterns:
                medical_device_code = re.sub(
                    pattern=pattern,
                    string=medical_device_code,
                    repl=''
                )

            # Replace every X (or x) holder (even repeated) with a single asterisk character
            patterns = [
                r"[X]+",
                r"[*]+"
            ]
            for pattern in patterns:
                medical_device_code = re.sub(
                    pattern=pattern,
                    string=medical_device_code,
                    repl='*'
                )

            returned_code = clean_extra_blank_spaces(medical_device_code)
            # Get all the unique characters in the code
            unique_characters = set(returned_code)
            # If the unique characters are blank spaces, commas, asterisks and/or -
            if unique_characters.issubset({' ', ',', '*', '-'}):
                return []
            else:
                return [returned_code]


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

        string_to_be_cleaned = string_to_be_cleaned.encode(
            encoding,
            errors='ignore'
        ).decode(encoding)

        string_to_be_cleaned = string_to_be_cleaned.upper()

        return string_to_be_cleaned


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
