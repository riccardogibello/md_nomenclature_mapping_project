def concatenate_emdn_gmdn_identifiers(
        emdn_id: int,
        gmdn_id: int,
) -> str:
    """
    This method, given two strings, returns a new string made of the concatenation of the two, comma-separated.

    :param emdn_id: The first string to be used.
    :param gmdn_id: The second string to be used.

    :return: A new string made of the concatenation of the two given ones, comma-separated.
    """
    return ','.join(
        [str(emdn_id), str(gmdn_id)]
    )
