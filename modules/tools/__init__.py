from typing import Any


def dictLookup(input: dict, search_param: Any) -> Any:
    """
    Searches `input` for first occurence of search_param.\n
    This search include both keys and values, starting with values.

    Note
    ----
    This search is NOT recursive!

    Parameters
    ----------
    input : dict
        The dict to search in.

    search_param : Any
        Key or value to search for.

    Returns
    -------
    Any
        - Key if searchParam == Value
        - Value if searchParam == Key
        - None if searchParam wasn't found in the input
    """
    if isinstance(input, dict):
        value = input.get(search_param, None)  # searchParam is a key mapping to value
        if not value:
            for k, v in input.items():  # searchParam is a value mapping to key
                if v == search_param:
                    return k
        return value
