import numpy as np

MU = '\u03BC'
METRIC_CONVERSION_FACTORS = {
    'q':-30, 'r':-27, 'y':-24, 'z':-21, 'a':-18, 'f':-15,
    'p':-12, 'n': -9,  MU: -6, 'm': -3, 'c': -2, 'd': -1,
    '': 0,
    'T': 12, 'G':  9, 'M':  6, 'k':  3, 'h':  2, 'da': 1,
    'Q': 30, 'R': 27, 'Y': 24, 'Z': 21, 'E': 18, 'P': 15
}
AVAILABLE_HEAT_KEYS = [
    'ech', 'nb1a', 'nb1b', 'nb2a', 'nb2b','nb3a', 'nb3b',
    'nb4ua', 'nb4la', 'nb4ub', 'nb4lb',
    'nb5ua', 'nb5la', 'nb5ub', 'nb5lb'
]

def value_from_string(string: str) -> str:
    """
    Convert string to int, float, or retain string.

    Parameters
    ----------
    string : str
        Input string

    Returns
    -------
    val : int or float or str
        Integer, or float, or string.
    """
    if ',' in string:
        string_list = string.split(',')
        return [value_from_string(elem.strip()) for elem in string_list]
    else:
        try:
            return int(string)
        except ValueError:
            try:
                return float(string)
            except:
                return string
                
def error(string: str, stop: bool=False) -> None:
    """
    Print ANSI escape code formatted error string.

    Parameters
    ----------
    string : str
        String to print as error code using ANSI escape sequences.
    stop : bool, optional
        If `True` raises Exception instead of printing with ANSI escape
        sequence.

    Returns
    -------
    None
    """
    if stop:
        raise Exception(string)
    else:
        print(f'\033[0;31m{string}\033[0m')

def warn(string: str) -> None:
    """
    Print ANSI escape code formatted warning string.

    Parameters
    ----------
    string : str
        String to print as warning code using ANSI escape sequences.

    Returns
    -------
    None
    """
    print(f'\033[0;33m{string}\033[0m')

def success(string: str) -> None:
    """
    Print ANSI escape code formatted success string.

    Parameters
    ----------
    string : str
        String to print as warning code using ANSI escape sequences.

    Returns
    -------
    None
    """
    print(f'\033[0;32m{string}\033[0m')

def conversion_factor(old_unit: str, new_unit: str) -> float:
    """
    Return metric conversion factor to convert `old_unit` to `new_unit`.

    Parameters
    ----------
    old_unit : str
        Unit to convert from.
    new_unit : str
        Unit to convert to.

    Returns
    -------
    factor : float
        Conversion factor: `new_unit` = `old_unit` / `factor`
    """
    old_unit_set = set(old_unit)
    new_unit_set = set(new_unit)
    n_base_unit = len(old_unit_set.intersection(new_unit_set))
    same_base_unit = bool(n_base_unit)

    assert same_base_unit, \
    error(
        f'old_unit "{old_unit}" and new_unit "{new_unit}" '
        f'must use same base_unit')

    old_prefix = old_unit[:-n_base_unit]
    new_prefix = new_unit[:-n_base_unit]
    old_power = METRIC_CONVERSION_FACTORS[old_prefix]
    new_power = METRIC_CONVERSION_FACTORS[new_prefix]
    return 10 ** (old_power - new_power)

def check_increasing_index(sequence: list):
    """
    Check if sequence is increasing monotonically at a uniform rate.

    Parameters
    ----------
    sequence : list, array-like
        1-D sequence of values

    Returns
    -------
    is_increasing : bool
        Is `True` if sequence is increasing monotonically with even
        spacing.
    """

    diff = np.diff(sequence)
    one_unique_diff = np.unique(diff).size == 1
    increasing_monotonically = all(diff > 0)
    return increasing_monotonically and one_unique_diff

def heat_keys_to_heat_index(heat_keys: list) -> int:
    """
    Convert list of NBI and ECH "on" keys to binary label

    Parameters
    ----------
    heat_keys : list, array-like

    Returns
    -------
    heat_idx : binary int
    """
    heat_keys = [key.lower() for key in heat_keys]
    
    error = False
    for key in heat_keys:
        if key not in AVAILABLE_HEAT_KEYS:
            error(f'Unknown key: {key}')
            error = True
    if error:
        raise ValueError(f'Unknown keys in input')
            
    index_binary_list = [0]*len(AVAILABLE_HEAT_KEYS)
    for ikey, key in enumerate(AVAILABLE_HEAT_KEYS):
        index_binary_list[ikey] = int(key in heat_keys)
    index_binary_str = ''.join(map(str, index_binary_list))
    return int(index_binary_str, 2)

def heat_index_to_heat_keys(heat_idx: int) -> list:
    """
    Convert heat index binary label to list of NBI and ECH "on" keys

    Parameters
    ----------
    heat_idx : binary int

    Returns
    -------
    heat_keys : list
    """
    index_binary_str = f'{int(bin(heat_idx)[2:]):011}'
    return [
        key for ison, key in zip(
            np.array(list(index_binary_str), dtype=int),
            AVAILABLE_HEAT_KEYS
        ) if ison]