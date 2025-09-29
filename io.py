import numpy as np
import pandas as pd

from .utils import conversion_factor, value_from_string

def read_data_header(file: str, comment: str='#') -> dict:
    """
    Read the header stored in a text data file when the header contains
    key-value pairs. Assumes key-value pairs separated by '='.

    Parameters
    ----------
    file : str
        Filepath for data file.
    comment : str, optional
        String that starts header lines in `file`. Default is '#'.

    Returns
    -------
    header : dict
        Dictionary containing key-value pairs.
    """
    header = {}
    with open(file, 'r') as file:
        for line in file:
            if line[0] != comment:
                break
            elif (
                (len(line[1:].strip().replace("'", '')) == 0) or 
                ('=' not in line)
            ):
                continue
            else:
                line = line[1:].strip().replace("'", '').split('=')
                line = [elem.strip() for elem in line]
                key = line[0]
                vals = value_from_string(line[1])
                header[key] = vals
    return header
    
def read_data_from_dat(
    file: str,
    index_names: list,
    column_names: list,
    index_units: list,
    column_units: list,
    comment: str='#',
    use_postgres_names: bool=False,
    convert_to_ms: bool=False,
    usecols: list=None
) -> pd.DataFrame:
    """
    Read data from .dat file.

    Parameters
    ----------
    file : str
        Filepath for .dat file.
    index_names : list
        List of names to use for index labels.
    column_names : list
        List of names to use for column labels.
    index_units : list
        List of units for index labels.
    column_units : list
        List of units for column labels.
    comment : str, optional
        String that starts comment lines in `file`. Comment lines are
        ignored. Default is '#'.
    use_postgres_names : bool, optional
        If `true`, converts all index and column labels to lower snake
        case. Default is `false`.
    convert_to_ms : bool, optional
        If `true` and if 'time' is in `index_names`, converts 'time'
        index to integer-represented milliseconds. Default is `false`.
    usecols : list
        See pandas.read_csv for details.

    Returns
    -------
    data : pd.DataFrame
        Pandas dataframe containing the data.
    """

    if use_postgres_names:
        index_names = [
            name.lower().replace('-', '_').replace(' ', '_') for
            name in index_names]
        column_names = [
            name.lower().replace('-', '_').replace(' ', '_') for
            name in column_names]
    names = index_names + column_names
    units = index_units + column_units
    if usecols:
        names = (np.array(names)[usecols]).tolist()
        units = (np.array(units)[usecols]).tolist()
    units_dict = {name:unit for name, unit in zip(names, units)}
    data = pd.read_csv(
        file, comment=comment, names=names, usecols=usecols)

    if convert_to_ms and 'time' in index_names:
        factor = conversion_factor(units_dict['time'], 'ms')
        data['time'] = round(data['time']*factor).astype(int)
        units_dict['time'] = 'ms'
    
    data.attrs['units'] = units_dict
    return data.set_index(index_names)

def read_data_basic(
    file: str,
    use_postgres_names: bool=False,
    convert_to_ms: bool=False
) -> pd.DataFrame:
    """
    Read data from .dat file as a dataframe with headers.

    Parameters
    ----------
    file: str
        Filepath for .dat file.
    use_postgres_names : bool, optional
        If `true`, converts all index and column labels to lower snake
        case. Default is `false`.
    convert_to_ms : bool, optional
        If `true` and if 'time' is in `index_names`, converts 'time'
        index to integer-represented milliseconds. Default is `false`.

    Returns
    -------
    data : pd.DataFrame
        Pandas dataframe containing the data with headers.
    """
    
    header = read_data_header(file)
    dim_names = header['DimName']
    dim_units = header['DimUnit']
    if isinstance(dim_names, str):
        dim_names = [dim_names]
        dim_units = [dim_units]
    val_names = header['ValName']
    val_units = header['ValUnit']

    n_dims = len(dim_names)
    usecols = [i for i in range(n_dims)] + \
    [i+n_dims for i, key in enumerate(val_names) if 
         ('?' not in key) and ('none' not in key)]
    
    data = read_data_from_dat(
        file,
        dim_names, val_names,
        dim_units, val_units,
        use_postgres_names=use_postgres_names,
        convert_to_ms=convert_to_ms,
        usecols=usecols)
    return data
    
def read_data_ech(
    ech_file: str,
    use_postgres_names: bool=False,
    convert_to_ms: bool=False
) -> pd.DataFrame:
    """
    Read ECH data from .dat file as a dataframe with headers.

    Parameters
    ----------
    file: str
        Filepath for ECH .dat file.
    use_postgres_names : bool, optional
        If `true`, converts all index and column labels to lower snake
        case. Default is `false`.
    convert_to_ms : bool, optional
        If `true` and if 'time' is in `index_names`, converts 'time'
        index to integer-represented milliseconds. Default is `false`.

    Returns
    -------
    data : pd.DataFrame
        Pandas dataframe containing ECH data with headers.
    """
    ech_data = read_data_basic(
        ech_file, use_postgres_names=use_postgres_names)

    if convert_to_ms:
        ech_data.index = (ech_data.index.values*1e3).round(1)
        ech_data = ech_data.loc[
            np.arange(ech_data.index.min(), ech_data.index.max()).round(0)]
        ech_data.index = ech_data.index.astype(int)
    return ech_data

def read_nbists_from_files(
    files: list,
    use_postgres_names: bool=False,
    convert_to_ms: bool=False
) -> pd.Series:
    """
    Read NBI on-off data from .dat files as a series.

    Parameters
    ----------
    files: list
        Filepaths for NBI .dat files.
    use_postgres_names : bool, optional
        If `true`, converts all index and column labels to lower snake
        case. Default is `false`.
    convert_to_ms : bool, optional
        If `true` and if 'time' is in `index_names`, converts 'time'
        index to integer-represented milliseconds. Default is `false`.

    Returns
    -------
    data : pd.Series
        Pandas series containing NBI on-off data. 1: on, 0: off.
    """
    
    if isinstance(files, str): files = [files]
    ists_ser_list = []
    for file in files:
        nbi_data = read_data_basic(
            file,
            use_postgres_names=use_postgres_names,
            convert_to_ms=convert_to_ms)
        ists_ser_single = nbi_data.filter(
            regex=r'ion_sts(a|b|u|l)_nb\d(a|b)*', axis=1).max(1).astype(int)
        ists_ser_list.append(ists_ser_single)
    ists_ser = pd.concat(ists_ser_list, axis=1).max(1)
    return ists_ser