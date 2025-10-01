import configparser
import numpy as np
import os
import pandas as pd

from .utils import conversion_factor, value_from_string, warn, error

PACKAGE_DIR = os.path.dirname(__file__)
DEFAULT_CONFIG_FILE = PACKAGE_DIR + '/config.ini'
DEFAULT_FIBERMAPS_FILE = PACKAGE_DIR + '/fibermaps.ini'
DEFAULT_FIBERMAPS_LOG_FILE = PACKAGE_DIR + '/fibermaps.log'
DEFAULT_GEOMETRY_FILE = PACKAGE_DIR + '/geometry.ini'

def read_config(config_file: str=None) -> dict:
    """
    Read in configuration from config file.

    Parameters
    ----------
    config_file : str, optional
        Configuration file path. Default is config.ini in package
        directory.

    Returns
    -------
    config : dict
        Dictionary containing configuration.
    """
    if config_file is None: config_file = DEFAULT_CONFIG_FILE
    config = {}
    config_obj = configparser.ConfigParser()
    config_obj.read(config_file)
    for key in ['data_dir', 'pqt_dir']:
        config[key] = os.path.abspath(config_obj['data_directories'][key])

    config['nbi_patts'] = {
        key:config_obj['data_file_patterns'][f'{key}_file']
        for key in ['nb1', 'nb2', 'nb3', 'nb4a', 'nb4b', 'nb5a', 'nb5b']}

    config['nnbi_patts'] = {
        f'nb{i}':config['nbi_patts'][f'nb{i}'] for i in [1, 2, 3]}
    config['pnbi_patts'] = {
        f'nb{i}{j}':config['nbi_patts'][f'nb{i}{j}'] for
        i in [4, 5] for j in ['a', 'b']}

    config['ech_patt'] = config_obj['data_file_patterns']['ech_file']
    config['ficxs_pqt_patt'] = \
        config_obj['data_file_patterns']['ficxs_pqt_file']

    for los in [6, 7]:
        config[f'{los}o_los_dnb_file_patterns'] = {
            key:patt for key, patt in 
            config_obj[f'{los}o_los_dnb_file_patterns'].items() }
    
    config['save_dir'] = os.path.abspath(
        config_obj['save_file_settings']['save_dir'] )
    config['save_patt'] = config_obj['save_file_settings']['save_file']


    config['png_dir'] = os.path.abspath(
        config_obj['save_png_settings']['png_dir'] )
    config['png_patt'] = config_obj['save_png_settings']['png_file']
    
    return config

def read_fibermaps(
    fibermaps_file: str=None,
    fibermaps: list=None
) -> pd.DataFrame:
    """
    Read in FICXS fiber configurations from fibermaps file. Default fibermaps
    file lists radial position in meters.

    Parameters
    ----------
    fibermaps_file : str, optional
        FICXS fiber configurations file path. Default is fibermaps.ini
        in package directory.
    fibermaps : list, optional
        List of fibermaps to read in configrations. If `None`, reads in
        all configurations in `fibermaps_file`.
        
    Returns
    -------
    fibers : pd.DataFrame
        Pandas dataframe containing fibermap configurations.
    """
    if fibermaps_file is None: fibermaps_file = DEFAULT_FIBERMAPS_FILE
    config_obj = configparser.ConfigParser()
    config_obj.read(fibermaps_file)
    if fibermaps is None: fibermaps = config_obj.sections()
    for fmap in fibermaps:
        if fmap not in ['6OFIDA', 'PNBFIDA', 'NNBFIDA']:
            error(
                f'Unknown fibermap "{fmap}". '
                f'Available fibermaps are '
                f'["6OFIDA", "PNBFIDA", "NNBFIDA"]',
                stop=True)

    n_fmaps = len(fibermaps)
    fibers = pd.DataFrame(
        np.zeros((16, n_fmaps*2)),
        index=np.arange(1, 17),
        columns=pd.MultiIndex.from_tuples([
            (fmap, label) for fmap in fibermaps for label in ['r', 'los']
        ]))
    fibers.index.name = 'ch'
    
    for fmap in fibermaps:
        fibers[(fmap, 'los')] = fibers[(fmap, 'los')].astype(int)
        for key, val in config_obj[fmap].items():
            ich = int(key)
            r, port = (elem.strip() for elem in val.split(','))
            r = float(r)
            port = int(port)
            fibers.loc[ich, (fmap, 'r')] = r
            fibers.loc[ich, (fmap, 'los')] = port
            
    return fibers

def read_fibermaps_log(log_file: str=None) -> dict:
    """
    Read shot_number:fibermap key-value pairs from log file.

    Parameters
    ----------
    log_file : str
        Fibermaps log file. Default is fibermaps.log in package
        directory.

    Returns
    -------
    shots_fibermaps : dict
        Dictionary containing shot_number:fibermaps key-value pairs.
    """
    if log_file is None: log_file = DEFAULT_FIBERMAPS_LOG_FILE
    if os.path.exists(log_file):
        fmaps_ser = pd.read_csv(
            log_file, names=['shot', 'fmap']).set_index('shot')['fmap']
        shots_fibermaps = {
            int(shot):fmap for shot, fmap in fmaps_ser.items()}
    else:
        warn(
            f'fibermaps log file [{log_file}] does not exist. '
            f'Creating new log file.')
        with open(log_file, 'w') as file:
            file.write('\n')
        shots_fibermaps = {}
    return shots_fibermaps

def write_fibermaps_log(
    new_shots_fibermaps : dict,
    log_file: str=None,
    overwrite: bool=False
) -> None:
    """
    Write fibermap configuration label to log file.

    Parameters
    ----------
    new_shots_fibermaps : dict
        Dictionary containing shot_number:fibermaps key-value pairs.
    log_file : str, optional
        Fibermaps log file. Default is fibermaps.log in package
        directory.
    overwrite : bool, optional
        If `True` and shot already listed in log file, replaces listed
        fibermap with new fibermap.
    """
    ierr = 0
    for shot, fmap in new_shots_fibermaps.items():
        if fmap not in ['6OFIDA', 'PNBFIDA', 'NNBFIDA']:
            ierr = 1
            error(
                f'Unknown fibermap "{fmap}" for {shot}. '
                f'Available fibermaps are '
                f'["6OFIDA", "PNBFIDA", "NNBFIDA"]')
    if ierr: raise Exception(f'Error with input shots_fibermaps dictionary.')
    
    if log_file is None: log_file = DEFAULT_FIBERMAPS_LOG_FILE
    if os.path.exists(log_file):
        shots_fibermaps = read_fibermaps_log(log_file=log_file)
        for shot, fmap in new_shots_fibermaps.items():
            if shot in shots_fibermaps:
                old_fmap = shots_fibermaps[shot]
                warn(f'{shot} already in fibermaps log.')
                if overwrite:
                    warn(
                        f'Overwriting existing fibermap '
                        f'[{old_fmap}] with new fibermap [{fmap}]')
                    shots_fibermaps[shot] = fmap
                else:
                    warn(
                        f'Keeping original fibermap [{old_fmap}]. '
                        f'If you would like to overwrite the listed '
                        f'fibermap, set overwrite to True.')
            else:
                shots_fibermaps[shot] = fmap     
    else:
        warn(
            f'fibermaps log file [{log_file}] does not exist. '
            f'Creating new log file.')
    shots_fibermaps = dict(sorted(shots_fibermaps.items(), key=lambda x:x[0]))
    with open(log_file, 'w') as file:
        for shot, fmap in shots_fibermaps.items():
            file.write(f'{shot},{fmap}\n')

def read_geometry(
    geometry_label: str,
    geometry_file: str=None
) -> dict:
    """
    Read in FICXS LOS geometry or FICXS DNB geometry from geometry file.
    Default geometry file lists source position in meters.

    Parameters
    ----------
    geometry_label : {'nbi_geo', '6OFIDA', 'PNBFIDA', 'NNBFIDA'}
        Label of geometry to read from file.
    geometry_file : str, optional
        Geometry file path. Default is geometry.ini in package
        directory.
        
    Returns
    -------
    geo : dict
        Dictionary containing source and axis of LOS or DNB geometry.
    """
    if geometry_file is None: geometry_file = DEFAULT_GEOMETRY_FILE
    config_obj = configparser.ConfigParser()
    config_obj.read(geometry_file)

    geo = {}
    for key, vals in config_obj[geometry_label].items():
        if key.isnumeric(): key = int(key)
        src = np.array([float(elem.strip()) for elem in vals.split(',')[:3]])
        axis = np.array([float(elem.strip()) for elem in vals.split(',')[3:]])
        geo[key] = dict(src=src, axis=axis)
    return geo

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
        If `True`, converts all index and column labels to lower snake
        case. Default is `False`.
    convert_to_ms : bool, optional
        If `True` and if 'time' is in `index_names`, converts 'time'
        index to integer-represented milliseconds. Default is `False`.
    usecols : list, optional
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
        If `True`, converts all index and column labels to lower snake
        case. Default is `False`.
    convert_to_ms : bool, optional
        If `True` and if 'time' is in `index_names`, converts 'time'
        index to integer-represented milliseconds. Default is `False`.

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
        If `True`, converts all index and column labels to lower snake
        case. Default is `False`.
    convert_to_ms : bool, optional
        If `True` and if 'time' is in `index_names`, converts 'time'
        index to integer-represented milliseconds. Default is `False`.

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
        If `True`, converts all index and column labels to lower snake
        case. Default is `False`.
    convert_to_ms : bool, optional
        If `True` and if 'time' is in `index_names`, converts 'time'
        index to integer-represented milliseconds. Default is `False`.

    Returns
    -------
    ists_ser : pd.Series
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

def read_full_nbists_from_config(
    shot: int,
    config: dict=None
) -> pd.DataFrame:
    """
    Read in NBI on-off data for all beamlines and ports.

    Parameters
    ----------
    shot : int
        Shot number.
    config: dict, optional
        Dictionary containing program configuration.

    Returns
    -------
    ists_df : pd.DataFrame
        DataFrame containing NBI on-off data. 1: on, 0: off.
    """
    if config is None: config = read_config()
    data_dir = config['data_dir']
    ists_df_list = []
    for ifile, (key, patt) in enumerate(config['nbi_patts'].items()):
        file = data_dir + '/' + patt.format(shot=shot)
        nbi_data = read_data_basic(
            file, use_postgres_names=True, convert_to_ms=True)

        ports = ['a', 'b'] if key in ['nb1', 'nb2', 'nb3'] else ['u', 'l']
        for port in ports:
            name = key+port if ifile < 3 else key[:-1]+port+key[-1]
            ists_ser = nbi_data[f'ion_sts{port}_{key}'].astype(int)
            ists_ser.name = name
            ists_df_list.append(ists_ser)
    ists_df = pd.concat(ists_df_list, axis=1)
    return ists_df

def read_ficxs_from_pqt(file: str, convert_to_ms: bool=False) -> pd.DataFrame:
    """
    Read FICXS data from a parquet file.
    
    Parameters
    ----------
    file : str
        Filepath to FICXS parquet file.
    convert_to_ms : bool, optional
        If `True` and if 'time' is in `index_names`, converts 'time'
        index to integer-represented milliseconds. Default is `False`.

    Returns
    -------
    ficxs_pqt : pd.DataFrame
        Dataframe containing FICXS data read from `file`.
    """
    ficxs_pqt = pd.read_parquet(file)
    index_names = list(ficxs_pqt.index.names)
    if 'wavelength' in ficxs_pqt.columns:
        ficxs_pqt = ficxs_pqt.reset_index().set_index(
            index_names + ['wavelength'])
    if 'xpix' in index_names:
        ficxs_pqt.index = ficxs_pqt.index.droplevel('xpix')
    if 's1' in ficxs_pqt.columns:
        ficxs_pqt.columns = [int(col[1:]) for col in ficxs_pqt.columns]
    ficxs_pqt.columns.names = ['ch']

    if (
        convert_to_ms and
        (ficxs_t_unit := ficxs_pqt.attrs['units']['time']) != 'ms'
    ):
        factor = conversion_factor(ficxs_t_unit, 'ms')
        ficxs_pqt.index = pd.MultiIndex.from_tuples([
            (round(idx[0]*factor).astype(int), idx[1]) 
            for idx in ficxs_pqt.index])
        ficxs_pqt.attrs['units']['time'] = 'ms'

    return ficxs_pqt.T.sort_index().T.sort_index()
    
def read_ficxs_timing(file: str) -> pd.Series:
    """
    Read FICXS timing data from program-generated file.

    Parameters
    ----------
    file : str
        FICXS timing data file generated by lhd_ficxs_py.run.

    returns
    -------
    ficxs_ts_ser : pd.Series
        Pandas series containing FICXS timing data.
    """
    ficxs_ts_ser = pd.read_csv(
        file, comment='#').set_index(['t_ms_idx', 't_ms']).sort_index()['on']
    return ficxs_ts_ser

def write_ficxs_timing(ficxs_ts_ser: pd.Series, file: str) -> None:
    """
    Write FICXS timing data generate by lhd_ficxs_py.run to file.

    Parameters
    ----------
    ficxs_ts_ser : pd.Series
        Pandas series containing FICXS timing data.
    file : str
        Filepath of save file.
    """
    with open(file, 'w') as file:
        ut = ficxs_ts_ser.attrs['input_settings']['ficxs_upper_threshold']
        lt = ficxs_ts_ser.attrs['input_settings']['ficxs_lower_threshold']
        file.write(f'# [input_settings]\n')
        file.write(f'# ficxs_upper_threshold = {ut:.5f}\n')
        file.write(f'# ficxs_lower_threshold = {lt:.5f}\n')
        file.write(f't_ms_idx,t_ms,on\n')
        for (t_ms_idx, t_ms), on in ficxs_ts_ser.items():
            file.write(f'{t_ms_idx:.0f}, {t_ms:.0f}, {on:.0f}\n')

def convert_dat_to_pqt(
    file: str,
    use_postgres_names: bool=False,
    convert_to_ms: bool=False,
    compression: str='brotli',
    engine: str='pyarrow'
) -> None:
    """
    Convert a .dat file to a parquet file.

    Parameters
    ----------
    file : str
        Filename of .dat file. The parquet file will be save under the
        same name but with a .pqt extension.
    use_postgres_names : bool, optional
        If `True`, converts all index and column labels to lower snake
        case. Default is `False`.
    convert_to_ms : bool, optional
        If `True` and if 'time' is in `index_names`, converts 'time'
        index to integer-represented milliseconds. Default is `False`.
    compression : str, optional
        Input for pd.DataFrame.to_parquet. Default is 'brotli'.
    engine : str, optional
        Input for pd.DataFrame.to_parque. Default is 'pyarrow'.
    """
    data = read_data_basic(
        file,
        use_postgres_names=use_postgres_names, convert_to_ms=convert_to_ms)
    data.to_parquet(
        file.replace('.dat', '.pqt'), compression=compression, engine=engine)

