#!/bin/sh
"exec" "/home/hayashiw/lhd_env/bin/python3" "$0" "$@"

import gc
import os
import re
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, TypedDict

LHD_DATA_DIR = '/home/hayashiw/LHD_data'
PQT_DATA_DIR = '/home/hayashiw/LHD_data_parquet'
NB_C = {'nb1':'r', 'nb2':'b', 'nb3':'g', 'nb4':'y', 'nb5':'m'}
MIN_FULL_PORT_RATIO = 0.8

FIBERS = {
    '6OFIDA':{
        1: {'R':295.80, 'port':6},
        2: {'R':302.90, 'port':6},
        3: {'R':309.90, 'port':6},
        4: {'R':316.80, 'port':6},
        5: {'R':323.70, 'port':6},
        6: {'R':328.30, 'port':6},
        7: {'R':332.90, 'port':6},
        8: {'R':337.40, 'port':6},
        9: {'R':341.90, 'port':6},
        10:{'R':346.40, 'port':6},
        11:{'R':350.90, 'port':6},
        12:{'R':355.30, 'port':6},
        13:{'R':359.70, 'port':6},
        14:{'R':364.20, 'port':6},
        15:{'R':368.50, 'port':6},
        16:{'R':372.90, 'port':6}},
    'PNBFIDA':{
        1: {'R':406.10, 'port':7},
        2: {'R':394.30, 'port':7},
        3: {'R':382.85, 'port':7},
        4: {'R':371.30, 'port':7},
        5: {'R':302.86, 'port':6},
        6: {'R':316.82, 'port':6},
        7: {'R':328.31, 'port':6},
        8: {'R':332.86, 'port':6},
        9: {'R':337.39, 'port':6},
        10:{'R':341.91, 'port':6},
        11:{'R':346.40, 'port':6},
        12:{'R':350.87, 'port':6},
        13:{'R':355.31, 'port':6},
        14:{'R':359.74, 'port':6},
        15:{'R':364.15, 'port':6},
        16:{'R':368.53, 'port':6}},
    'NNBFIDA':{
        1: {'R':394.30, 'port':7},
        2: {'R':394.30, 'port':7},
        3: {'R':412.00, 'port':7},
        4: {'R':406.10, 'port':7},
        5: {'R':400.20, 'port':7},
        6: {'R':395.50, 'port':7},
        7: {'R':394.30, 'port':7},
        8: {'R':393.15, 'port':7},
        9: {'R':382.85, 'port':7},
        10:{'R':371.30, 'port':7},
        11:{'R':323.73, 'port':6},
        12:{'R':332.86, 'port':6},
        13:{'R':341.91, 'port':6},
        14:{'R':350.87, 'port':6},
        15:{'R':359.74, 'port':6},
        16:{'R':368.53, 'port':6}}
}

# =============================================================================
# ====================================================== CHECK_INCREASING_INDEX
def check_increasing_index(index: np.ndarray, dx: int = 1) -> bool:
    return all([(idx - index[i]) == dx for i, idx in enumerate(index[1:])])

# =============================================================================
# ===================================================== HEAT_KEYS_TO_HEAT_INDEX
def heat_keys_to_heat_index(heat_keys: list) -> int:
    """
    heat_keys_to_heat_index(heat_keys)
    Converts list of heat key labels to binary label

    Parameters
    ----------
    heat_keys : list

    Returns
    -------
    heat_idx : binary int
    """
    heat_keys = [key.lower() for key in heat_keys]
    error = False
    for key in heat_keys:
        if key not in [
            'ech', 'nb1a', 'nb1b', 'nb2a', 'nb2b',
            'nb3a', 'nb3b', 'nb4a', 'nb4b', 'nb5a', 'nb5b'
        ]:
            print(f'\033[0;31mUnknown key: {key}\033[0m')
            error = True
    if error:
        raise ValueError(f'Unknown keys in input')
            
    index_binary_list = [0]*11
    for ikey, key in enumerate([
        'ech', 'nb1a', 'nb1b', 'nb2a', 'nb2b',
        'nb3a', 'nb3b', 'nb4a', 'nb4b', 'nb5a', 'nb5b'
    ]):
        index_binary_list[ikey] = int(key in heat_keys)
    index_binary_str = ''.join(map(str, index_binary_list))
    return int(index_binary_str, 2)

# =============================================================================
# ===================================================== HEAT_INDEX_TO_HEAT_KEYS
def heat_index_to_heat_keys(heat_idx: int) -> list:
    """
    heat_index_to_heat_keys(heat_idx)
    Converts heat index binary label to list of heat key labels

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
            [
                'ech', 'nb1a', 'nb1b', 'nb2a', 'nb2b',
                'nb3a', 'nb3b', 'nb4a', 'nb4b', 'nb5a', 'nb5b'
            ]
        ) if ison]

# =============================================================================
# ============================================================ READ_DATA_HEADER
def read_data_header(file_path: str) -> Tuple[list, list, list, list, int]:
    header = []
    with open(file_path, 'r') as f:
        line = f.readline()
        skiprows = 0
        while line[0] == '#':
            header.append(line.strip())
            skiprows += 1
            line = f.readline()
    
    dim_line = [dim for dim in header if 'DimName' in dim][0]
    dim_units_line = [unit for unit in header if 'DimUnit' in unit][0]
    val_line = [
        val for val in header if 'ValName' in val or 'Valname' in val][0]
    val_units_line = [unit for unit in header if 'ValUnit' in unit][0]
    dims = [
        dim.replace(" ", "").strip("'") for dim in
        dim_line[dim_line.rfind('=')+2:].strip().strip(',').split(',')]
    dim_units = [
        unit.replace(" ", "").strip("'") for unit in
        dim_units_line[
        dim_units_line.rfind('=')+2:].strip().strip(',').split(',')]
    vals = [
        val.replace(" ", "").strip("'") for val in
        val_line[val_line.rfind('=')+2:].strip().strip(',').split(',')]
    val_units = [
        unit.replace(" ", "").strip("'") for unit in
        val_units_line[
        val_units_line.rfind('=')+2:].strip().strip(',').split(',')]
    dims = [re.sub('-', '_', dim.lower()) for dim in dims]
    vals = [re.sub('-', '_', val.lower()) for val in vals]
    return dims, vals, dim_units, val_units, skiprows

# =============================================================================
# ==================================================================== READ_NBI
def read_nbi(
    shot: int,
    file_dir: str=LHD_DATA_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if os.path.exists(file_dir):
        print(f'Reading in NBI data in dir: {file_dir}')
    else:
        raise Exception(f'File directory [{file_dir}] does not exist')
        
    names = ['time', 'einj', 'pinj']
    comment = '#'
    usecols = [0, 1, 2, 3, 4]
    
    einj_df_list = []
    pinj_df_list = []
    ists_df_list = []
    for ikey in [1, 2, 3]:
        key = f'nb{ikey}'
        file = file_dir + f'/{key}pwr_temporal@{shot}.dat'
        if os.path.exists(file):
            print(f'Reading in NB{ikey} data from {file}')
        else:
            print(f'\033[0;31mNB{ikey} file [{file}] missing\033[0m')
        kwargs = dict(
            comment=comment, usecols=usecols,
            names=[key + '_' + name for name in (names + ['istsa', 'istsb'])])
        data = pd.read_csv(file, **kwargs)
        t_ms_idx = (data[key+'_time']*1e4).round(0).astype(int)
        data.index = t_ms_idx
        data.index.name = 'time_ms'
        data = data.drop(key+'_time', axis=1)
        data = data.loc[[idx for idx in data.index if idx%10 == 0]]
        data.index = data.index // 10
        einj_df_list.append(data[key+'_einj'].rename(key))
        pinj_df_list.append(data[key+'_pinj'].rename(key))
        ists_df_list.append(data[key+'_istsa'].rename(key+'a'))
        ists_df_list.append(data[key+'_istsb'].rename(key+'b'))
        
    for ikey in [4, 5]:
        for ab in ['a', 'b']:
            key = f'nb{ikey}{ab}'
            file = file_dir + f'/{key}pwr_temporal@{shot}.dat'
            if os.path.exists(file):
                print(f'Reading in NB{ikey}{ab} data from {file}')
            else:
                print(f'\033[0;31mNB{ikey}{ab} file [{file}] missing\033[0m')
            kwargs = dict(
                comment=comment, usecols=usecols,
                names=[key + '_' + name for \
                    name in (names + ['istsu', 'istsl'])])
            data = pd.read_csv(file, **kwargs)
            t_ms_idx = (data[key+'_time']*1e4).round(0).astype(int)
            data.index = t_ms_idx
            data.index.name = 'time_ms'
            data = data.drop(key+'_time', axis=1)
            data = data.loc[[idx for idx in data.index if idx%10 == 0]]
            data.index = data.index // 10
            einj_df_list.append(data[key+'_einj'].rename(key))
            pinj_df_list.append(data[key+'_pinj'].rename(key))
            ists_df_list.append(
                data[key+'_istsu'].rename(key[:-1]+'u'+key[-1]))
            ists_df_list.append(
                data[key+'_istsl'].rename(key[:-1]+'l'+key[-1]))
    einj_df = pd.concat(einj_df_list, axis=1)
    pinj_df = pd.concat(pinj_df_list, axis=1)
    ists_df = pd.concat(ists_df_list, axis=1)
    for ikey in [4, 5]:
        einj_df[f'nb{ikey}'] = einj_df[[f'nb{ikey}a', f'nb{ikey}b']].max(1)
        pinj_df[f'nb{ikey}'] = pinj_df[[f'nb{ikey}a', f'nb{ikey}b']].sum(1)
        for ab in ['a', 'b']:
            ists_df[f'nb{ikey}{ab}'] = \
                ists_df[[f'nb{ikey}u{ab}', f'nb{ikey}l{ab}']].max(1)
    return einj_df, pinj_df, ists_df

# =============================================================================
# ==================================================================== READ_ECH
def read_ech(shot: int, file_dir: str=LHD_DATA_DIR) -> pd.DataFrame:
    if os.path.exists(file_dir):
        print(f'Reading in ECH data in dir: {file_dir}')
    else:
        raise Exception(f'File directory [{file_dir}] does not exist')
    
    names = ['time', 'totalech']
    comment = '#'
    usecols = [0, 13]
    kwargs = dict(names=names, comment=comment, usecols=usecols)
        
    file = file_dir + f'/echpw@{shot}.dat'
    if os.path.exists(file):
        print(f'Reading in ECH data from {file}')
    else:
        print(f'\033[0;31mECH file [{file}] missing\033[0m')
    ech_df = pd.read_csv(file, **kwargs)
    t_ms_idx = (ech_df['time']*1e4).round(0).astype(int)
    ech_df.index = t_ms_idx
    ech_df.index.name = 'time_ms'
    ech_df = ech_df.drop('time', axis=1)['totalech']
    ech_df = ech_df.loc[[idx for idx in ech_df.index if idx%10 == 0]]
    ech_df.index = ech_df.index // 10
    return ech_df

# =============================================================================
# ================================================================== READ_FICXS
def read_ficxs(
    shot: int,
    file_dir: str=PQT_DATA_DIR,
    verbose: bool=False
) -> pd.DataFrame:
    if os.path.exists(file_dir):
        if verbose: print(f'Reading in FICXS data in dir: {file_dir}')
    else:
        raise Exception(f'File directory [{file_dir}] does not exist')

    file = file_dir + f'/ficxs_2_calib@{shot}.pqt'
    if os.path.exists(file):
        if verbose: print(f'Reading in FICXS data from {file}')
    else:
        raise Exception(f'FICXS file [{file}] missing')
    ficxs = pd.read_parquet(file).droplevel('shots', axis=1)
    ficxs.index = (ficxs.index*1e3).round(0).astype(int)
    return ficxs

# =============================================================================
# ============================================================ READ_FICXS_MULTI
def read_ficxs_multi(
    shots: np.ndarray, file_dir: str=PQT_DATA_DIR) -> pd.DataFrame:
    if os.path.exists(file_dir):
        print(f'Reading in FICXS data in dir: {file_dir}')
    else:
        raise Exception(f'File directory [{file_dir}] does not exist')

    shots = np.unique(np.array(shots).flatten())
    nshots = shots.size
    ficxs_list = []
    for ishot, shot in enumerate(shots):
        prefix = f'[{ishot+1}/{nshots}] {shot}'
        file = file_dir + f'/ficxs_2_calib@{shot}.pqt'
        if os.path.exists(file):
            print(f'{prefix} Reading in FICXS data from {file}')
            ficxs = pd.read_parquet(file).droplevel('shots', axis=1)
            ficxs.index = (ficxs.index*1e3).round(0).astype(int)
            ficxs.index = pd.MultiIndex.from_tuples(
                [(shot, t) for t in ficxs.index])
            ficxs_list.append(ficxs)
        else:
            print(f'\033[0;31m{prefix} FICXS file [{file}] missing\033[0m')
            # raise Exception(f'FICXS file [{file}] missing')
    ficxs_df = pd.concat(ficxs_list, axis=0)
    ficxs_df.index.names = ['shot', 't_ms']
    _ = gc.collect()
    
    return ficxs_df
    
# =============================================================================
# ================================================== CALCULATE_FICXS_CYCLE_TIME
def calculate_ficxs_cycle_time(
    shot: int,
    species: str,
    fmap: str,
    los: int,
    exp_t: int = 7,
    ficxs_threshold_u: float = 0.9,
    ficxs_threshold_l: float = 0.1,
    pad_threshold_u: float = 0.05,
    pad_threshold_l: float = 0.05,
    besw: list = [656.6, 657.7]
) -> pd.DataFrame:
    ierr = False
    if species not in ['D', 'H']:
        print(
            f'\033[0;31mSpecies must be Deuterium ("D") or Hydrogen ("H"), '
            f'species: "{species}"\033[0m')
        ierr = True
    if fmap not in ['6OFIDA', 'PNBFIDA', 'NNBFIDA']:
        print(
            f'\033[0;31mFiber map must be "6OFIDA" or "PNBFIDA" or "NNBFIDA", '
            f'fmap: "{fmap}"\033[0m')
        ierr = True
    if ierr:
        raise Exception('Error with inputs')

    # los = 6
    # besw = [657, 658]
    dnb = 10 - los
    ports = ['ua', 'ub', 'la', 'lb'] if 7-los else ['a', 'b']
    ichs = np.array([
        ich for ich in range(1, 17) if FIBERS[fmap][ich]['port'] == los])
    chs = np.array([f's{ich}' for ich in ichs])
    print(
        f'{shot} {los}-O LOS (DNB-{dnb}) '
        f'with {fmap} ({ichs.size:>2} Chs)')

    # Read in data
    einj_df, pinj_df, ists_df = read_nbi(shot)
    ech_df = read_ech(shot)
    ficxs = read_ficxs(shot)
    
    # Check FICXS data time index
    ts_index_good = check_increasing_index(ficxs.index.values, dx=10)
    if ts_index_good:
        pass
    else:
        t_index_diff = ficxs.index.diff().dropna().unique()
        raise Exception(
            f'Time index in ms is not monotonic increasing'
            f'\nt_index.diff.unique: '
            f'{t_index_diff.values.tolist()}')
    
    # Check NBI data time index
    ts_index_good = check_increasing_index(ists_df.index.values)
    if ts_index_good:
        pass
    else:
        t_index_diff = ists_df.index.diff().dropna().unique()
        raise Exception(
            f'Time index in ms is not monotonic increasing'
            f'\nt_index.diff.unique: '
            f'{t_index_diff.values.tolist()}')
    
    # Check ECH data time index
    ts_index_good = check_increasing_index(ech_df.index.values)
    if ts_index_good:
        pass
    else:
        t_index_diff = ech_df.index.diff().dropna().unique()
        raise Exception(
            f'Time index in ms is not monotonic increasing'
            f'\nt_index.diff.unique: '
            f'{t_index_diff.values.tolist()}')
        
    ists_dnb = ists_df[[f'nb{dnb}{port}' for port in ports]].max(1)
    dnb_off = ists_dnb.max() == 0
    if dnb_off:
        raise Exception(
            f'DNB-{dnb} is not on')
    
    up_edges_temp = ists_dnb[
        ists_dnb[ists_dnb.diff()>0].index.values-0].index.values
    dn_edges_temp = ists_dnb[
        ists_dnb[ists_dnb.diff()<0].index.values-1].index.values
    up_edges, dn_edges = [], []
    for i, (up, dn) in enumerate(zip(up_edges_temp, dn_edges_temp)):
        if up == dn:
            continue
        else:
            up_edges.append(up)
            dn_edges.append(dn)
    up_edges, dn_edges = np.array(up_edges), np.array(dn_edges)
    edges_paired = (
        (up_edges.size == dn_edges.size) &
        all([up < dn for up, dn in zip(up_edges, dn_edges)]) )
    if edges_paired:
        pass
    else:
        raise Exception(
            f'Rising and falling edges are offset'
            f'\nUp {up_edges.size}, Dn {dn_edges.size}'
            f'\nUp edges: {up_edges}'
            f'\nDn edges: {dn_edges}')
        
    n_cycles_0 = up_edges.size
    if n_cycles_0 <= 1:
        raise Exception(
            f'Number of cycles {n_cycles_0} <= 1')
    else:
        print(f'DNB-{dnb} has {n_cycles_0} cycles total')
    
    print(
        ' '*16 + 
        'dt_on dt_off_l dt_off_u t_cycle  duty_cycle'
        '    Einj    Pinj DNB_ports on:off heat_idx heat_keys')
    
    n_ports_max = 0
    delta_t_ons_arr = []
    t_cycles_arr = []
    ficxs_ts_list = []
    for i_cycle, (up, dn) in enumerate(zip(up_edges, dn_edges)):
        t_ms_idx = round( ((up + dn) // 2)/10 ) * 10
        delta_t_on = dn - up + 1
        if (i_cycle < 0) or (i_cycle > n_cycles_0 - 1):
            print(
                f'\033[0;31m[{i_cycle+1:>2}/{n_cycles_0:>2}: '
                f'{t_ms_idx}ms] Unknown cycle index [{i_cycle}]\033[0m')
            continue
        
        if i_cycle > 0:
            delta_t_off_l = (up-1) - (dn_edges[i_cycle-1]+1) + 1
            if i_cycle < n_cycles_0 - 1:
                delta_t_off_u = (up_edges[i_cycle+1]-1) - (dn+1) + 1
            elif i_cycle == n_cycles_0 - 1:
                delta_t_off_u = delta_t_off_l
        elif i_cycle == 0:
            delta_t_off_u = (up_edges[i_cycle+1]-1) - (dn+1) + 1
            delta_t_off_l = delta_t_off_u
        delta_t_off = delta_t_off_u
    
        t_cycle = delta_t_on + delta_t_off
        duty_cycle = delta_t_on / t_cycle
        tc_min = up-delta_t_off_l
        tc_max = dn+delta_t_off_u
        
        einj_avg = einj_df.loc[up:dn, f'nb{dnb}'].mean()
        pinj_avg = pinj_df.loc[up:dn, f'nb{dnb}'].mean()
    
        # Some cases have DNB source failure during the discharge
        # t.f. DNB port status has to be checked for every cycle
        ists_port_sum = ists_df.loc[up:dn, [
            f'nb{dnb}{port}' for port in ports]].sum(1)
        ists_nports_max = ists_port_sum.max()
        ists_full_port_ratio = ists_port_sum[(
            ists_port_sum == ists_nports_max)].size / delta_t_on
        port_failure = ists_full_port_ratio < MIN_FULL_PORT_RATIO
        if port_failure:
            print(
                f'\033[0;31m[{i_cycle+1:>2}/{n_cycles_0:>2}: '
                f'{t_ms_idx}ms] DNB port failure '
                f'{ists_port_sum.unique().tolist()}, '
                f'nports = {ists_nports_max} for '
                f'{ists_full_port_ratio*100:.1f}%\033[0m')
            continue
        
        n_ports = ists_df.loc[up:dn, [
            f'nb{dnb}{port}' for port in ports]].max(0).sum()
        n_ports_max = max(n_ports_max, n_ports)
    
        heat_idxs_df = pd.Series(
            np.zeros(tc_max-tc_min+1), dtype=int,
            index=np.arange(tc_min, tc_max+1))
        for t in np.arange(tc_min, tc_max+1):
            heat_keys = []
            for ikey in range(1, 6):
                key = f'nb{ikey}'
                if key == f'nb{dnb}': continue
                nb_ports = ['ua', 'ub', 'la', 'lb'] if ikey>3 else ['a', 'b']
                for port in nb_ports:
                    if ists_df.loc[t, f'{key}{port}']:
                        ab = 'a' if 'a' in port else 'b'
                        heat_keys.append(key+ab)
            if ech_df.loc[t]: heat_keys = heat_keys + ['ech']
            heat_idx = heat_keys_to_heat_index(heat_keys)
            heat_idxs_df.loc[t] = heat_idx
        heat_idx_0 = heat_idxs_df.mode()[0]
        heat_keys_0 = heat_index_to_heat_keys(heat_idx_0)
    
        ficxs_chs_ts = ficxs[chs].loc[tc_min:tc_max]
        ficxs_t_series = ficxs_chs_ts.apply(
            lambda x: np.mean([
                x[ch].loc[besw[0]:besw[1]].sum() for ch in chs]), axis=1)
        ficxs_t_on_window = ficxs_t_series.loc[
            up-delta_t_off_l//2:dn+delta_t_off_u//2]
    
        use_t_ons = []
        for t in ficxs_t_on_window.index:
            heat_idxs = []
            for sub_t in np.arange(t-exp_t, t+1):
                heat_keys = []
                for ikey in range(1, 6):
                    key = f'nb{ikey}'
                    if key == f'nb{dnb}': continue
                    nb_ports = ['ua', 'ub', 'la', 'lb'] if ikey>3 \
                        else ['a', 'b']
                    for port in nb_ports:
                        if ists_df.loc[sub_t, f'{key}{port}']:
                            ab = 'a' if 'a' in port else 'b'
                            heat_keys.append(key+ab)
                if ech_df.loc[sub_t]: heat_keys = heat_keys + ['ech']
                heat_idx = heat_keys_to_heat_index(heat_keys)
                heat_idxs.append(heat_idx)
            if np.unique(heat_idxs).size == 1 and heat_idxs[0] == heat_idx_0:
                use_t_ons.append(t)
        if len(use_t_ons):
            pass
        else:
            print(
                f'\033[0;31mDNB-{dnb} does not overlap '
                f'with primary heating\033[0m')
            continue
        ficxs_t_on_window = ficxs_t_on_window[use_t_ons]
        ficxs_t_max = ficxs_t_on_window.max()
    
        use_t_offs = []
        for t in np.arange(ficxs_t_series.index.min(), up, 10)[::-1]:
            heat_idxs = []
            for sub_t in np.arange(t-exp_t, t+1):
                heat_keys = []
                for ikey in range(1, 6):
                    key = f'nb{ikey}'
                    if key == f'nb{dnb}': continue
                    nb_ports = ['ua', 'ub', 'la', 'lb'] if ikey>3 \
                        else ['a', 'b']
                    for port in nb_ports:
                        if ists_df.loc[sub_t, f'{key}{port}']:
                            ab = 'a' if 'a' in port else 'b'
                            heat_keys.append(key+ab)
                if ech_df.loc[sub_t]: heat_keys = heat_keys + ['ech']
                heat_idx = heat_keys_to_heat_index(heat_keys)
                heat_idxs.append(heat_idx)
            if np.unique(heat_idxs).size == 1 and heat_idxs[0] == heat_idx_0:
                use_t_offs.append(t)
            else:
                break
        for t in np.arange(ficxs_t_series.index.max(), dn, 10)[::-1]:
            heat_idxs = []
            for sub_t in np.arange(t-exp_t, t+1):
                heat_keys = []
                for ikey in range(1, 6):
                    key = f'nb{ikey}'
                    if key == f'nb{dnb}': continue
                    nb_ports = ['ua', 'ub', 'la', 'lb'] if ikey>3 \
                        else ['a', 'b']
                    for port in nb_ports:
                        if ists_df.loc[sub_t, f'{key}{port}']:
                            ab = 'a' if 'a' in port else 'b'
                            heat_keys.append(key+ab)
                if ech_df.loc[sub_t]: heat_keys = heat_keys + ['ech']
                heat_idx = heat_keys_to_heat_index(heat_keys)
                heat_idxs.append(heat_idx)
            if np.unique(heat_idxs).size == 1 and heat_idxs[0] == heat_idx_0:
                use_t_offs.append(t)
            else:
                break
        ficxs_t_off_window = ficxs_t_series[use_t_offs]
        ficxs_t_min = ficxs_t_off_window.min()
        
        ficxs_t_ptp = ficxs_t_max - ficxs_t_min
        ficxs_t_on_window = (ficxs_t_on_window - ficxs_t_min)/ficxs_t_ptp
        ficxs_t_off_window = (ficxs_t_off_window - ficxs_t_min)/ficxs_t_ptp

        ficxs_on_ts = ficxs_t_on_window[
            ficxs_t_on_window > ficxs_threshold_u].index.values
        if ficxs_on_ts.size <= 1:
            ficxs_on_ts = ficxs_t_on_window[
                ficxs_t_on_window > (ficxs_threshold_u-pad_threshold_u)
            ].index.values
        n_on = ficxs_on_ts.size
        if n_on <= 1:
            print(
                f'\033[0;33mNumber of DNB-{dnb} on samples <= 1, '
                f'n_on: {n_on}\033[0m')
            continue
        
        ficxs_off_ts = ficxs_t_off_window[
            ficxs_t_off_window < ficxs_threshold_l].index.values
        if ficxs_off_ts.size <= 1:
            ficxs_off_ts = ficxs_t_off_window[
                ficxs_t_off_window < (ficxs_threshold_l+pad_threshold_l)
            ].index.values
        n_off = ficxs_off_ts.size
        if n_off <= 1:
            print(
                f'\033[0;33mNumber of DNB-{dnb} off samples <= 1, '
                f'n_off: {n_off}\033[0m')
            continue
        
        delta_t_ons_arr.append(delta_t_on)
        t_cycles_arr.append(t_cycle)

        for t in ficxs_on_ts:
            ficxs_ts_list.append([t_ms_idx, t, 1])
        for t in ficxs_off_ts:
            ficxs_ts_list.append([t_ms_idx, t, 0])
        
        print(
            f'[{i_cycle+1:>2}/{n_cycles_0:>2}: {t_ms_idx}ms] '
            f'{delta_t_on:>3}ms '
            f'{delta_t_off_l:>6}ms '
            f'{delta_t_off_u:>6}ms '
            f'{t_cycle:>5}ms '
            f'{duty_cycle*100:5.1f}%({duty_cycle*100:.0f}%) '
            f'{einj_avg:>5.1f}keV '
            f'{pinj_avg:>4.1f}MW '
            f'{n_ports:>4}/{n_ports_max:<4} '
            f'{n_on:>2}:{n_off:<2} '
            f'{heat_idx_0:>9} '
            f'[{", ".join(heat_keys_0)}]')
    
    delta_t_ons_arr = np.array(delta_t_ons_arr)
    t_cycles_arr = np.array(t_cycles_arr)
    n_cycles = delta_t_ons_arr.size
    if n_cycles == 0:
        raise Exception(
            f'{n_cycles:<2} out of {n_cycles_0:<2} '
            f'cycles recorded ({n_cycles/n_cycles_0*100:.1f}%)')
    
    t_ons_avg = delta_t_ons_arr.mean()
    t_ons_err = delta_t_ons_arr.std()
    t_cyc_avg = t_cycles_arr.mean()
    t_cyc_err = t_cycles_arr.std()
    print(
        f'{n_cycles:<2} out of {n_cycles_0:<2} cycles recorded '
        f'({n_cycles/n_cycles_0*100:.1f}%), '
        f'avg. DNB-on time: {t_ons_avg:5.1f} +- {t_ons_err:.2f} ms, '
        f'avg. DNB cycle time: {t_cyc_avg:5.1f} +- {t_cyc_avg:.2f} ms')
    
    ficxs_ts_df = pd.DataFrame(
        ficxs_ts_list, columns=['t_ms_idx', 't_ms', 'on']
    ).sort_values(['t_ms_idx', 't_ms']).reset_index(drop=True)
    ficxs_ts_df.attrs['desc'] = {
        't_ms_idx':'DNB cycle label (average of DNB-on time [ms])',
        't_ms':'Time stamp [ms]',
        'on':'1 if DNB on else DNB off'}
    ficxs_ts_df.index.name = 'i'
    return ficxs_ts_df

# =============================================================================
# ======================================================================== MAIN
def main(args: argparse.Namespace) -> None:
    shot = args.shot
    species = args.species
    fmap = args.fmap
    los = args.los
    exp_t = args.exp_t
    ficxs_threshold_u = args.ftu
    ficxs_threshold_l = args.ftl
    pad_threshold_u = args.ptu
    pad_threshold_l = args.ptl

    kwargs = dict(exp_t=exp_t,
        ficxs_threshold_u=ficxs_threshold_u, pad_threshold_u=pad_threshold_u,
        ficxs_threshold_l=ficxs_threshold_l, pad_threshold_l=pad_threshold_l)
    ficxs_ts_df = calculate_ficxs_cycle_time(
        shot, species, fmap, los, **kwargs)

    cwd = os.getcwd()
    save_file = f'{cwd}/{shot}_{los}o.txt'
    print(f'Saving FICXS timing info to {save_file}')
    with open(save_file, 'w') as file:
        file.write('t_ms_idx,t_ms,on\n')
        for _, i1, i2, i3 in ficxs_ts_df.itertuples():
            file.write(
                f'{i1:>6}, '
                f'{i2:>6}, '
                f'{i3:>2}\n')

# =============================================================================
# =================================================================== ARGPARSER
def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate ficxs cycle times')
    parser.add_argument('shot', type=int, help='Shot number')
    parser.add_argument('species', type=str, help='Fast ion species')
    parser.add_argument('fmap', type=str, help=(
        'FICXS channels fibermap label'
        '[6OFIDA, PNBFIDA, NNBFIDA]'))
    parser.add_argument('los', type=int, help='FICXS LOS label')
    parser.add_argument('--exp-t', type=int, default=7, help=(
        'FICXS camera exposure time [ms]'))
    parser.add_argument('--ftu', type=float, default=0.9, help=(
        'Threshold for DNB-on signals [%]'))
    parser.add_argument('--ftl', type=float, default=0.1, help=(
        'Threshold for DNB-off signals [%]'))
    parser.add_argument('--ptu', type=float, default=0.05, help=(
        'Pad for upper threshold for DNB-on signals [%]'))
    parser.add_argument('--ptl', type=float, default=0.05, help=(
        'Pad for lower threshold for DNB-off signals [%]'))
    return parser

if __name__ == '__main__':
    args = argparser().parse_args()
    main(args)