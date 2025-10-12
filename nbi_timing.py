import numpy as np
import pandas as pd

from .io import read_full_nbists_from_config as read_ists
from .io import read_data_basic as read_data
from .io import read_config
from .utils import error, check_increasing_index, heat_keys_to_heat_index

def calculate_nbi_on_cycles(
    ists_ser: pd.Series
) -> pd.DataFrame:
    r"""
    Calculate NBI on and off timestamps. Assumes index is time in ms.

    Parameters
    ----------
    ists_ser : pd.Series
        Time series with 0: beam off, 1: beam on.

    Returns
    -------
    nbi_timing_df : pd.DataFrame
        Dataframe with two columns: "on", "off", where beam is on for
        the range of [on, off]. Note that the range is inclusive on both
        ends. The dataframe index is set to the "on" value in ms rounded
        to the nearest 10 ms.
        
    """
    is_increasing_evenly = check_increasing_index(ists_ser.index.values)
    if not is_increasing_evenly:
        t_index_diff = ists_ser.index.diff().dropna()
        error(
            f'Time index in ms is not monotonic increasing\n'
            f't_index.diff.unique: {np.unique(t_index_diff)}\n'
            f't_index.diff.values: {t_index_diff.values}\n'
            f't_index.values: {ists_ser.index.values}',
            stop=True)

    up_edges = ists_ser[
        ists_ser[ists_ser.diff()>0].index.values-0
    ].index.values
    dn_edges = ists_ser[
        ists_ser[ists_ser.diff()<0].index.values-1
    ].index.values
    edges_paired = (
        (up_edges.size == dn_edges.size) &
        all([up <= dn for up, dn in zip(up_edges, dn_edges)]) )
    if not edges_paired:
        error(
            f'Rising and falling edges are offset'
            f'\nNumber of rising edges: {up_edges.size} '
            f'\nNumber of falling edges: {dn_edges.size}'
            f'\nRising edges: {up_edges}'
            f'\nFalling edges: {dn_edges}', stop=True)

    edge_pairs = []
    for up, dn in zip(up_edges, dn_edges):
        t_ms_idx = int(round(up / 10) * 10)
        edge_pairs.append([t_ms_idx, up, dn])
    data = pd.DataFrame(
        edge_pairs, columns=['t_ms_idx', 'on', 'off']).set_index('t_ms_idx')
    return data

def is_modulated(
    ists_ser: pd.Series,
    n_cycles_min: int=3,
    err_ratio_max: float=0.2
) -> bool:
    r"""
    Returns `True` if NBI is *intentionally* modulated.

    Parameters
    ----------
    ists_ser : pd.Series
        Time series with 0: beam off, 1: beam on.
    n_cycles_min : int, optional
        Minimum number of cycles for a periodic beam signal to be
        considered modulated. Default is 3.
    err_ratio_max : float, optional
        Maximum ratio of the standard deviation of the cycle widths to
        the average cycle width. Default is 0.2.

    Returns
    -------
    bool
        `True` if beam is modulated.
    """
    if ists_ser.max() == 0: return False

    timing = calculate_nbi_on_cycles(ists_ser)
    n_cycles = timing.shape[0]
    if n_cycles < n_cycles_min: return False

    deltas = timing['off'] - timing['on']
    deltas_avg = deltas.mean()
    deltas_err = deltas.std()
    err_ratio = deltas_err / deltas_avg
    if err_ratio > err_ratio_max: return False
    return True

def is_heating(
    ists_ser: pd.Series,
    t_span_min: float=300.0,
    t_on_ratio_min: float=0.4,
    n_cycles_min: int=3,
    err_ratio_max: float=0.2
) -> bool:
    r"""
    Returns `True` if NBI is a heating beam.

    Parameters
    ----------
    ists_ser : pd.Series
        Time series with 0: beam off, 1: beam on.
    t_span_min : float, optional
        Minimum width of beam pattern in milliseconds to be considered a
        heating beam. Default is 300 milliseconds.
    t_on_ratio_min : float, optional
        Only relevant for modulated beams. Minimum proportion of beam
        pattern where NBI is on in order to be considered a heating
        beam. Default is 0.4.
    n_cycles_min : int, optional
        Passed to `is_modulated`. Minimum number of cycles for a
        periodic beam signal to be considered modulated. Default is 3.
    err_ratio_max : float, optional
        Passed to `is_modulated`. Maximum ratio of the standard
        deviation of the cycle widths to the average cycle width.
        Default is 0.2.

    Returns
    -------
    bool
        `True` if beam is a heating beam.
    """
    is_mod = is_modulated(
        ists_ser, n_cycles_min=n_cycles_min, err_ratio_max=err_ratio_max)
    
    dt = np.diff(ists_ser.index).mean()
    nz_sts = ists_ser[ists_ser > 0]
    tmin = nz_sts.index.min()
    tmax = nz_sts.index.max()
    t_span = tmax - tmin
    t_on_ratio = nz_sts.size*dt / t_span
    
    if (
        (is_mod and (t_on_ratio < t_on_ratio_min)) or
        (t_span < t_span_min) ):
        return False
    else:
        return True

def diagnosed_hnb_keys(
    ists_df: pd.DataFrame,
    dnb_overlap_min: float=0.5,
    t_span_min: float=300.0,
    t_on_ratio_min: float=0.4,
    n_cycles_min: int=3,
    err_ratio_max: float=0.2
) -> list:
    r"""
    Returns labels of beam-ports that are heating beams with overlapping
    diagnostic beams.

    Parameters
    ----------
    ists_df : pd.DataFrame
        Dataframe with NBI time series, 0: beam off, 1: beam on.
    dnb_overlap_min : float, optional
        Minimum overlap with diagnostic beam for heating beam to be
        considered "diagnosed". Default is 0.5.
    t_span_min : float, optional
        Minimum width of beam pattern in milliseconds to be considered a
        heating beam. Default is 300 milliseconds.
    t_on_ratio_min : float, optional
        Only relevant for modulated beams. Minimum proportion of beam
        pattern where NBI is on in order to be considered a heating
        beam. Default is 0.4.
    n_cycles_min : int, optional
        Passed to `is_modulated`. Minimum number of cycles for a
        periodic beam signal to be considered modulated. Default is 3.
    err_ratio_max : float, optional
        Passed to `is_modulated`. Maximum ratio of the standard
        deviation of the cycle widths to the average cycle width.
        Default is 0.2.

    Returns
    -------
    heating_beam_keys : list
        List of beam-port labels for heating beams that are diagnosed.
    """
    heating_kwargs = dict(
        n_cycles_min=n_cycles_min, err_ratio_max=err_ratio_max,
        t_span_min=t_span_min, t_on_ratio_min=t_on_ratio_min )
    heating_beam_keys = []
    keys = [f'nb{i}{ab}' for i in [1, 2, 3] for ab in ['a', 'b']] + \
        [f'nb{i}{ul}{ab}' for i in [4, 5]
         for ab in ['a', 'b'] for ul in ['u', 'l']]
    for key in keys:
        if ists_df[key].max() == 0: continue
        if not is_heating(ists_df[key], **heating_kwargs): continue
        if ('nb3' in key or 'nb4' in key) and is_modulated(ists_df[key]):
            heating_beam_keys.append(key)
            continue
        is_diagnosed = False
        nz_sts = ists_df[key][ists_df[key] > 0]
        tmin = nz_sts.index.min()
        tmax = nz_sts.index.max()
        t_span = tmax - tmin
        if ('nb1' in key or 'nb2' in key or 'nb5' in key):
            dnb_keys = ['nb3', 'nb4']
        elif 'nb3' in key:
            dnb_keys = ['nb4']
        elif 'nb4' in key:
            dnb_keys = ['nb3']
        else:
            raise Exception(f'Uknown key: {key}')
        for dnb_key in dnb_keys:
            dnb_is_diag = is_modulated(ists_df[dnb_key])
            dnb_nz_sts = ists_df[dnb_key][ists_df[dnb_key] > 0]
            if dnb_nz_sts.size == 0: continue
            dnb_tmin = dnb_nz_sts.index.min()
            dnb_tmax = dnb_nz_sts.index.max()

            dnb_overlap = np.intersect1d(
                np.arange(tmin, tmax+1),
                np.arange(dnb_tmin, dnb_tmax + 1)
            ).size / t_span

            if dnb_is_diag and dnb_overlap > dnb_overlap_min:
                is_diagnosed = True
                
        if is_diagnosed: heating_beam_keys.append(key)
    return heating_beam_keys

def identify_heat_idx_sections(
    shot: int,
    config: dict=None,
    heat_idx_t_span_min: float=300.0,
    heat_idx_t_on_ratio_min: float=0.7,
    dnb_overlap_min: float=0.5,
    t_span_min: float=300.0,
    t_on_ratio_min: float=0.4,
    n_cycles_min: int=3,
    err_ratio_max: float=0.2,
) -> pd.DataFrame:
    r"""
    Returns heating sections grouped by heating schemes (NBI and ECH).

    Parameters
    ----------
    shot : int
        Shot number.
    config : dict
        Dictionary containing program configuration.
    heat_idx_t_span_min : float, optional
        Minimum width of section in milliseconds to be
        considered a heating section. Default is 300 milliseconds.
    heat_idx_t_on_ratio_min : float, optional
        Minimum proportion of section where heat index is the selected
        index. This is to deal with heating sections with gaps in the
        middle. Default is 0.7.
    dnb_overlap_min : float, optional
        Minimum overlap with diagnostic beam for heating beam to be
        considered "diagnosed". Default is 0.5.
    t_span_min : float, optional
        Minimum width of beam pattern in milliseconds to be considered a
        heating beam. Default is 300 milliseconds.
    t_on_ratio_min : float, optional
        Only relevant for modulated beams. Minimum proportion of beam
        pattern where NBI is on in order to be considered a heating
        beam. Default is 0.4.
    n_cycles_min : int, optional
        Passed to `is_modulated`. Minimum number of cycles for a
        periodic beam signal to be considered modulated. Default is 3.
    err_ratio_max : float, optional
        Passed to `is_modulated`. Maximum ratio of the standard
        deviation of the cycle widths to the average cycle width.
        Default is 0.2.

    Returns
    -------
    heat_idx_sects_df : pd.DataFrame
        Dataframe containing the start and stop time (in milliseconds)
        of the heating sections along with the heat index.
    """
    heating_kwargs = dict(
        dnb_overlap_min=dnb_overlap_min, n_cycles_min=n_cycles_min,
        err_ratio_max=err_ratio_max, t_span_min=t_span_min,
        t_on_ratio_min=t_on_ratio_min )
    ists_df = read_ists(shot)
    for i in [1, 2, 3, 4, 5]:
        ists_df[f'nb{i}'] = ists_df.filter(regex=rf'nb{i}\w+').max(1)

    if config is None: config = read_config()
    data_dir = config['data_dir']
    ech_patt = config['ech_patt']
    ech_file = data_dir + '/' + ech_patt.format(shot=shot)
    ech_sts_df = read_data(
        ech_file, use_postgres_names=True, convert_to_ms=True)['total_ech']
    ech_sts_df = ech_sts_df.where(ech_sts_df == 0, 1).astype(int)
    heat_keys = diagnosed_hnb_keys(ists_df, **heating_kwargs)
    
    dt = np.diff(ists_df.index).mean()
    heat_idx_sects_list = []
    if len(heat_keys) == 0:
        raise Exception(
            f'No viable heating sections diagnosed by NB3 or NB4')
    elif len(heat_keys) == 1:
        key = heat_keys[0]
        nz_sts = ists_df[key][ists_df[key] > 0]
        tmin = nz_sts.index.min()
        tmax = nz_sts.index.max()
        heat_idx = heat_keys_to_heat_index(heat_keys)
        heat_idx_sects_list.append([tmin, tmax, heat_idx])
    else:
        hnb_sts = ists_df[heat_keys]
        hnb_sts = hnb_sts[hnb_sts.sum(1) > 0]
        hnb_sts['ech'] = ech_sts_df.loc[hnb_sts.index]
        heat_idxs_df = hnb_sts.apply(
            lambda x:
            heat_keys_to_heat_index([name for name, val in x.items() if val]),
            axis=1)
        heat_idxs = heat_idxs_df.unique()
        for heat_idx in heat_idxs:
            select_heat_idx = heat_idxs_df[heat_idxs_df == heat_idx]
            tmin = select_heat_idx.index.min()
            tmax = select_heat_idx.index.max()
            heat_idx_t_span = tmax - tmin
            if heat_idx_t_span < heat_idx_t_span_min: continue
            heat_idx_t_on_ratio = select_heat_idx.size*dt / heat_idx_t_span
            if heat_idx_t_on_ratio < heat_idx_t_on_ratio_min: continue
            heat_idx_sects_list.append([tmin, tmax, heat_idx])
    heat_idx_sects_df = pd.DataFrame(
        heat_idx_sects_list, columns=['start', 'stop', 'heat_idx'])
    
    if heat_idx_sects_df.index.size > 1:
        ierr = 0
        for i, start, stop in heat_idx_sects_df[['start','stop']].itertuples():
            other_i = ~heat_idx_sects_df.index.isin([i])
            other_start_stops = heat_idx_sects_df[other_i][['start','stop']]
            for j, a, b in other_start_stops.itertuples():
                overlap = np.intersect1d(
                    np.arange(start, stop+1),
                    np.arange(a, b+1) )
                if overlap.size:
                    ierr = 1
                    print(
                        f'[{i} ({start}, {stop}), '
                        f'{j} ({a}, {b})] '
                        f'\033[0;31mOverlap\033[0m')
        if ierr: raise Exception('Overlap in sections')

    return heat_idx_sects_df