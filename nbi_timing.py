import pandas as pd

from .utils import error, check_increasing_index

def calculate_nbi_on_cycles(ists_ser: pd.Series) -> pd.DataFrame:
    """
    Calculate NBI on and off timestamps. Assumes index is time in ms.

    Parameters
    ----------
    ists_ser : pd.Series
        Time series with 0: beam off, 1: beam on

    Returns
    -------
    nbi_timing_df : pd.DataFrame
        Dataframe with two columns: "on", "off", where beam is on for
        the range of [on, off]. Not that the range is inclusive on both
        ends. The dataframe index is set to avg(on, off) in ms.
        
    """
    is_increasing_evenly = check_increasing_index(ists_ser.index.values)
    if not is_increasing_evenly:
        t_index_diff = ists_ser.index.diff().dropna().unique()
        error(
            f'Time index in ms is not monotonic increasing'
            f'\nt_index.diff.unique: '
            f'{t_index_diff.values.tolist()}', stop=True)

    up_edges = ists_ser[
        ists_ser[ists_ser.diff()>0].index.values-0
    ].index.values
    dn_edges = ists_ser[
        ists_ser[ists_ser.diff()<0].index.values-1
    ].index.values
    edges_paired = (
        (up_edges.size == dn_edges.size) &
        all([up < dn for up, dn in zip(up_edges, dn_edges)]) )
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