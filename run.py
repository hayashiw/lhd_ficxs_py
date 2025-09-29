import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import heat_index_to_heat_keys, heat_keys_to_heat_index, warn
from .io import read_data_basic, read_data_ech, read_nbists_from_files
from .nbi_timing import calculate_nbi_on_cycles

DATA_DIR = '/home/hayashiw/LHD_data'
PQT_DIR = '/home/hayashiw/LHD_data_parquet'

def full_heat_idxs(shot: int, ignore_keys: list=[]) -> pd.Series:
    nb1_file  = DATA_DIR + f'/nb1pwr_temporal@{shot}.dat'
    nb2_file  = DATA_DIR + f'/nb2pwr_temporal@{shot}.dat'
    nb3_file  = DATA_DIR + f'/nb3pwr_temporal@{shot}.dat'
    nb4a_file = DATA_DIR + f'/nb4apwr_temporal@{shot}.dat'
    nb4b_file = DATA_DIR + f'/nb4bpwr_temporal@{shot}.dat'
    nb5a_file = DATA_DIR + f'/nb5apwr_temporal@{shot}.dat'
    nb5b_file = DATA_DIR + f'/nb5bpwr_temporal@{shot}.dat'
    ech_file  = DATA_DIR + f'/echpw@{shot}.dat'
    
    ists_df_list = []
    for ifile, (file, key) in enumerate(zip(
        [
            nb1_file, nb2_file, nb3_file,
            nb4a_file, nb4b_file,
            nb5a_file, nb5b_file
        ], [
            'nb1', 'nb2', 'nb3',
            'nb4a', 'nb4b',
            'nb5a', 'nb5b'
        ]
    )):
        ports = ['a', 'b'] if ifile < 3 else ['u', 'l']

        nbi_data = read_data_basic(
            file, use_postgres_names=True, convert_to_ms=True)

        for port in ports:
            name = key+port if ifile < 3 else key[:-1]+port+key[-1]
            if name in ignore_keys: continue
            ists_ser = nbi_data[f'ion_sts{port}_{key}'].astype(int)
            ists_ser.name = name
            ists_df_list.append(ists_ser)
    ists_df = pd.concat(ists_df_list, axis=1)

    if 'ech' not in ignore_keys:
        ech_data = read_data_ech(
            ech_file, use_postgres_names=True, convert_to_ms=True)
        ists_ech = ech_data['total_ech']
        ists_ech = ists_ech.where(ists_ech == 0, 1).astype(int)

    heat_idxs_ser_list = []
    for t in ists_df.index:
        heat_keys = [key for key, val in ists_df.loc[t].items() if val]
        if 'ech' not in ignore_keys and t in ists_ech and ists_ech.loc[t]:
            heat_keys = heat_keys + ['ech']
        heat_idx = heat_keys_to_heat_index(heat_keys)
        heat_idxs_ser_list.append(heat_idx)
    heat_idxs_ser = pd.Series(
        heat_idxs_ser_list, index=ists_df.index)
    return heat_idxs_ser

def calculate_ficxs_timing(
    shot: int,
    los: int,
    verbose: bool=False,
    ficxs_upper_threshold: float=0.8,
    ficxs_lower_threshold: float=0.2
):
    ficxs_pqt_file = PQT_DIR + f'/ficxs_2_calib@{shot}.pqt'
    max_cycle_time = 200 if los == 6 else 100 # ms
    besw = [656.6, 657.7] if los == 6 else [664.5, 666.5]
    dnb = 10 - los
    ch = 's16' if los == 6 else 's4'
    ignore_keys = (
        ['nb4ua', 'nb4la', 'nb4ub', 'nb4lb'] if los == 6 else ['nb3a', 'nb3b'])
    if verbose: print(f'{los}-O LOS, diganostic beam {dnb}')
    
    heat_idxs_ser = full_heat_idxs(
        shot, ignore_keys=ignore_keys)

    dnb_files = \
    ( [
        DATA_DIR + f'/nb{dnb}apwr_temporal@{shot}.dat',
        DATA_DIR + f'/nb{dnb}bpwr_temporal@{shot}.dat']
        if los == 6 else 
        [DATA_DIR + f'/nb{dnb}pwr_temporal@{shot}.dat'] )
    dnb_ists = read_nbists_from_files(
        dnb_files, use_postgres_names=True, convert_to_ms=True)
    nbi_timing_df = calculate_nbi_on_cycles(dnb_ists)
    t_ms_idxs = nbi_timing_df.index.values
    n_cycles = t_ms_idxs.size
    if verbose:
        print(f'{n_cycles} cycles for diagnostic beam {dnb}')
        for i_cycle, (t_ms_idx, on, off) in \
        enumerate(nbi_timing_df.itertuples()):
            delta_t_on = off - on + 1
            delta_t_off = 0
            if n_cycles > 1:
                if i_cycle == 0:
                    delta_t_off = \
                    nbi_timing_df.loc[t_ms_idxs[i_cycle+1], 'on'] - off - 1
                else:
                    delta_t_off = \
                    on - nbi_timing_df.loc[t_ms_idxs[i_cycle-1], 'off'] - 1
            print(
                f'[{i_cycle+1}/{n_cycles} {t_ms_idx}] '
                f'{delta_t_on} ms on, {delta_t_off} ms off')
    
    ficxs_pqt = pd.read_parquet(ficxs_pqt_file)
    ficxs_pqt_index = ficxs_pqt.index.values
    ficxs_pqt_index = (ficxs_pqt_index*1e3).astype(int)
    ficxs_pqt.index = ficxs_pqt_index

    if verbose: print('Calculating DNB timing')
    dnb_ts_ser_list = []
    for i_cycle, (t_ms_idx, on, off) in enumerate(nbi_timing_df.itertuples()):
        if verbose: print(f'[{i_cycle+1}/{n_cycles} {t_ms_idx}] {on} - {off}')
        ts_on = []
        heat_idx = heat_idxs_ser.loc[t_ms_idx]
        heat_keys = heat_index_to_heat_keys(heat_idx)
        if verbose:
            print(f'    Background heat keys: {heat_keys} ({heat_idx})')
        n_ts = off - on + 1
        for i_t, t in enumerate(np.arange(on, off+1)):
            if t not in ficxs_pqt_index: continue
            dnb_7ms = dnb_ists.loc[t-7:t].mean()
            if dnb_7ms < 0.8:
                if verbose:
                    warn(f'    [{i_t+1}/{n_ts} {t}] dnb_7ms: {dnb_7ms:.3f}')
                continue
            heat_idxs = heat_idxs_ser.loc[t-7:t].unique()
            n_heat_idxs = heat_idxs.size
            if n_heat_idxs > 1:
                if verbose:
                    warn(
                        f'    [{i_t+1}/{n_ts} {t}] '
                        f'n_heat_idxs: {n_heat_idxs}')
                    for heat_idx_i in sorted(heat_idxs):
                        heat_keys_i = heat_index_to_heat_keys(heat_idx_i)
                        warn(f'        {heat_idx_i}: {heat_keys_i}')
                continue
            if heat_idxs[0] != heat_idx:
                heat_key_0 = heat_index_to_heat_keys(heat_idxs[0])
                if verbose:
                    warn(
                        f'    [{i_t+1}/{n_ts} {t}] '
                        f'heat_keys[0]: {heat_key_0} ({heat_idxs[0]})')
                continue
            if verbose: print(f'    [{i_t+1}/{n_ts} {t}]')
            ts_on.append(t)
        if len(ts_on) == 0:
            if verbose: warn(f'    0 DNB-on time stamps recorded')
            continue
        if verbose: print(f'    {len(ts_on)} DNB-on time stamps recorded')
            
        ts_off_pre = []
        for i_t, t in enumerate(np.arange(on-max_cycle_time//2, on)[::-1]):
            if t not in ficxs_pqt_index: continue
            dnb_7ms = dnb_ists.loc[t-7:t].mean()
            if dnb_7ms > 0.0:
                if verbose:
                    warn(f'    [{i_t+1}/??? {t}] dnb_7ms: {dnb_7ms:.3f}')
                break
            heat_idxs = heat_idxs_ser.loc[t-7:t].unique()
            n_heat_idxs = heat_idxs.size
            if n_heat_idxs > 1:
                if verbose:
                    warn(
                        f'    [{i_t+1}/??? {t}] '
                        f'n_heat_idxs: {n_heat_idxs}')
                    for heat_idx_i in sorted(heat_idxs):
                        heat_keys_i = heat_index_to_heat_keys(heat_idx_i)
                        warn(f'        {heat_idx_i}: {heat_keys_i}')
                break
            if heat_idxs[0] != heat_idx:
                heat_key_0 = heat_index_to_heat_keys(heat_idxs[0])
                if verbose:
                    warn(
                        f'    [{i_t+1}/??? {t}] '
                        f'heat_keys[0]: {heat_key_0} ({heat_idxs[0]})')
                break
            if verbose: print(f'    [{i_t+1}/??? {t}]')
            ts_off_pre.append(t)
        if verbose:
            if len(ts_off_pre) == 0:
                warn(f'    0 DNB-off pre-cycle time stamps recorded')
            else:
                print(
                    f'    {len(ts_off_pre)} DNB-off '
                    f'pre-cycle time stamps recorded')
        ts_off_post = []
        for i_t, t in enumerate(np.arange(off+1, off+max_cycle_time//2)):
            if t not in ficxs_pqt_index: continue
            dnb_7ms = dnb_ists.loc[t-7:t].mean()
            if dnb_7ms > 0.0:
                if verbose:
                    warn(f'    [{i_t+1}/??? {t}] dnb_7ms: {dnb_7ms:.3f}')
                break
            heat_idxs = heat_idxs_ser.loc[t-7:t].unique()
            n_heat_idxs = heat_idxs.size
            if n_heat_idxs > 1:
                if verbose:
                    warn(
                        f'    [{i_t+1}/??? {t}] '
                        f'n_heat_idxs: {n_heat_idxs}')
                    for heat_idx_i in sorted(heat_idxs):
                        heat_keys_i = heat_index_to_heat_keys(heat_idx_i)
                        warn(f'        {heat_idx_i}: {heat_keys_i}')
                break
            if heat_idxs[0] != heat_idx:
                heat_key_0 = heat_index_to_heat_keys(heat_idxs[0])
                if verbose:
                    warn(
                        f'    [{i_t+1}/??? {t}] '
                        f'heat_keys[0]: {heat_key_0} ({heat_idxs[0]})')
                break
            if verbose: print(f'    [{i_t+1}/??? {t}]')
            ts_off_post.append(t)
        if verbose:
            if len(ts_off_post) == 0:
                warn(f'    0 DNB-off post-cycle time stamps recorded')
            else:
                print(
                    f'    {len(ts_off_post)} DNB-off '
                    f'post-cycle time stamps recorded')
        ts_off = sorted(ts_off_pre + ts_off_post)
        if len(ts_off) == 0:
            if verbose: warn(f'    0 DNB-off time stamps recorded')
            continue
        if verbose: print(f'    {len(ts_off)} DNB-off time stamps recorded')

        for t in ts_off:
            dnb_ts_ser_list.append([t_ms_idx, t, 0])
        for t in ts_on:
            dnb_ts_ser_list.append([t_ms_idx, t, 1])
    dnb_ts_ser = pd.DataFrame(
        dnb_ts_ser_list, columns=['t_ms_idx', 't_ms', 'on']
    ).set_index(['t_ms_idx', 't_ms'])['on']

    ficxs_success_count = 0
    ficxs_ts_ser_list = []
    for i_cycle, t_ms_idx in enumerate(t_ms_idxs):
        if verbose: print(f'[{i_cycle+1}/{n_cycles} {t_ms_idx}]')
        ficxs_ts = dnb_ts_ser.loc[t_ms_idx]
        t_ons = ficxs_ts[ficxs_ts > 0].index.values
        t_offs = ficxs_ts[ficxs_ts == 0].index.values
        tmin = min(t_ons.min(), t_offs.min())
        tmax = max(t_ons.max(), t_offs.max())

        ficxs_ists = \
            ficxs_pqt[ch, shot].loc[tmin:tmax, besw[0]:besw[1]].sum(1)
        ficxs_max = ficxs_ists.loc[t_ons].max()
        ficxs_min = ficxs_ists.loc[t_offs].min()
        ficxs_ptp = ficxs_max - ficxs_min
        ficxs_ists = (ficxs_ists - ficxs_min) / ficxs_ptp
        
        t_ons = [
            t_on for t_on in t_ons if
            ficxs_ists.loc[t_on] > ficxs_upper_threshold]
        t_offs = [
            t_off for t_off in t_offs if
            ficxs_ists.loc[t_off] < ficxs_lower_threshold]
        
        if len(t_ons)*len(t_offs):
            ficxs_success_count = ficxs_success_count + 1
            for t in t_ons:
                ficxs_ts_ser_list.append([t_ms_idx, t, 1])
            for t in t_offs:
                ficxs_ts_ser_list.append([t_ms_idx, t, 0])
        else:
            warn(
                f'    No available FICXS: '
                f'len(t_ons)={len(t_ons)}, len(t_offs)={len(t_offs)}')
    ficxs_ts_ser = pd.DataFrame(
        ficxs_ts_ser_list, columns=['t_ms_idx', 't_ms', 'on']
    ).set_index(['t_ms_idx', 't_ms'])['on'].sort_index()
    if verbose:
        if ficxs_success_count == 0:
            warn(
                f'{ficxs_success_count}/{n_cycles} cycles '
                f'recorded for FICXS {los}-O LOS with DNB-{dnb}')
        else:
            print(
                f'{ficxs_success_count}/{n_cycles} cycles '
                f'successfully recorded for FICXS {los}-O LOS with DNB-{dnb}')
    
    return ficxs_ts_ser

def plot_ficxs_cycle(shot, los, ficxs_ts_ser_idx):
    ficxs_pqt_file = PQT_DIR + f'/ficxs_2_calib@{shot}.pqt'
    besw = [656.6, 657.7] if los == 6 else [664.5, 666.5]
    dnb = 10 - los
    ch = 's16' if los == 6 else 's4'

    ficxs_pqt = pd.read_parquet(ficxs_pqt_file)
    ficxs_pqt_index = ficxs_pqt.index.values
    ficxs_pqt_index = (ficxs_pqt_index*1e3).astype(int)
    ficxs_pqt.index = ficxs_pqt_index
    
    fig, ax = plt.subplots(
        1, 2, figsize=(8, 6), dpi=120, layout='constrained', gridspec_kw=dict(
            width_ratios=(3, 1)))

    pports_list = []
    for key in ['nb1', 'nb2', 'nb3']:
        file = DATA_DIR + f'/{key}pwr_temporal@{shot}.dat'
        pport = read_data_basic(
            file, use_postgres_names=True, convert_to_ms=True
        )[f'pport_through_{key}']
        pport.name = key
        pports_list.append(pport)
    for key in ['nb4', 'nb5']:
        pports = []
        for port in ['a', 'b']:
            file = DATA_DIR + f'/{key}{port}pwr_temporal@{shot}.dat'
            pport = read_data_basic(
                file, use_postgres_names=True, convert_to_ms=True
            )[f'pport_through_{key}{port}']
            pports.append(pport)
        pports = pd.concat(pports, axis=1).sum(1)
        pports.name = key
        pports_list.append(pports)
    pports = pd.concat(pports_list, axis=1)
    
    ech_file = DATA_DIR + f'/echpw@{shot}.dat'
    ech = read_data_ech(
        ech_file, use_postgres_names=True, convert_to_ms=True
    )['total_ech']

    ts = ficxs_ts_ser_idx.index.values
    t_ons = ficxs_ts_ser_idx[ficxs_ts_ser_idx == 1].index.values
    t_offs = ficxs_ts_ser_idx[ficxs_ts_ser_idx == 0].index.values
    tmin = ts.min()
    tmax = ts.max()

    on = ficxs_pqt.loc[t_ons, (ch, shot)]
    x = on.columns.values
    off = ficxs_pqt.loc[t_offs, (ch, shot)]
    on_avg, on_err = on.mean(), on.std()
    off_avg, off_err = off.mean(), off.std()
    net, err = on_avg-off_avg, np.sqrt(on_err**2 + off_err**2)
    err = np.where(net > 0, err, np.nan)
    net = np.where(net > 0, net, np.nan)
    for y, e, c in \
    zip([on_avg, off_avg, net], [on_err, off_err, err], ['b', 'r', 'k']):
        ax[0].plot(x, y, c=c, lw=1)
        ax[0].fill_between(
            x, y-e, y+e, color=c, alpha=0.3)
    ymin = 1e15
    ax[0].fill_between(
        x[(x>besw[0]) & (x<besw[1])],
        ymin, net[(x>besw[0]) & (x<besw[1])], color='none', edgecolor='m',
        hatch='////')
    ax[0].set(xlim=(655.5, 665.5), ylim=(ymin, 1e20), yscale='log')
    
    ficxs_ists = ficxs_pqt[ch, shot].loc[tmin:tmax, besw[0]:besw[1]].sum(1)
    ficxs_max = ficxs_ists.loc[t_ons].max()
    ficxs_min = ficxs_ists.loc[t_offs].min()
    ficxs_ptp = ficxs_max - ficxs_min
    ficxs_ists = (ficxs_ists - ficxs_min) / ficxs_ptp * 7.5
    ficxs_on = ficxs_ists.loc[t_ons]
    ficxs_off = ficxs_ists.loc[t_offs]
    for i, c in zip([1, 2, 3, 4, 5], ['r', 'b', 'g', 'y', 'm']):
        key = f'nb{i}'
        ax[1].plot(
            pports.loc[tmin-20:tmax+20, key], c=c, lw=2, label=key, zorder=0)
    ax[1].plot(ech.loc[tmin-20:tmax+20], c='c', lw=2, label='ech', zorder=0)
    ax[1].plot(
        ficxs_ists.index, ficxs_ists, 'o-', c='gray', lw=1, ms=4, zorder=5)
    ax[1].scatter(
        t_ons, ficxs_on, c='b', s=36, zorder=10)
    ax[1].scatter(
        t_offs, ficxs_off, c='r', s=36, zorder=10)

def main():
    pass

if __name__ == '__main__':
    main()