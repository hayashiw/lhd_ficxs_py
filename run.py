import argparse
# import configparser
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from typing import Tuple

from .io import (
    read_data_basic, read_data_ech, read_nbists_from_files, read_config,
    read_fibermaps, read_fibermaps_log, write_fibermaps_log,
    read_ficxs_from_pqt, read_ficxs_timing, write_ficxs_timing,
    read_full_nbists_from_config )
from .nbi_timing import calculate_nbi_on_cycles
from .physics import J_PER_EV, DEUT_MASS_KG, HYDR_MASS_KG
from .physics import convert_kinetic_energy_to_Doppler_shift as convert_J_to_nm
from .utils import (
    heat_index_to_heat_keys, heat_keys_to_heat_index,
    success, warn, error, conversion_factor )
from .process_signal import determine_fibermap, find_bes_wavelengths

def full_heat_idxs(
    shot: int,
    ignore_keys: list=[],
    config: dict=None
) -> pd.Series:
    ists_df = read_full_nbists_from_config(shot, config=config)

    data_dir = config['data_dir']
    if 'ech' not in ignore_keys:
        ech_file = data_dir + '/' + config['ech_patt'].format(shot=shot)
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
    ch: int,
    bes_w: list[float, float],
    verbose: bool=False,
    ficxs_upper_threshold: float=0.8,
    ficxs_lower_threshold: float=0.2,
    config: dict=None,
    n_cycles_min: int=3
) -> pd.Series:
    if config is None: config = read_config()
    data_dir = config['data_dir']
    pqt_dir = config['pqt_dir']
        
    max_cycle_time = 200 if los == 6 else 100 # ms
    
    dnb = 10 - los
    ignore_keys = (
        ['nb4ua', 'nb4la', 'nb4ub', 'nb4lb'] if dnb == 4 else ['nb3a', 'nb3b'])
    if verbose: print(f'{los}-O LOS, diganostic beam {dnb}')
    
    heat_idxs_ser = full_heat_idxs(
        shot, ignore_keys=ignore_keys)

    dnb_files = [
        data_dir + '/' + patt.format(shot=shot) for key, patt in 
        config[f'{los}o_los_dnb_file_patterns'].items()]
    dnb_ists = read_nbists_from_files(
        dnb_files, use_postgres_names=True, convert_to_ms=True)
    nbi_timing_df = calculate_nbi_on_cycles(dnb_ists)
    t_ms_idxs = nbi_timing_df.index.values
    n_cycles = t_ms_idxs.size
    if n_cycles < n_cycles_min:
        error(
            f'{n_cycles} cycles is less than minimum {n_cycles_min}',
            stop=True)
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

    ficxs_pqt_file = pqt_dir + '/' + config['ficxs_pqt_patt'].format(shot=shot)
    ficxs_pqt = read_ficxs_from_pqt(ficxs_pqt_file, convert_to_ms=True)
    ficxs_pqt_t_idx = ficxs_pqt.index.droplevel('wavelength').unique()

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
            if t not in ficxs_pqt_t_idx: continue
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
            if t not in ficxs_pqt_t_idx: continue
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
            if t not in ficxs_pqt_t_idx: continue
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
            ficxs_pqt[ch].unstack().loc[tmin:tmax, bes_w[0]:bes_w[1]].sum(1)
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

    ficxs_ts_ser.attrs['input_settings'] = dict(
        shot=shot,
        los=los,
        ch=ch,
        ficxs_upper_threshold=ficxs_upper_threshold,
        ficxs_lower_threshold=ficxs_lower_threshold )
    return ficxs_ts_ser

def plot_ficxs_cycle(
    shot: int,
    los: int,
    ch: int,
    bes_w: list[float, float],
    ficxs_ts_ser_idx: pd.Series,
    config: dict=None
) -> Tuple[plt.figure, plt.axes]:
    if config is None: config = read_config()
    data_dir = config['data_dir']
    pqt_dir = config['pqt_dir']
    
    ficxs_pqt_file = pqt_dir + '/' + config['ficxs_pqt_patt'].format(shot=shot)
    ficxs_pqt = read_ficxs_from_pqt(ficxs_pqt_file, convert_to_ms=True)
    ficxs_pqt_t_idx = ficxs_pqt.index.droplevel('wavelength').unique()
    
    fig, ax = plt.subplots(
        1, 2, figsize=(7, 4.5), dpi=120, layout='constrained', gridspec_kw=dict(
            width_ratios=(5, 2)))

    pports_list = []
    for key, patt in config['nnbi_patts'].items():
        file = data_dir + '/' + patt.format(shot=shot)
        pport = read_data_basic(
            file, use_postgres_names=True, convert_to_ms=True
        )[f'pport_through_{key}']
        pport.name = key
        pports_list.append(pport)
    for key in ['nb4', 'nb5']:
        pports = []
        for port in ['a', 'b']:
            file = data_dir + '/' + \
                config['pnbi_patts'][key+port].format(shot=shot)
            pport = read_data_basic(
                file, use_postgres_names=True, convert_to_ms=True
            )[f'pport_through_{key}{port}']
            pports.append(pport)
        pports = pd.concat(pports, axis=1).sum(1)
        pports.name = key
        pports_list.append(pports)
    pports = pd.concat(pports_list, axis=1)

    ech_file = data_dir + '/' + config['ech_patt'].format(shot=shot)
    ech = read_data_ech(
        ech_file, use_postgres_names=True, convert_to_ms=True
    )['total_ech']

    ts = ficxs_ts_ser_idx.index.values
    t_ons = ficxs_ts_ser_idx[ficxs_ts_ser_idx == 1].index.values
    t_offs = ficxs_ts_ser_idx[ficxs_ts_ser_idx == 0].index.values
    tmin = ts.min()
    tmax = ts.max()

    on = ficxs_pqt.loc[t_ons, ch].unstack()
    x = on.columns.values
    off = ficxs_pqt.loc[t_offs, ch].unstack()
    on_avg, on_err = on.mean(), on.std()
    off_avg, off_err = off.mean(), off.std()
    net, err = on_avg-off_avg, np.sqrt(on_err**2 + off_err**2)
    err = np.where(net > 0, err, np.nan)
    net = np.where(net > 0, net, np.nan)
    for y, e, c, label in zip(
        [on_avg, off_avg, net],
        [on_err, off_err, err],
        ['b', 'r', 'k'],
        ['on', 'off', 'net']
    ):
        ax[0].plot([], [], c=c, lw=2, label=label)
        ax[0].plot(x, y, c=c, lw=1)
        ax[0].fill_between(
            x, y-e, y+e, color=c, alpha=0.3)
    ymin = 1e15
    ax[0].fill_between(
        x[(x>bes_w[0]) & (x<bes_w[1])],
        ymin, net[(x>bes_w[0]) & (x<bes_w[1])], color='none', edgecolor='m',
        hatch='////', label='BES')
    ax[0].set(xlim=(655.5, 665.5), ylim=(ymin, 1e20), yscale='log')
    ax[0].set_ylabel(
        r'FICXS [Ph/(s$\cdot$m$^2\cdot$sr$\cdot$nm)]', fontsize=14)
    ax[0].set_xlabel('Wavelength [nm]', fontsize=14)
    ax[0].set_title(f'{shot} {los}-O LOS Ch{ch}', fontsize=14)
    ax[0].legend(loc='upper right', fontsize=12, ncol=2, framealpha=1)

    ficxs_ists = ficxs_pqt[ch].unstack().loc[tmin:tmax, bes_w[0]:bes_w[1]].sum(1)
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
    ax[1].set_xlabel('Time [ms]', fontsize=14)
    ax[1].set_ylabel('Power [MW]', fontsize=14)
    ax[1].legend(
        loc='upper left', bbox_to_anchor=(1.01, 1.02), fontsize=12,
        framealpha=1)
    return fig, ax

def main(args):
    shot = args.shot
    los = args.los
    verbose = args.verbose
    overwrite = args.overwrite
    ut = args.upper_threshold
    lt = args.lower_threshold
    make_spec_plot = args.plot_spectrum
    ch = args.ch
    t_ms_idx = args.t_ms_idx
    config_file = args.config
    species = args.species
    blue_shift = args.blue_shift
    png_file = args.png_file

    config = read_config(config_file=config_file)
    save_dir = config['save_dir']
    save_patt = config['save_patt']
    
    png_dir = config['png_dir']
    png_patt = config['png_patt']
    
    if verbose: print(
        f'Calculating diagnostic beam timing for LHD discharge #{shot} '
        f'FICXS {los}-O LOS.')
    run_prog = False
    save_file = os.path.abspath(
        save_dir + '/' + save_patt.format(shot=shot, los=los) )
    if os.path.exists(save_file) and not overwrite:
        if verbose: success(
            f'File [{save_file}] already exists.\nIf you would like to '
            f'overwrite the data, rerun the command with -o.')
    elif not os.path.exists(save_file):
        if verbose: print(
            f'File [{save_file}] does not exist. '
            f'Calculating beam timing.')
        run_prog = True
    elif overwrite:
        if verbose: print(f'Overwriting data in file [{save_file}].')
        run_prog = True

    fmap = 'fmap'
    if run_prog or make_spec_plot:
        shots_fibermaps = read_fibermaps_log()
        data_dir = config['data_dir']
        if shot not in shots_fibermaps:
            pqt_dir = config['pqt_dir']
            ficxs_pqt_file = pqt_dir + '/' + \
                config['ficxs_pqt_patt'].format(shot=shot)
            ficxs_pqt = read_ficxs_from_pqt(ficxs_pqt_file, convert_to_ms=True)
            ficxs_pqt_t_idx = ficxs_pqt.index.droplevel('wavelength').unique()
            file = data_dir + '/' + \
                config['nbi_patts']['nb3'].format(shot=shot)
            ebeam = read_data_basic(
                file, use_postgres_names=True, convert_to_ms=True
            )[f'ebeam_nb3']
            ebeam_nz = ebeam[ebeam > 0]
            if ebeam_nz.size == 0:
                error(
                    f'NB3 is not on for {shot}, cannot determine fibermap.',
                    stop=True)
            ebeam_ts = [
                t for t in ebeam_nz.index.values if t in ficxs_pqt_t_idx]
            t_idx = ebeam_nz.loc[ebeam_ts].idxmax()
            ficxs_at_t = ficxs_pqt.loc[t_idx]
            fmap = determine_fibermap(ficxs_at_t, species)
            write_fibermaps_log({shot:fmap})
        else:
            fmap = shots_fibermaps[shot]
        fibers = read_fibermaps()[fmap]
        available_chs = fibers[fibers['los'] == los].index.values
        if ch is None:
            ch = fibers.loc[available_chs, 'r'].apply(
                lambda x:np.abs(x - 3.6)).idxmin()
        if ch not in available_chs:
            error(
                f'Channel {ch} is not available for {los}-O LOS. '
                f'{available_chs = }',
                stop=True)
        dnb = 10 - los

        ebeams = []
        dnb_keys = [f'nb{dnb}'] if dnb == 3 else [f'nb{dnb}a', f'nb{dnb}b']
        for dnb_key in dnb_keys:
            file = data_dir + '/' + \
                config['nbi_patts'][dnb_key].format(shot=shot)
            ebeam = read_data_basic(
                file, use_postgres_names=True, convert_to_ms=True
            )[f'ebeam_{dnb_key}']
            ebeam_nz = ebeam[ebeam > 0]
            if ebeam_nz.size == 0:
                continue
            else:
                ebeams.append(ebeam_nz.mean())
        if len(ebeams) == 0:
            error(
                f'No measured injected energy for DNB{dnb} for {shot}, '
                f'cannot determine BES.',
                stop=True)
        einj = max(ebeams)

        bes_df = find_bes_wavelengths(
            shot, los, fmap, einj, species=species, blue_shift=blue_shift,
            config=config)
        bes_ser = bes_df.loc[ch]
        if los == 6:
            bes_wmin = bes_ser['third'].min()
            bes_wmax = bes_ser['full'].max()
        else:
            bes_wmin = bes_ser['full'].mean() - 0.33
            bes_wmax = bes_ser['full'].mean() + 0.33
        bes_w = [bes_wmin, bes_wmax]

    if run_prog:
        ficxs_ts_ser = calculate_ficxs_timing(
            shot, los, ch, bes_w, verbose=verbose, config=config,
            ficxs_upper_threshold=ut, ficxs_lower_threshold=lt)
        write_ficxs_timing(ficxs_ts_ser, save_file)
        if verbose: success(
            f'Timing data saved to file [{save_file}]')
        
    if make_spec_plot:
        if not run_prog:
            ficxs_ts_ser = read_ficxs_timing(save_file)
        
        if t_ms_idx is None:
            t_ms_idxs = ficxs_ts_ser.index.droplevel('t_ms').unique()
            t_ms_idx = t_ms_idxs[-2]
        fig, ax = plot_ficxs_cycle(
            shot, los, ch, bes_w, ficxs_ts_ser.loc[t_ms_idx], config=config)
        if png_file is None:
            png_file = os.path.abspath(
                png_dir + '/' + \
                png_patt.format(shot=shot, los=los, ch=ch, t_ms_idx=t_ms_idx) )
        fig.savefig(png_file, format='png', dpi=120)

    return 0        
    
def argparser():
    parser = argparse.ArgumentParser(
        description=(
            'Calculate DNB timing for FICXS system and save to a text file.'))
    parser.add_argument('shot', type=int, help='Discharge number.')
    parser.add_argument('los', type=int, help='FICXS line-of-sight.')
    parser.add_argument(
        '-o', '--overwrite', action='store_true', help='Overwrite text file.')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbose output.')
    parser.add_argument(
        '-u', '--upper-threshold', type=float, default=0.8,
        help=(
            'DNB on signals are taken above this threshold as a percentage of'
            ' the maximum signal within a DNB cycle.'))
    parser.add_argument(
        '-l', '--lower-threshold', type=float, default=0.2,
        help=(
            'DNB off signals are taken below this threshold as a percentage of'
            ' the maximum signal within a DNB cycle.'))
    parser.add_argument(
        '-ps', '--plot-spectrum', action='store_true',
        help='Plot wavelength spectrum and DNB cycle.')
    parser.add_argument(
        '-ch', '--ch', type=int, default=None, help='FICXS channel to plot.')
    parser.add_argument(
        '-t', '--t-ms-idx', type=int, default=None,
        help='Time stamp index for DNB cycle to plot.')
    parser.add_argument(
        '-c', '--config', default=None, help='Config file.')
    parser.add_argument(
        '-s', '--species', default='d',
        help='Plasma thermal ion majority species ["h", "d"].')
    parser.add_argument(
        '-b', '--blue-shift', action='store_true',
        help='Set to true if using blue-shifted signal.')
    parser.add_argument(
        '-f', '--png-file', default=None, help='Save file for FICXS plot.')

    return parser

if __name__ == '__main__':
    args = argparser().parse_args()
    main(args)