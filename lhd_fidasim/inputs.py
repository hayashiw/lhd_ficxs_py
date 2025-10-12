import configparser
import numpy as np
import os

from lhd_ficxs_py.lhd_fidasim.geometry import read_bgrids
from lhd_ficxs_py.io import read_config
from lhd_ficxs_py import read_data
from lhd_ficxs_py.utils import error

FDIR = os.path.dirname(__file__)
INPUTS_CONFIG = 'inputs.ini'

def read_bgrid_from_bgrids_file(
    dnb: int,
    port: str,
    nx: int=None,
    ny: int=None,
    nz: int=None,
    dl: float=5.0,
    geo_dir: str=None
) -> dict:
    """
    """
    bgrids = read_bgrids(geo_dir=geo_dir)
    bgrids = bgrids[bgrids['name'] == f'nb{dnb}{port}']

    bgrid = {}
    for name in bgrids.dtype.names:
        if name == 'name': continue
        if 'orig' in name: continue
        bgrid[name] = bgrids[name][0]
    bgrid['origin'] = np.array(list(bgrids[['orig_x', 'orig_y', 'orig_z']][0]))

    for n, label in zip([nx, ny, nz], ['x', 'y', 'z']):
        if n is None:
            val_min = bgrid[f'{label}min']
            val_max = bgrid[f'{label}max']
            delta = val_max - val_min
            n = int(np.ceil(delta / dl))
        bgrid[f'n{label}'] = n
    return bgrid

def read_inputs_from_ini() -> dict:
    inputs_ini = {}
    config_obj = configparser.ConfigParser()
    config_obj.read(INPUTS_CONFIG)
    for key, val in config_obj['inputs'].items():
        try:
            inputs_ini[key] = eval(val)
        except:
            inputs_ini[key] = val
    return inputs_ini

def make_inputs(
    shot: int,
    time_ms: int,
    dnb: int,
    port: str,
    config: dict=None,
    ab: float=None,
    nx: int=None,
    ny: int=None,
    nz: int=None,
    dl: float=5.0,
    out_dir: str='./',
    geo_dir: str=None,
    write_nml: bool=False
) -> None:
    """
    """
    from fidasim.preprocessing import check_inputs, write_namelist
    out_dir = os.path.abspath(out_dir)
    if config is None: config = read_config()

    runid = f'{shot}t{time_ms}{dnb}{port}LHDFICXS'
    inputs = dict(
        shot=shot, time=time_ms/1e3, runid=runid,
        comment=f'LHD-FICXS with DNB{dnb}-{port.upper()}',
        result_dir=os.path.abspath(out_dir),
        tables_file=config['tables_file'] )
    inputs.update(read_inputs_from_ini())
    
    bgrid = read_bgrid_from_bgrids_file(
        dnb, port, nx=nx, ny=ny, nz=nz, dl=dl, geo_dir=geo_dir)
    inputs.update(bgrid)

    data_dir = config['data_dir']
    if ab is None:
        ha3_patt = config['ha3_patt']
        ha3_file = data_dir + '/' + ha3_patt.format(shot=shot)
        ha3 = read_data(
            ha3_file, use_postgres_names=True)
        print(f'Input time {time_ms} ms')
        ts = ha3.index.values
        loc_t = ts[np.abs(ts*1e3 - time_ms).argmin()]
        print(f'HA3 time {loc_t*1e3:.0f} ms')
        deut_ratio = ha3.loc[loc_t-0.1:loc_t, 'd/(h+d)']
        deut_ratio = deut_ratio[deut_ratio > 0].mean()
        if deut_ratio > 0.5:
            print('Main ion plasma species is [deuterium]')
            ab = 2.014
        else:
            print('Main ion plasma species is [hydrogen]')
            ab = 1.007
    inputs['ab'] = ab

    if dnb <= 3:
        current_fractions = np.array([1.0, 0.0, 0.0])
        dnb_patt = config['nbi_patts'][f'nb{dnb}']
        dnb_file = data_dir + '/' + dnb_patt.format(shot=shot)
        nbi = read_data(dnb_file, use_postgres_names=True)
        print(f'Input time {time_ms} ms')
        ts = nbi.index.values
        loc_t = ts[np.abs(ts*1e3 - time_ms).argmin()]
        print(f'DNB time {loc_t*1e3:.0f} ms')
        nbi = nbi.loc[loc_t-0.1:loc_t+0.1]
        ists = nbi[f'ion_stsa_nb{dnb}'].max()
        if ists == 0:
            error(
                f'DNB{dnb}-{port.upper()} is not on for {shot} ({time_ms} ms)',
                stop=True )
        total_ists = nbi[[f'ion_stsa_nb{dnb}',
                          f'ion_stsb_nb{dnb}']].sum(1).max()
        
        dnb_patt = config['nnb_calib'][f'nb{dnb}']
        dnb_file = data_dir + '/' + dnb_patt.format(shot=shot)
        nbi = read_data(dnb_file, use_postgres_names=True)
        print(f'Input time {time_ms} ms')
        ts = nbi.index.values
        loc_t = ts[np.abs(ts*1e3 - time_ms).argmin()]
        print(f'NNB-calib time {loc_t*1e3:.0f} ms')
        nbi = nbi.loc[loc_t-0.1:loc_t+0.1]

        pinj = nbi[f'pport_through_nb{dnb}']
        pinj = pinj[pinj > 0] / 1e3 # kW -> MW
        if pinj.size == 0:
            error(
                f'DNB{dnb}-{port.upper()} Pinj = 0 for {shot} ({time_ms} ms)',
                stop=True )
        pinj = pinj.mean() / total_ists

        einj = nbi[f'ebeam_nb{dnb}']
        einj = einj[einj > 0]
        if einj.size == 0:
            error(
                f'DNB{dnb}-{port.upper()} Einj = 0 for {shot} ({time_ms} ms)',
                stop=True )
        einj = einj.mean()
    else:
        dnb_patt = config['nbi_patts'][f'nb{dnb}{port[1]}']
        dnb_file = data_dir + '/' + dnb_patt.format(shot=shot)
        nbi = read_data(dnb_file, use_postgres_names=True)
        print(f'Input time {time_ms} ms')
        ts = nbi.index.values
        loc_t = ts[np.abs(ts*1e3 - time_ms).argmin()]
        print(f'DNB time {loc_t*1e3:.0f} ms')
        nbi = nbi.loc[loc_t-0.1:loc_t+0.1]
        pinj_ul = nbi[f'pport_through_nb{dnb}{port[1]}']
        pinj_ul = pinj_ul[pinj_ul > 0]
        if pinj_ul.size == 0:
            error(
                f'DNB{dnb}-{port.upper()} Pinj = 0 for {shot} ({time_ms} ms)',
                stop=True )
        pinj_ul = pinj_ul.mean()

        einj = nbi[f'ebeam_nb{dnb}{port[1]}']
        einj = einj[einj > 0]
        if einj.size == 0:
            error(
                f'DNB{dnb}-{port.upper()} Einj = 0 for {shot} ({time_ms} ms)',
                stop=True )
        einj = einj.mean()

        dnb_patt = config['pnb_frac'][f'nb{dnb}{port[1]}']
        dnb_file = data_dir + '/' + dnb_patt.format(shot=shot)
        nbi = read_data(dnb_file, use_postgres_names=True)
        print(f'Input time {time_ms} ms')
        ts = nbi.index.values
        loc_t = ts[np.abs(ts*1e3 - time_ms).argmin()]
        print(f'PNB-frac time {loc_t*1e3:.0f} ms')
        nbi = nbi.loc[loc_t-0.1:loc_t+0.1]
        pinj_frac_ul = nbi[f'fraction_power_#4{port}']
        pinj_frac_ul = pinj_frac_ul[pinj_frac_ul > 0]
        if pinj_frac_ul.size == 0:
            error(
                f'DNB{dnb}-{port.upper()} Pinj (% of upper and lower ports) = '
                f'0 for {shot} ({time_ms} ms)', stop=True )
        pinj_frac_ul = pinj_frac_ul.mean()
        pinj = pinj_ul * pinj_frac_ul

        curr_frac_1 = nbi[f'fraction_1st_#4{port}']
        curr_frac_2 = nbi[f'fraction_2nd_#4{port}']
        curr_frac_3 = nbi[f'fraction_3rd_#4{port}']
        curr_frac_1 = curr_frac_1[curr_frac_1 > 0]
        curr_frac_2 = curr_frac_2[curr_frac_2 > 0]
        curr_frac_3 = curr_frac_3[curr_frac_3 > 0]
        if curr_frac_1.size + curr_frac_2.size + curr_frac_3.size == 0:
            error(
                f'DNB{dnb}-{port.upper()} Current fractions are all 0 for '
                f'{shot} ({time_ms} ms)', stop=True )
        curr_frac_1 = curr_frac_1.mean()
        curr_frac_2 = curr_frac_2.mean()
        curr_frac_3 = curr_frac_3.mean()
        current_fractions = np.array([curr_frac_1, curr_frac_2, curr_frac_3])
        current_fractions = current_fractions / current_fractions.sum()

    inputs['pinj'] = pinj
    inputs['einj'] = einj
    inputs['current_fractions'] = current_fractions

    if write_nml:
        if geo_dir is None: geo_dir = FDIR + '/geometry_files'
        inputs_file = out_dir + f'/{runid}_inputs.dat'
        geo_file = geo_dir + f'/nb{dnb}{port}LHDFICXS_geometry.h5'
        equi_file = out_dir + f'/{shot}t{time_ms}_equilibrium.h5'
        dist_file = out_dir + f'/{shot}t{time_ms}_distribution.h5'
        inputs = check_inputs(inputs)
        inputs['geometry_file'] = geo_file
        inputs['equilibrium_file'] = equi_file
        inputs['distribution_file'] = dist_file
        write_namelist(inputs_file, inputs)

    return inputs