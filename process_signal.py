import numpy as np
import pandas as pd

from typing import Tuple

from .io import (
    error, read_geometry, read_fibermaps, read_config,
    read_full_nbists_from_config )
from .physics import DEUT_MASS_KG, HYDR_MASS_KG, D_ALPHA, H_ALPHA, J_PER_EV
from .physics import (
    convert_kinetic_energy_to_Doppler_shift as convert_J_to_nm)
from scipy.integrate import simpson

def determine_fibermap(
    ficxs_at_t: pd.DataFrame,
    species: str='d'
) -> str:
    """
    Determine the fibermap ['6OFIDA', 'PNBFIDA', 'NNBFIDA'] from
    measured FICXS.

    Parameters
    ----------
    ficxs_at_t : pd.DataFrame
        Pandas dataframe containing snapshot of FICXS data at a time
        stamp when NB3 is on. Columns are channels and rows are
        wavelength.
    species : {'h', 'd'}, optional
        Species label. Must be 'h' for hydrogen or 'd' for deuterium.
        Default is 'd'.

    Returns
    -------
    fmap : {'6OFIDA', 'PNBFIDA', 'NNBFIDA'}
        Label for fibermap configuration.
    """
    if ficxs_at_t.columns.size != 16:
        error(
            f'ficxs_at_t should have 16 columns, '
            f'{ficxs_at_t.columns.size = }',
            stop=True)
    if species not in ['h', 'd']:
        error(f'Species must be "h" or "d", {species = }', stop=True)

    bes_w = [664.5, 665.5] if species == 'd' else [667.5, 668.5]
    fida_w = [661.5, 662.5] if species == 'd' else [663.5, 664.5]

    x_w = ficxs_at_t.index.values
    dx = np.diff(x_w).mean()
    ch_los = {}
    for ch in range(1, 17):
        bes = ficxs_at_t.loc[bes_w[0]:bes_w[1], ch].sum()*dx
        fida = ficxs_at_t.loc[fida_w[0]:fida_w[1], ch].sum()*dx
        ratio = bes/fida
        los = 6 if np.log(ratio) < 1 else 7
        ch_los[ch] = los
    if all([ch_los[ch] == 6 for ch in range(1, 17)]):
        fmap = '6OFIDA'
    elif (
        all([ch_los[ch] == 7 for ch in range(1, 5)]) and
        all([ch_los[ch] == 6 for ch in range(5, 17)])
    ):
        fmap = 'PNBFIDA'
    elif (
        all([ch_los[ch] == 7 for ch in range(1, 11)]) and
        all([ch_los[ch] == 6 for ch in range(11, 17)])
    ):
        fmap = 'NNBFIDA'
    else:
        error(
            f'Cannot determine fibermap from ch counts, {ch_los = }',
            stop=True)
        
    return fmap

def find_bes_wavelengths(
    shot: int,
    los: int,
    fmap: str,
    einj: float,
    species: str='d',
    blue_shift: bool=False,
    config: dict=None,
    fibermaps_file: str=None,
    geometry_file: str=None
) -> pd.DataFrame:
    """
    Find the Doppler-shifted wavelength for beam emission features for
    full, half, and third energy compoonents.

    Parameters
    ----------
    shot : int
        Shot number.
    fmap : {'6OFIDA', 'PNBFIDA', 'NNBFIDA'}
        Fibermap label.
    los : {6, 7}
        FICXS LOS label.
    einj : float
        Injected energy of diagnostic beam in kiloelectronvolts (keV).
    species : {'h', 'd'}, optional
        Species label. Must be 'h' for hydrogen or 'd' for deuterium.
        Default is 'd'.
    blue_shift : bool, optional
        Set to `True` if source is moving toward detector.
    config: dict, optional
        Dictionary containing program configuration.
    fibermaps_file : str, optional
        FICXS fiber configurations file path. Default is fibermaps.ini
        in package directory.
    log_file : str, optional
        Fibermaps log file. Default is fibermaps.log in package
        directory.

    returns
    -------
    bes_df : pd.DataFrame
        Pandas dataframe containing BES component wavelengths.
    """
    if species not in ['h', 'd']:
        error(f'Species must be "h" or "d", {species = }', stop=True)
    if los not in [6, 7]:
        error(f'LOS must be 6 or 7, {los = }', stop=True)
    if config is None: config = read_config()

    mass = DEUT_MASS_KG if species == 'd' else HYDR_MASS_KG
    cold_alpha = D_ALPHA if species == 'd' else H_ALPHA
    einj_J = (einj*1e3) * J_PER_EV

    dnb = 10 - los
    ports = ['a', 'b'] if dnb == 3 else ['ua', 'la', 'ub', 'lb']
    ists_df = read_full_nbists_from_config(
        shot, config=config
    )[[f'nb{dnb}{port}' for port in ports]]
    ports = [port for port in ports if ists_df[f'nb{dnb}{port}'].max() == 1]
    
    fibers = read_fibermaps(fibermaps_file=fibermaps_file)[fmap]
    available_chs = fibers[fibers['los'] == los].index.values

    nbi_geo_full = read_geometry('nbi_geo', geometry_file=geometry_file)
    nbi_geo = {
        f'nb{dnb}{port}':nbi_geo_full[f'nb{dnb}{port}'] for port in ports }

    los_geo = read_geometry(fmap, geometry_file=geometry_file)

    bes_df = pd.DataFrame(
        np.zeros((available_chs.size, 3*len(ports))),
        index=available_chs,
        columns=pd.MultiIndex.from_tuples([
            (comp, f'nb{dnb}{port}') for comp in ['third', 'half', 'full'] \
            for port in ports]) )
    bes_df.index.names = ['ch']
    for ch in available_chs:
        los_axis = los_geo[ch]['axis']
        los_unit = los_axis / np.linalg.norm(los_axis)
        for port in ports:
            nbi_axis = nbi_geo[f'nb{dnb}{port}']['axis']
            nbi_unit = nbi_axis / np.linalg.norm(nbi_axis)

            vec_para = (nbi_unit @ los_unit)**2
            einj_J_para = einj_J * vec_para
            w_third = convert_J_to_nm(
                einj_J_para/3, mass=mass, unshifted_wavelength=cold_alpha,
                blue_shift=blue_shift)  
            w_half = convert_J_to_nm(
                einj_J_para/2, mass=mass, unshifted_wavelength=cold_alpha,
                blue_shift=blue_shift)
            w_full = convert_J_to_nm(
                einj_J_para/1, mass=mass, unshifted_wavelength=cold_alpha,
                blue_shift=blue_shift)
            bes_df.loc[ch, ('third', f'nb{dnb}{port}')] = w_third
            bes_df.loc[ch, ('half', f'nb{dnb}{port}')] = w_half
            bes_df.loc[ch, ('full', f'nb{dnb}{port}')] = w_full

    return bes_df

def ficxs_bes_density(
    on_df: pd.DataFrame,
    off_df: pd.DataFrame,
    fida_w: list[float, float],
    bes_w: list[float, float]
) -> Tuple[float, float]:
    """
    Calculate BES-normalized FICXS signal with error propagation.

    Parameters
    ----------
    on_df : pd.DataFrame
        Pandas dataframe containing time (rows) and wavelength (columns)
        dependent FICXS data for one diagnostic beam "on" cycle and one
        channel.
    off_df : pd.DataFrame
        Pandas dataframe containing time (rows) and wavelength (columns)
        dependent FICXS data for one diagnostic beam "off" cycle and one
        channel.
    fida_w : list
        Wavelength range of FID(H)A feature of interest in nanometers
        (nm).
    bes_w : list
        Wavelength range of BES feature in nanometers (nm).

    Returns
    -------
    y : float
        BES-normalized FICXS signal.
    e : float
        Measured error of signal due to propagation of variance in time.

    Notes
    -----
    Wavelength integration of measured FICXS relies on `simpson` from 
    `scipy.integrate`.

    Error propagation is calculated according to Taylor[1]_.

    References
    ----------
    .. [1] J. R. Taylor, "An Introduction to Error Analysis," University
        Science Books, 2nd ed., 1997.
    """
    on_fida  = on_df.loc[:, fida_w[0]:fida_w[1]]
    off_fida = off_df.loc[:, fida_w[0]:fida_w[1]]
    on_bes   = on_df.loc[:, bes_w[0]:bes_w[1]]
    off_bes  = off_df.loc[:, bes_w[0]:bes_w[1]]
    x_fida = on_fida.column.values
    x_bes  = on_bes.column.values
    active_fida  = simpson(on_fida,  x=x_fida, axis=-1)
    passive_fida = simpson(off_fida, x=x_fida, axis=-1)
    active_bes   = simpson(on_bes,   x=x_bes,  axis=-1)
    passive_bes  = simpson(off_bes,  x=x_bes,  axis=-1)
    net_fida = active_fida.mean() - passive_fida.mean()
    net_bes  = active_bes.mean()  - passive_bes.mean()
    if net_fida <= 0: net_fida = np.nan
    if net_bes <= 0: net_bes = np.nan
    y = net_fida / net_bes
    act_fida_err = active_fida.std()
    pas_fida_err = passive_fida.std()
    net_fida_err = np.sqrt(act_fida_err**2 + pas_fida_err**2)
    act_bes_err  = active_bes.std()
    pas_bes_err  = passive_bes.std()
    net_bes_err  = np.sqrt(act_bes_err**2 + pas_bes_err**2)
    e = np.abs(y) * np.sqrt(
        (net_fida_err/net_fida)**2 +
        (net_bes_err /net_bes )**2 )
    return y, e

def construct_ficxs_density(
    on_df_chs: pd.DataFrame,
    off_df_chs: pd.DataFrame,
    ch_rs: pd.Series,
    fida_w: list,
    bes_df: pd.Series
) -> pd.DataFrame:
    """
    Construct a radial FICXS density (BES-normalized) profile.

    Parameters
    ----------
    on_df_chs : pd.DataFrame
        Pandas dataframe containing time and wavelength (rows) FICXS
        data per channel (columns) for one diagnostic beam "on" cycle.
    off_df_chs : pd.DataFrame
        Pandas dataframe containing time and wavelength (rows) FICXS
        data per channel (columns) for one diagnostic beam "off" cycle.
    ch_rs : pd.Series
        Pandas series containing sightline radii indexed by channels.
        Note that the channels in this series should be filtered for the
        desired FICXS sightline (6-O or 7-O LOS).
    fida_w: list
        Wavelength range of FID(H)A feature of interest in nanometers
        (nm).
    bes_df: pd.DataFrame
        Pandas series containing BES wavelength ranges indexed by
        channels.

    Returns
    -------
    ficxs_dens : pd.DataFrame
        Pandas dataframe containing sightline radii, FICXS density
        signal, and FICXS density error.
    """
    chs = ch_rs.index.values
    ficxs_dens = pd.DataFrame(
        np.zeros((chs.size, 3)),
        index=chs,
        columns=['r', 'y', 'e'] )
    for ch, r in ch_rs.items():
        ficxs_dens.loc[ch, 'r'] = r
        on_df = on_df_chs[ch].unstack()
        off_df = off_df_chs[ch].unstack()
        bes_w = bes_df.loc[ch].values
        ficxs_dens.loc[ch, ['y', 'e']] = \
            ficxs_bes_density(on_df, off_df, fida_w, bes_w)
    return ficxs_dens
