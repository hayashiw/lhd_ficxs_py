import argparse
import numpy as np
import os
import sys

from netCDF4 import Dataset
from scipy.interpolate import CloughTocher2DInterpolator as interpolator
from scipy.interpolate import interp1d
from typing import Tuple

PKG_PAR_DIR = os.path.abspath('../..')
sys.path.insert(0, PKG_PAR_DIR)
from lhd_ficxs_py import read_data, read_config

def coeffs_from_tswpe_a99(
    tswpe_file: str,
    time_ms: int
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Retrieve electron density and temperature fitting coefficients.

    Parameters
    ----------
    tswpe_file : str
        LHD `tswpe_a99` .dat file.
    time_ms : int
        Time stamp in milliseconds (ms).

    Returns
    -------
    cne : np.ndarray
        Electron density profile fitting coefficients in units of
        inverse cubic centimeters (cm-3).
    cte : np.ndarray
        Electron temperature profile fitting coefficients in units of
        kiloelectronvolts (keV).
    """
    tswpe = read_data(tswpe_file, use_postgres_names=True)

    print(f'Input time {time_ms} ms')
    ts = tswpe.index.values
    loc_t = ts[np.abs(ts*1e3 - time_ms).argmin()]
    tswpe = tswpe.loc[loc_t]
    print(f'TSWPE time {loc_t*1e3:.0f} ms')
    
    cne = tswpe.filter(regex=r'cne\d').sort_index().values
    cte = tswpe.filter(regex=r'cte\d').sort_index().values
    return cne, cte

def coeffs_from_lhdcxs7_nion_only5_fit(
    lhdcxs_file: str,
    time_ms: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Retrieve thermal ion density fitting coefficients.
    
    Parameters
    ----------
    lhdcxs_file : str
        LHD `lhdcxs7_nion_only5_fit` .dat file.
    time_ms : int
        Time stamp in milliseconds (ms).
        
    Returns
    -------
    cnd : np.ndarray
        Thermal deuterium density profile fitting coefficients in units
        of inverse cubic centimeters (cm-3).
    cnh : np.ndarray
        Thermal hydrogen density profile fitting coefficients in units
        of inverse cubic centimeters (cm-3).
    cnc : np.ndarray
        Thermal carbon density profile fitting coefficients in units of
        inverse cubic centimeters (cm-3).
    """
    lhdcxs = read_data(
        lhdcxs_file, use_postgres_names=True)
    
    print(f'Input time {time_ms} ms')
    ts = lhdcxs.index.values
    loc_t = ts[np.abs(ts*1e3 - time_ms).argmin()]
    lhdcxs = lhdcxs.loc[loc_t]
    print(f'LHDCXS7 time {loc_t*1e3:.0f} ms')
    
    cnd = lhdcxs.filter(regex=r'nd\d').sort_index().values
    cnh = lhdcxs.filter(regex=r'nh\d').sort_index().values
    # FIDASIM only takes in H and D as main ion species and only allows
    # one impurity species. The 1/9 factor combines the carbon and
    # helium impurities while keeping Zeff constant, assuming that
    # carbon is the main impurity.
    cnc = (lhdcxs.filter(regex=r'nc\d').sort_index().values +
           (1/9)*lhdcxs.filter(regex=r'nhe\d').sort_index().values)
    return cnd, cnh, cnc

def coeffs_from_cxswpi7(
    cxswpi_file: str,
    time_ms: int
) -> np.ndarray:
    r"""
    Retrieve thermal ion temperature fitting coefficients.

    Parameters
    ----------
    cxswpi_file : str
        LHD `cxswpi7` .dat file.
    time_ms : int
        Time stamp in milliseconds (ms).

    Returns
    -------
    cti : np.ndarray
        Thermal ion temperature profile fitting coefficients in units of
        kiloelectronvolts (keV).
    """
    cxswpi = read_data(
        cxswpi_file, use_postgres_names=True)
    cxswpi = cxswpi[cxswpi['cti0'] != 0]

    print(f'Input time {time_ms} ms')
    ts = cxswpi.index.values
    loc_t = ts[np.abs(ts*1e3 - time_ms).argmin()]
    cxswpi = cxswpi.loc[loc_t]
    print(f'CXSWPI7 time {loc_t*1e3:.0f} ms')

    cti = cxswpi.filter(regex=r'cti\d').sort_index().values
    return cti

def make_plasma_from_vmec(
    wout_file: str,
    time_ms: int,
    cne: np.ndarray,
    cnd: np.ndarray,
    cnh: np.ndarray,
    cnc: np.ndarray,
    cte: np.ndarray,
    cti: np.ndarray,
    nr: int=31,
    nz: int=29,
    ntheta: int=90,
    theta_min: float=0.0,
    theta_max: float=2*np.pi,
    n_per_fp: int=12,
    i_fp_min: int=6,
    i_fp_max: int=10,
) -> dict:
    r"""
    Make FIDASIM plasma input from VMEC wout file and plasma profile
    fitting coefficients.

    Parameters
    ----------
    wout_file : str
        Filepath to VMEC wout file.
    time_ms : int
        Time stamp in milliseconds.
    cne : np.ndarray
        Electron density profile fitting coefficients.
    cnd : np.ndarray
        Thermal deuterium density profile fitting coefficients.
    cnh : np.ndarray
        Thermal hydrogen density profile fitting coefficients.
    cnc : np.ndarray
        Thermal carbon density profile fitting coefficients.
    cte : np.ndarray
        Electron temperature profile fitting coefficients.
    cti : np.ndarray
        Thermal ion temperature profile fitting coefficients.
    nr : int, optional
        Size of radial grid for FIDASIM interpolation grid.
        Default is 31.
    nz : int, optional
        Size of vertical grid for FIDASIM interpolation grid.
        Default is 29.
    ntheta : int, optional
        Size of poloidal grid for VMEC. Default is 90.
    theta_min : float, optional
        Minimum value for VMEC poloidal grid. Default is 0.0.
    theta_max : float, optional
        Maximum value for VMEC poloidal grid. Default is 2pi.
    n_per_fp : int, optional
        Number of toroidal slices per field period. Default is 12.
    i_fp_min : int, optional
        Field period index number for phi min. Default is 6.
    i_fp_max : int, optional
        Field period index number for phi max. Default is 10.

    Returns
    -------
    plasma : dict
        FIDASIM-like plasma dictionary.
    """
    from fidasim.preprocessing import check_plasma

    with Dataset(wout_file, 'r', format='NETCDF4') as f:
        ns   = f['ns'][()]
        nfp  = f['nfp'][()]
        ixm  = f['xm'][()]
        ixn  = f['xn'][()]
        rmnc = f['rmnc'][()]
        zmns = f['zmns'][()]
    theta = np.linspace(theta_min, theta_max, ntheta)
    phi = np.linspace(0.0, 2*np.pi/nfp, n_per_fp)
    pol, tor = np.meshgrid(theta, phi)
    shape = (ns, n_per_fp, ntheta)
        
    pol_m = ixm[:, np.newaxis] @ pol.flatten()[np.newaxis, :]
    tor_n = ixn[:, np.newaxis] @ tor.flatten()[np.newaxis, :]
    cos_mu_nv = np.cos(pol_m - tor_n)
    sin_mu_nv = np.sin(pol_m - tor_n)
    r_svu = np.dot(rmnc, cos_mu_nv).reshape(shape)
    z_svu = np.dot(zmns, sin_mu_nv).reshape(shape)

    # Shift R and Z limits out by 5 cm
    rmin, rmax = r_svu.min()-0.05, r_svu.max()+0.05
    zmin, zmax = z_svu.min()-0.05, z_svu.max()+0.05

    r = np.linspace(rmin, rmax, nr)
    z = np.linspace(zmin, zmax, nz)
    z2d, r2d = np.meshgrid(z, r)

    s = np.linspace(0, 1, ns)
    rho = np.sqrt(s)
    ne_1d = np.polyval(cne[::-1], rho)
    te_1d = np.polyval(cte[::-1], rho)
    nd_1d = np.polyval(cnd[::-1], rho)
    nh_1d = np.polyval(cnh[::-1], rho)
    nc_1d = np.polyval(cnc[::-1], rho)
    ti_1d = np.polyval(cti[::-1], rho)
    ne_1d = np.where(ne_1d < 0, 0, ne_1d)
    te_1d = np.where(te_1d < 0, 0, te_1d)
    nd_1d = np.where(nd_1d < 0, 0, nd_1d)
    nh_1d = np.where(nh_1d < 0, 0, nh_1d)
    nc_1d = np.where(nc_1d < 0, 0, nc_1d)
    ti_1d = np.where(ti_1d < 0, 0, ti_1d)

    rho_1d = np.tile(rho, (ntheta, 1)).T.flatten()
    _pmask  = np.zeros((nr, nz, n_per_fp), dtype=int)
    _dene   = np.zeros((nr, nz, n_per_fp))
    _deni   = np.zeros((2, nr, nz, n_per_fp))
    _denimp = np.zeros((nr, nz, n_per_fp))
    _te     = np.zeros((nr, nz, n_per_fp))
    _ti     = np.zeros((nr, nz, n_per_fp))
    for iphi in range(n_per_fp):
        print(f'[{iphi+1}/{n_per_fp}]')
        r1d_svu = r_svu[:, iphi, :].flatten()
        z1d_svu = z_svu[:, iphi, :].flatten()
        rzdata = list(zip(r1d_svu, z1d_svu))

        s_to_rz = interpolator(rzdata, rho_1d)((r2d, z2d))
        fin_loc = ~np.isnan(s_to_rz)

        _pmask[..., iphi]            = fin_loc.astype(int)
        _dene[..., iphi][fin_loc]    = interp1d(rho, ne_1d)(s_to_rz[fin_loc])
        _deni[0, ..., iphi][fin_loc] = interp1d(rho, nh_1d)(s_to_rz[fin_loc])
        _deni[1, ..., iphi][fin_loc] = interp1d(rho, nd_1d)(s_to_rz[fin_loc])
        _denimp[..., iphi][fin_loc]  = interp1d(rho, nc_1d)(s_to_rz[fin_loc])
        _te[..., iphi][fin_loc]      = interp1d(rho, te_1d)(s_to_rz[fin_loc])
        _ti[..., iphi][fin_loc]      = interp1d(rho, ti_1d)(s_to_rz[fin_loc])
    
    if i_fp_min == i_fp_max:
        i_fp_max = i_fp_min + 1
    elif i_fp_min > i_fp_max:
        i_fp_min = i_fp_min - nfp
    delta_fp = i_fp_max - i_fp_min
    nphi = delta_fp*n_per_fp + 1
    phi = np.linspace(i_fp_min*2*np.pi/nfp, i_fp_max*2*np.pi/nfp, nphi)

    pmask  = np.zeros((nr, nz, nphi), dtype=int)
    dene   = np.zeros((nr, nz, nphi))
    deni   = np.zeros((2, nr, nz, nphi))
    denimp = np.zeros((nr, nz, nphi))
    te     = np.zeros((nr, nz, nphi))
    ti     = np.zeros((nr, nz, nphi))
    for i in range(delta_fp):
        pmask[..., i*n_per_fp:(i+1)*n_per_fp]   = _pmask
        dene[..., i*n_per_fp:(i+1)*n_per_fp]    = _dene
        deni[0, ..., i*n_per_fp:(i+1)*n_per_fp] = _deni[0]
        deni[1, ..., i*n_per_fp:(i+1)*n_per_fp] = _deni[1]
        denimp[..., i*n_per_fp:(i+1)*n_per_fp]  = _dene
        te[..., i*n_per_fp:(i+1)*n_per_fp]      = _te
        ti[..., i*n_per_fp:(i+1)*n_per_fp]      = _ti
    pmask[..., -1]    = _pmask[..., 0]
    dene[..., -1]     = _dene[..., 0]
    deni[0, ..., -1]  = _deni[0, ..., 0]
    deni[0, ..., -1]  = _deni[0, ..., 0]
    denimp[..., -1]   = _denimp[..., 0]
    te[..., -1]       = _te[..., 0]
    ti[..., -1]       = _ti[..., 0]
        
    zeff = (deni.sum(0) + 36*denimp) * np.divide(
        1, dene, out=np.zeros_like(dene), where=dene!=0)
    
    # Convert meters to centimeters
    fsim_igrid = dict(
        r=r*100, z=z*100, phi=phi,
        nr=nr, nz=nz, nphi=nphi,
        r2d=r2d*100, z2d=z2d*100 )

    denn = np.zeros_like(dene)
    vr, vz, vt = denn.copy(), denn.copy(), denn.copy()

    impurity_charge = 6
    nthermal = 2
    species_mass = np.array([1.007, 2.014])
    fsim_inputs = dict(time=time_ms/1e3)
    fsim_plasma = dict(
        data_source=wout_file, time=time_ms/1e3,
        dene=dene, deni=deni, denimp=denimp, denn=denn,
        te=te, ti=ti, zeff=zeff, mask=pmask,
        species_mass=species_mass, nthermal=nthermal,
        impurity_charge=impurity_charge,
        vr=vr, vz=vz, vt=vt)
    fsim_plasma = check_plasma(fsim_inputs, fsim_igrid, fsim_plasma)
    return fsim_plasma

def make_igrid_from_vmec(
    wout_file: str,
    nr: int=31,
    nz: int=29,
    n_per_fp: int=12,
    i_fp_min: int=6,
    i_fp_max: int=10,
) -> dict:
    r"""
    Make FIDASIM interpolation grid from VMEC wout file.

    Parameters
    ----------
    wout_file : str
        Filepath to VMEC wout file.
    nr : int, optional
        Size of radial grid for FIDASIM interpolation grid.
        Default is 31.
    nz : int, optional
        Size of vertical grid for FIDASIM interpolation grid.
        Default is 29.
    n_per_fp : int, optional
        Number of toroidal slices per field period. Default is 12.
    i_fp_min : int, optional
        Field period index number for phi min. Default is 6.
    i_fp_max : int, optional
        Field period index number for phi max. Default is 10.

    Returns
    -------
    igrid : dict
        FIDASIM-like interpolation grid dictionary.
    """
    with Dataset(wout_file, 'r', format='NETCDF4') as f:
        ns = f['ns'][()]
        nfp = f['nfp'][()]
        ixm = f['xm'][()]
        ixn = f['xn'][()]
        rmnc = f['rmnc'][()]
        zmns = f['zmns'][()]
        
    theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    _phi = np.linspace(0.0, 2*np.pi/nfp, n_per_fp)
    pol, tor = np.meshgrid(theta, _phi)

    pol_m = ixm[:, np.newaxis] @ pol.flatten()[np.newaxis, :]
    tor_n = ixn[:, np.newaxis] @ tor.flatten()[np.newaxis, :]
    cos_mu_nv = np.cos(pol_m - tor_n)
    sin_mu_nv = np.sin(pol_m - tor_n)
    r_svu = (rmnc @ cos_mu_nv).reshape((ns, n_per_fp, 4))
    z_svu = (zmns @ sin_mu_nv).reshape((ns, n_per_fp, 4))

    # Shift R and Z limits out by 5 cm
    rmin, rmax = r_svu.min()-0.05, r_svu.max()+0.05
    zmin, zmax = z_svu.min()-0.05, z_svu.max()+0.05
    
    if i_fp_min == i_fp_max:
        i_fp_max = i_fp_min + 1
    elif i_fp_min > i_fp_max:
        i_fp_min = i_fp_min - nfp
    delta_fp = i_fp_max - i_fp_min
    nphi = delta_fp*n_per_fp + 1
    phi = np.linspace(i_fp_min*2*np.pi/nfp, i_fp_max*2*np.pi/nfp, nphi)

    r = np.linspace(rmin, rmax, nr)
    z = np.linspace(zmin, zmax, nz)
    z2d, r2d = np.meshgrid(z, r)
    # Convert meters to centimeters
    igrid = dict(
        r=r*100, z=z*100, phi=phi,
        nr=nr, nz=nz, nphi=nphi,
        r2d=r2d*100, z2d=z2d*100 )
    return igrid

def make_fields_from_vmec(
    wout_file: str,
    time_ms: int,
    nr: int=31,
    nz: int=29,
    ntheta: int=90,
    theta_min: int=10,
    theta_max: int=10,
    n_per_fp: int=12,
    i_fp_min: int=6,
    i_fp_max: int=10,
) -> dict:
    r"""
    Make FIDASIM fields input from VMEC wout file.

    Parameters
    ----------
    wout_file : str
        Filepath to VMEC wout file.
    time_ms : int
        Time stamp in milliseconds.
    nr : int, optional
        Size of radial grid for FIDASIM interpolation grid.
        Default is 31.
    nz : int, optional
        Size of vertical grid for FIDASIM interpolation grid.
        Default is 29.
    ntheta : int, optional
        Size of poloidal grid for VMEC. Default is 90.
    theta_min : float, optional
        Minimum value for VMEC poloidal grid. Default is 0.0.
    theta_max : float, optional
        Maximum value for VMEC poloidal grid. Default is 2pi.
    n_per_fp : int, optional
        Number of toroidal slices per field period. Default is 12.
    i_fp_min : int, optional
        Field period index number for phi min. Default is 6.
    i_fp_max : int, optional
        Field period index number for phi max. Default is 10.

    Returns
    -------
    fields : dict
        FIDASIM-like fields dictionary.
    """
    from fidasim.preprocessing import check_fields
    print(f'Reading in VMEC data from {os.path.abspath(wout_file)}')

    with Dataset(wout_file, 'r', format='NETCDF4') as f:
        ns      = f['ns'][()]
        nfp     = f['nfp'][()]
        ixm     = f['xm'][()]
        ixn     = f['xn'][()]
        ixm_nyq = f['xm_nyq'][()]
        ixn_nyq = f['xn_nyq'][()]
        rmnc    = f['rmnc'][()]
        zmns    = f['zmns'][()]
        bsmns   = f['bsubsmns'][()]
        bvmnc   = f['bsubvmnc'][()]
        bumnc   = f['bsubumnc'][()]
    phi = np.linspace(0.0, 2*np.pi/nfp, n_per_fp)
    theta = np.linspace(theta_min, theta_max, ntheta)
    pol, tor = np.meshgrid(theta, phi)
    shape = (ns, n_per_fp, ntheta)
        
    print(f'Constructing B grid in VMEC coordinates')
    pol_m = ixm[:, np.newaxis] @ pol.flatten()[np.newaxis, :]
    tor_n = ixn[:, np.newaxis] @ tor.flatten()[np.newaxis, :]
    cos_mu_nv = np.cos(pol_m - tor_n)
    sin_mu_nv = np.sin(pol_m - tor_n)
    r_svu = np.dot(rmnc, cos_mu_nv).reshape(shape)
    z_svu = np.dot(zmns, sin_mu_nv).reshape(shape)

    s = np.linspace(0, 1, ns)
    drds = np.dot(
        np.gradient(rmnc, s, axis=0), cos_mu_nv).reshape(shape)
    drdv = np.dot( rmnc * ixn, sin_mu_nv).reshape(shape)
    drdu = np.dot(-rmnc * ixm, sin_mu_nv).reshape(shape)
    dzds = np.dot(
        np.gradient(zmns, s, axis=0), sin_mu_nv).reshape(shape)
    dzdv = np.dot(-zmns * ixn, cos_mu_nv).reshape(shape)
    dzdu = np.dot( zmns * ixm, cos_mu_nv).reshape(shape)

    pol_m = ixm_nyq[:, np.newaxis] @ pol.flatten()[np.newaxis, :]
    tor_n = ixn_nyq[:, np.newaxis] @ tor.flatten()[np.newaxis, :]
    cos_mu_nv = np.cos(pol_m - tor_n)
    sin_mu_nv = np.sin(pol_m - tor_n)
    bs = np.dot(bsmns, sin_mu_nv).reshape(shape)
    bv = np.dot(bvmnc, cos_mu_nv).reshape(shape)
    bu = np.dot(bumnc, cos_mu_nv).reshape(shape)

    den = drds*dzdu - drdu*dzds
    bnorm = np.divide(
        1, den, out=np.zeros_like(r_svu), where=den!=0)
    
    # Correct for the singularity on the magnetic axis in VMEC
    itheta_0  = np.abs(theta - 0.0).argmin()
    itheta_pi = np.abs(theta - np.pi).argmin()
    for iphi in range(n_per_fp):
        bu[0, iphi] = interp1d(
            np.append(-s[1:][::-1], s[1:]),
            np.append(bu[1:, iphi, itheta_pi][::-1], bu[1:, iphi, itheta_0])
        )([0])[0]
        bv[0, iphi] = interp1d(
            np.append(-s[1:][::-1], s[1:]),
            np.append(bv[1:, iphi, itheta_pi][::-1], bv[1:, iphi, itheta_0])
        )([0])[0]
        bnorm[0, iphi] = interp1d(
            np.append(-s[1:][::-1], s[1:]),
            np.append(
                bnorm[1:, iphi, itheta_pi][::-1], bnorm[1:, iphi, itheta_0])
        )([0])[0]

    br_svu = (dzdu*bs - dzds*bu) * bnorm
    bz_svu = (drds*bu - drdu*bs) * bnorm
    bt_svu = ((
        (bs*(drdu*dzdv - drdv*dzdu) + bu*(drdv*dzds - drds*dzdv)) * bnorm
    ) + bv) / r_svu

    # Shift R and Z limits out by 5 cm
    rmin, rmax = r_svu.min()-0.05, r_svu.max()+0.05
    zmin, zmax = z_svu.min()-0.05, z_svu.max()+0.05
    r = np.linspace(rmin, rmax, nr)
    z = np.linspace(zmin, zmax, nz)
    z2d, r2d = np.meshgrid(z, r)

    r1d = r2d.flatten()
    z1d = z2d.flatten()
    print(
        'Intepolating B to R, Z, phi coordinates and smoothing field outside')
    _br = np.zeros((nr, nz, n_per_fp))
    _bz = np.zeros((nr, nz, n_per_fp))
    _bt = np.zeros((nr, nz, n_per_fp))
    for iphi in range(n_per_fp):
        print(f'[{iphi+1}/{n_per_fp}]')
        r1d_svu = r_svu[:, iphi, :].flatten()
        z1d_svu = z_svu[:, iphi, :].flatten()
        rzdata = list(zip(r1d_svu, z1d_svu))
        
        br1d_svu = br_svu[:, iphi, :].flatten()
        bz1d_svu = bz_svu[:, iphi, :].flatten()
        bt1d_svu = bt_svu[:, iphi, :].flatten()

        br_rz = interpolator(rzdata, br1d_svu)((r2d, z2d))
        bz_rz = interpolator(rzdata, bz1d_svu)((r2d, z2d))
        bt_rz = interpolator(rzdata, bt1d_svu)((r2d, z2d))
        _br[..., iphi] = br_rz
        _bz[..., iphi] = bz_rz
        _bt[..., iphi] = bt_rz

        br_1d = br_rz.flatten()
        bz_1d = bz_rz.flatten()
        bt_1d = bt_rz.flatten()
        
        nan_idxs = np.where( np.isnan(br_1d))[0]
        fin_idxs = np.where(~np.isnan(br_1d))[0]
        for i in nan_idxs:
            ir, iz = np.unravel_index(i, r2d.shape)
            dsq = np.sqrt(
                (r1d[fin_idxs]-r1d[i])**2 + (z1d[fin_idxs]-z1d[i])**2)
            nearest = np.argsort(dsq)[:4]
            wgt = dsq[nearest] / dsq[nearest].sum()
            _br[ir, iz, iphi] = np.sum(br_1d[fin_idxs[nearest]] * wgt)
            _bz[ir, iz, iphi] = np.sum(bz_1d[fin_idxs[nearest]] * wgt)
            _bt[ir, iz, iphi] = np.sum(bt_1d[fin_idxs[nearest]] * wgt)

    if i_fp_min == i_fp_max:
        i_fp_max = i_fp_min + 1
    elif i_fp_min > i_fp_max:
        i_fp_min = i_fp_min - nfp
    delta_fp = i_fp_max - i_fp_min
    nphi = delta_fp*n_per_fp + 1
    phi = np.linspace(i_fp_min*2*np.pi/nfp, i_fp_max*2*np.pi/nfp, nphi)

    br = np.zeros((nr, nz, nphi))
    bz = np.zeros((nr, nz, nphi))
    bt = np.zeros((nr, nz, nphi))
    for i in range(delta_fp):
        br[..., i*n_per_fp:(i+1)*n_per_fp] = _br
        bz[..., i*n_per_fp:(i+1)*n_per_fp] = _bz
        bt[..., i*n_per_fp:(i+1)*n_per_fp] = _bt
    br[..., -1] = _br[..., 0]
    bz[..., -1] = _bz[..., 0]
    bt[..., -1] = _bt[..., 0]

    # Convert meters to centimeters
    fsim_igrid = dict(
        r=r*100, z=z*100, phi=phi,
        nr=nr, nz=nz, nphi=nphi,
        r2d=r2d*100, z2d=z2d*100 )

    bmask = (~np.isnan(br*bz*bt)).astype(int)
    efield = np.zeros(br.shape)
    fsim_inputs = dict(time=time_ms/1e3)
    fsim_fields = dict(
        data_source=wout_file, time=time_ms/1e3,
        br=br, bz=bz, bt=bt, mask=bmask,
        er=efield, ez=efield, et=efield )
    fsim_fields = check_fields(fsim_inputs, fsim_igrid, fsim_fields)
    return fsim_fields

def make_equilibrium(
    shot: int,
    time_ms: int,
    out_dir: str='./',
    nr: int=31,
    nz: int=29,
    ntheta: int=90,
    theta_min: float=0.0,
    theta_max: float=2*np.pi,
    n_per_fp: int=12,
    i_fp_min: int=6,
    i_fp_max: int=10,
) -> None:
    r"""
    Make FIDASIM fields input from VMEC wout file.

    Parameters
    ----------
    shot : int
        Shot number.
    time_ms : int
        Time stamp in milliseconds (ms).
    out_dir : str, optional
        Directory to output FIDASIM equilibrium .h5 file.
    nr : int, optional
        Size of radial grid for FIDASIM interpolation grid.
        Default is 31.
    nz : int, optional
        Size of vertical grid for FIDASIM interpolation grid.
        Default is 29.
    ntheta : int, optional
        Size of poloidal grid for VMEC. Default is 90.
    theta_min : float, optional
        Minimum value for VMEC poloidal grid. Default is 0.0.
    theta_max : float, optional
        Maximum value for VMEC poloidal grid. Default is 2pi.
    n_per_fp : int, optional
        Number of toroidal slices per field period. Default is 12.
    i_fp_min : int, optional
        Field period index number for phi min. Default is 6.
    i_fp_max : int, optional
        Field period index number for phi max. Default is 10.
    """
    from fidasim.preprocessing import write_equilibrium
    out_dir = os.path.abspath(out_dir)
    equi_file = out_dir + f'/{shot}t{time_ms}_equilibrium.h5'

    config = read_config()

    ierr = 0
    vmec_dir = config['vmec_dir']
    wout_patt = config['wout_patt']
    wout_file = vmec_dir + '/' + wout_patt.format(shot=shot, time_ms=time_ms)
    if not os.path.exists(wout_file):
        base = os.path.basename(wout_file)
        print(f'[{base}] \033[0;31mMissing wout file\033[0m')
        ierr = 1
    
    data_dir = config['data_dir']
    tswpe_patt  = config['tswpe_patt']
    lhdcxs_patt = config['lhdcxs_patt']
    cxswpi_patt = config['cxswpi_patt']
    tswpe_file  = data_dir + '/' + tswpe_patt.format(shot=shot)
    lhdcxs_file = data_dir + '/' + lhdcxs_patt.format(shot=shot)
    cxswpi_file = data_dir + '/' + cxswpi_patt.format(shot=shot)
    for file in [tswpe_file, lhdcxs_file, cxswpi_file]:
        if not os.path.exists(file):
            base = os.path.basename(file)
            print(f'[{base}] \033[0;31mMissing file\033[0m')
            ierr = 1

    if ierr: raise FileNotFoundError('Missing input files')

    fields = make_fields_from_vmec(
        wout_file, time_ms, nr=nr, nz=nz,
        ntheta=ntheta, theta_min=theta_min, theta_max=theta_max,
        n_per_fp=n_per_fp, i_fp_min=i_fp_min, i_fp_max=i_fp_max )
    
    cne, cte = coeffs_from_tswpe_a99(tswpe_file, time_ms)
    cnd, cnh, cnc = coeffs_from_lhdcxs7_nion_only5_fit(lhdcxs_file, time_ms)
    cti = coeffs_from_cxswpi7(cxswpi_file, time_ms)
    plasma = make_plasma_from_vmec(
        wout_file, time_ms, cne, cnd, cnh, cnc, cte, cti, nr=nr, nz=nz,
        ntheta=ntheta, theta_min=theta_min, theta_max=theta_max,
        n_per_fp=n_per_fp, i_fp_min=i_fp_min, i_fp_max=i_fp_max )
    
    write_equilibrium(equi_file, plasma, fields)

def argparser():
    parser = argparse.ArgumentParser(
        description='Make FIDASIM equilibrium.h5 files' )
    parser.add_argument(
        'shot', type=int, help='Shot number.' )
    parser.add_argument(
        'time_ms', type=int, help='Time stamp in milliseconds (ms).')
    parser.add_argument(
        '-o', '--out-dir', default='./',
        help='Directory to output FIDASIM equilibrium .h5 file.' )
    parser.add_argument(
        '--nr', type=int, default=31, help=(
            'Size of radial grid for FIDASIM interpolation grid. '
            'Default is 31.' ))
    parser.add_argument(
        '--nz', type=int, default=29, help=(
            'Size of vertical grid for FIDASIM interpolation grid. '
            'Default is 29.' ))
    parser.add_argument(
        '-nth', '--ntheta', type=int, default=90,
        help='Size of poloidal grid for VMEC. Default is 90.' )
    parser.add_argument(
        '--theta-min', type=float, default=0.0,
        help='Minimum value for VMEC poloidal grid. Default is 0.0.' )
    parser.add_argument(
        '--theta-max', type=float, default=2*np.pi,
        help='Maximum value for VMEC poloidal grid. Default is 2pi.' )
    parser.add_argument(
        '--n-per-fp', type=int, default=12,
        help=f'Number of toroidal slices per field period. Default is 12.' )
    parser.add_argument(
        '--i-fp-min', type=int, default=6,
        help='Field period index number for phi min. Default is 6.' )
    parser.add_argument(
        '--i-fp-max', type=int, default=10,
        help='Field period index number for phi max. Default is 10.' )
    return parser

def main(args):
    shot = args.shot
    time_ms = args.time_ms
    out_dir = args.out_dir
    nr, nz = args.nr, args.nz
    ntheta = args.ntheta
    theta_min, theta_max = args.theta_min, args.theta_max
    n_per_fp = args.n_per_fp
    i_fp_min, i_fp_max = args.i_fp_min, args.i_fp_max

    kwargs = dict(
        out_dir=out_dir, nr=nr, nz=nz,
        ntheta=ntheta, theta_min=theta_min, theta_max=theta_max,
        n_per_fp=n_per_fp, i_fp_max=i_fp_max, i_fp_min=i_fp_min )
    make_equilibrium(shot, time_ms, **kwargs)

if __name__ == '__main__':
    args = argparser().parse_args()
    main(args)