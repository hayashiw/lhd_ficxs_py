import numpy as np
import numpy.typing as npt
import os

FDIR = os.path.dirname(__file__)
lhdficxs_spec_file = FDIR + '/geometry_files/spec_geo_lhdficxs.txt'
lhdficxs_map_file = FDIR + '/geometry_files/lhdficxs_to_fmap.txt'
nbi_geo_file = FDIR + '/geometry_files/neutral_beam_geo.txt'

def read_lhdficxs() -> npt.NDArray:
    r"""
    Read in the diagnostic sightline geometry for the LHD FICXS system.

    Returns
    -------
    spec : np.ndarray
        Structured array with fields:
        - ch : int
        - lens_u, lens_v, lens_w : float
        - axis_u, axis_v, axis_w : float
        - radius, sigma_pi, spot_size : float
    """
    dtype = [
        ('ch', int),
        ('lens_u', float), ('lens_v', float), ('lens_w', float),
        ('axis_u', float), ('axis_v', float), ('axis_w', float),
        ('radius', float), ('sigma_pi', float), ('spot_size', float) ]
    data = np.loadtxt(lhdficxs_spec_file, skiprows=1, dtype=dtype)
    return data

def lhdficxs_to_fmap(fmap: str, lhdficxs_ch: int) -> int:
    r"""
    Retrieve the corrseponding individual fibermap channel for a
    LHD-FICXS channel.

    Parameters
    ----------
    fmap : {"6OFIDA", "PNBFIDA", NNBFIDA"}
        Fibermap label.
    lhdficxs_ch : int
        LHD-FICXS channel number.

    Returns
    -------
    fmap_ch : int
        Fibermap channel number.
    """
    dtype = [
        ('fmap_ch', int), ('fmap', 'U7'), ('lhdficxs_ch', int) ]
    data = np.loadtxt(lhdficxs_map_file, skiprows=1, dtype=dtype)
    cond = (data['fmap'] == fmap) & (data['lhdficxs_ch'] == lhdficxs_ch)
    if not any(cond):
        raise Exception(f'{fmap = }, {lhdficxs_ch = }')
    fmap_ch = data[cond]['fmap_ch'][0]
    return fmap_ch

def fmap_to_lhdficxs(fmap: str, fmap_ch: int) -> int:
    r"""
    Retrieve the corrseponding LHD-FICXS channel for a fibermap channel.

    Parameters
    ----------
    fmap : {"6OFIDA", "PNBFIDA", NNBFIDA"}
        Fibermap label.
    fmap_ch : int
        Fibermap channel number.

    Returns
    -------
    lhdficxs_ch : int
        LHD-FICXS channel number.
    """
    dtype = [
        ('fmap_ch', int), ('fmap', 'U7'), ('lhdficxs_ch', int) ]
    data = np.loadtxt(lhdficxs_map_file, skiprows=1, dtype=dtype)
    cond = (data['fmap'] == fmap) & (data['fmap_ch'] == fmap_ch)
    if not any(cond):
        raise Exception(f'{fmap = }, {fmap_ch = }')
    lhdficxs_ch = data[cond]['lhdficxs_ch'][0]
    return lhdficxs_ch

def read_diag_geo() -> npt.NDArray:
    r"""
    Read in the diagnostic newutral beam geometry for the LHD FICXS
    system.

    Returns
    -------
    nbi : np.ndarray
        Structured array with fields:
        - name : unicode string of length 5
        - shape : int
        - src_x, src_y, src_z : float
        - axis_x, axis_y, axis_z : float
        - widy, widz, divy, divz, focy, focz : float
        - naperture, ashape : int
        - awidy_1, awidz_1, aoffy_1, aoffz_1, adist_1 : float
    """
    dtype = [
        ('name', 'U5'), ('shape', int),
        ('src_x',  float), ('src_y',  float), ('src_z',  float),
        ('axis_x', float), ('axis_y', float), ('axis_z', float),
        ('widy', float), ('widz', float),
        ('divy', float), ('divz', float),
        ('focy', float), ('focz', float),
        ('naperture', int), ('ashape', int),
        ('awidy_1', float), ('awidz_1', float),
        ('aoffy_1', float), ('aoffz_1', float),
        ('adist_1', float) ]
    data = np.loadtxt(nbi_geo_file, skiprows=1, dtype=dtype)
    return data

def read_bgrids(geo_dir: str=None) -> np.ndarray:
    r"""
    Read in beam grid parameters for all beam sources.

    Parameters
    ----------
    geo_dir : str, optional
        Directory where "bgrids.txt" is saved. If `None`, reads file from
        `geometry_files`.

    Returns
    -------
    bgrids : np.ndarray
        Structured array with fields:
        - name : unicode string of length 5
        - alpha, beta, gamma : float
        - orig_x, orig_y, orig_z : float
        - xmin, xmax, ymin, ymax, zmin, zmax : float
    """

    if geo_dir is None: geo_dir = FDIR + '/geometry_files'
    bgrid_file = geo_dir + '/bgrids.txt'
    dtype = [
        ('name', 'U5'),
        ('alpha', float), ('beta', float), ('gamma', float),
        ('orig_x', float), ('orig_y', float), ('orig_z', float),
        ('xmin', float), ('xmax', float), ('ymin', float), ('ymax', float),
        ('zmin', float), ('zmax', float) ]
    data = np.loadtxt(bgrid_file, skiprows=1, dtype=dtype)
    return data


def make_beam_grids(geo_dir: str=None) -> None:
    r"""
    Generate beam grid parameters for all possible FIDASIM geometry
    files for LHD-FICXS system.

    Parameters
    ----------
    geo_dir : str, optional
        Directory to store "bgrids.txt" file. If `None`, stores file in
        `geometry_files`.
    
    Notes
    -----
    A total of 14 NBI sources are available on LHD. The FICXS system has
    three possible channel configurations ("6OFIDA", "PNBFIDA",
    "NNBFIDA"). However, they often map to the same channel. The system
    is simplified to a singled 24 channel configuration rather than the
    three 16 channel configurations. Therefore, there are only 14
    possible NBI-FICXS combinations.
    """
    from fidasim.preprocessing import beam_grid
    if geo_dir is None: geo_dir = FDIR + '/geometry_files'
    bgrid_file = geo_dir + '/bgrids.txt'
    with open(bgrid_file, 'w') as file:
        file.write(
            'name alpha beta gamma orig_x orig_y orig_z '
            'xmin xmax ymin ymax zmin zmax\n')
        all_nbi = read_diag_geo()
        for (
            name, shape,
            src_x, src_y, src_z, vec_x, vec_y, vec_z,
            widy, widz, divy, divz, focy, focz,
            naperture, ashape, awidy_1, awidz_1, aoffy_1, aoffz_1, adist_1
        ) in all_nbi:
            src = np.array([src_x, src_y, src_z])
            vec = np.array([vec_x, vec_y, vec_z])
            divy3 = np.full(3, divy)
            divz3 = np.full(3, divz)

            nbi = dict(
                data_source=nbi_geo_file, name=name, shape=shape,
                src=src, axis=vec, divy=divy3, divz=divz3,
                widy=widy, widz=widz, focy=focy, focz=focz,
                naperture=naperture, ashape=ashape, adist_1=adist_1,
                awidy_1=awidy_1, awidz_1=awidz_1,
                aoffy_1=aoffy_1, aoffz_1=aoffz_1 )
            
            rstart =  600
            length = 1000 if int(name[2]) <= 3 else 300
            width, height = 200.0, 200.0
            dv = 3.0**3
            kwargs = dict(dv=dv, length=length, width=width, height=height)
            bgrid = beam_grid(nbi, rstart, **kwargs)
            alpha, beta, gamma = bgrid['alpha'], bgrid['beta'], bgrid['gamma']
            orig_x, orig_y, orig_z = bgrid['origin']
            xmin, ymin, zmin = bgrid['xmin'], bgrid['ymin'], bgrid['zmin']
            xmax, ymax, zmax = bgrid['xmax'], bgrid['ymax'], bgrid['zmax']
            file.write(
                f'{name:<5} '
                f'{alpha:9.6f} {beta:9.6f} {gamma:9.6f} '
                f'{orig_x:11.6f} {orig_y:11.6f} {orig_z:11.6f}'
                f'{xmin:11.6f} {xmax:11.6f} '
                f'{ymin:11.6f} {ymax:11.6f} '
                f'{zmin:11.6f} {zmax:11.6f}\n')


def make_geometries(geo_dir: str=None) -> None:
    r"""
    Generate all possible FIDASIM geometry files for LHD-FICXS system.

    Parameters
    ----------
    geo_dir : str, optional
        Directory to store geometry .h5 files. If `None`, stores files
        in `geometry_files`.
    
    Notes
    -----
    A total of 14 NBI sources are available on LHD. The FICXS system has
    three possible channel configurations ("6OFIDA", "PNBFIDA",
    "NNBFIDA"). However, they often map to the same channel. The system
    is simplified to a singled 24 channel configuration rather than the
    three 16 channel configurations. Therefore, there are only 14
    possible NBI-FICXS combinations.
    """
    from fidasim.preprocessing import check_spec, check_beam, write_geometry
    if geo_dir is None: geo_dir = FDIR + '/geometry_files'

    spec = read_lhdficxs()
    ch = spec['ch'].astype(bytes)
    nchan = ch.size
    lens = np.array([
        list(val) for val in spec[['lens_u', 'lens_v', 'lens_w']]]).T
    axis = np.array([
        list(val) for val in spec[['axis_u', 'axis_v', 'axis_w']]]).T
    radius = spec['radius']
    sigma_pi = spec['sigma_pi']
    spot_size = spec['spot_size']
    spec_fidasim = dict(
        data_source=lhdficxs_spec_file, system='LHD-FICXS',
        lens=lens, axis=axis, nchan=nchan, id=ch,
        radius=radius, sigma_pi=sigma_pi, spot_size=spot_size )
    spec_inputs = dict(
        alpha=0.0, beta=0.0, gamma=0.0, origin=np.zeros(3),
        xmin=-360.0, ymin=-540.0, zmin=-200.0,
        xmax= 360.0, ymax=   0.0, zmax= 200.0 )
    check_spec(spec_inputs, spec_fidasim)

    bgrids = read_bgrids(geo_dir=geo_dir)
    all_nbi = read_diag_geo()
    for (
        name, shape,
        src_x, src_y, src_z, vec_x, vec_y, vec_z,
        widy, widz, divy, divz, focy, focz,
        naperture, ashape, awidy_1, awidz_1, aoffy_1, aoffz_1, adist_1
    ) in all_nbi:
        src = np.array([src_x, src_y, src_z])
        vec = np.array([vec_x, vec_y, vec_z])
        divy3 = np.full(3, divy)
        divz3 = np.full(3, divz)

        awidy = np.array([awidy_1])
        awidz = np.array([awidz_1])
        aoffy = np.array([aoffy_1])
        aoffz = np.array([aoffz_1])
        adist = np.array([adist_1])

        nbi_fidasim = dict(
            data_source=nbi_geo_file, name=name, shape=int(shape),
            src=src, axis=vec, divy=divy3, divz=divz3, widy=widy, widz=widz,
            focy=focy, focz=focz, naperture=int(naperture), ashape=ashape,
            awidy=awidy, awidz=awidz, aoffy=aoffy, aoffz=aoffz, adist=adist )
        
        bgrid = bgrids[bgrids['name'] == name]
        nbi_inputs = {name:bgrid[name].item() for name in bgrid.dtype.names}
        nbi_inputs['origin'] = [nbi_inputs[f'orig_{i}'] for i in ['x','y','z']]
        nbi_fidasim = check_beam(nbi_inputs, nbi_fidasim)

        geo_file = geo_dir + f'/{name}LHDFICXS_geometry.h5'
        write_geometry(geo_file, nbi_fidasim, spec=spec_fidasim)