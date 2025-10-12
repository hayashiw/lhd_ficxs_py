import numpy as np
import numpy.typing as npt

from typing import Tuple, Annotated

ZEFF_0 = 3.0

D_ALPHA = 656.104 # nm
H_ALPHA = 656.279 # nm

HBAR_SI = 1.054571817e-34 # J s
SPEED_OF_LIGHT = 299792458 # m/s
MU_NAUGHT_SI = 4*np.pi*1e-7 # N A-2
EPS_NAUGHT_SI = 1/(MU_NAUGHT_SI*SPEED_OF_LIGHT**2) # C^2 J-1 m-1

ELEMENTARY_CHARGE = 1.602176634e-19 # C 
J_PER_EV = ELEMENTARY_CHARGE # J/eV

HYDR_MASS_KG  = 1.6726219260e-27

def conv_kinetic_energy_to_speed(
    energy: float | npt.NDArray,
    mass: float,
) -> float | npt.NDArray:
    r"""
    Convert kinetic energy to speed. Uses SI units.

    Parameters
    ----------
    energy : float or np.ndarray of shape (n_energy,), dtype float64
        Energy in Joules (J).
    mass : float
        Mass in kilograms (kg).

    Returns
    -------
    speed : float or np.ndarray of shape (n_energy,), dtype float64
        Speed calculated from particle kinetic energy.
    """
    if mass == 0: return 0
    if np.isscalar(energy):
        if energy/mass < 0: return 0
        speed = np.sqrt(2*energy/mass)
    else:
        speed = np.sqrt(
            2*energy/mass, out=np.zeros_like(energy), where=energy >= 0)
    return speed

def conv_speed_to_kinetic_energy(
    speed: float | npt.NDArray,
    mass: float
) -> float | npt.NDArray:
    r"""
    Convert speed to kinetic energy in SI units.

    Parameters
    ----------
    speed : float or np.ndarray of shape (n_energy,), dtype float64
        Speed calculated from particle kinetic energy.
    mass : float
        Mass in kilograms (kg).

    Returns
    -------
    energy : float or np.ndarray of shape (n_energy,), dtype float64
        Energy in Joules (J).
    """
    return 0.5 * mass * speed**2

def conv_speed_to_Doppler_shift(
    speed: float,
    unshifted_wavelength: float,
    blue_shift: bool=False
) -> float:
    r"""
    Convert speed to Doppler shifted wavelength. Uses SI units.
    
    Parameters
    ----------
    speed : float
        Speed in meters per second (m/s)
    unshifted_wavelength : float
        Unshifted wavelength in nanometers (nm).
    blue_shift : bool, optional
        Set to `True` if source is moving toward detector.
    """
    Dopp_sign = 1 if blue_shift else -1
    speed_to_c_ratio = speed / SPEED_OF_LIGHT
    shifted_wavelength = unshifted_wavelength*(1 - Dopp_sign*speed_to_c_ratio)
    return shifted_wavelength

def conv_kinetic_energy_to_Doppler_shift(
    energy: float,
    mass: float,
    unshifted_wavelength: float,
    blue_shift: bool=False
) -> float:
    r"""
    Convert kinetic energy to Doppler shifted wavelength. Uses SI units.

    Parameters
    ----------
    energy : float
        Energy in Joules (J).
    mass : float
        Mass in kilograms (kg).
    unshifted_wavelength : float
        Unshifted wavelength in nanometers (nm).
    blue_shift : bool, optional
        Set to `True` if source is moving toward detector.

    Returns
    -------
    shifted_wavelength : float
        Doppler-shifted wavelength of emission from source in 
        nanometers.
    """
    speed = conv_kinetic_energy_to_speed(energy, mass)
    return conv_kinetic_energy_to_Doppler_shift(
        speed, unshifted_wavelength, blue_shift=blue_shift)

def calc_Debye_length(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')]
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate the Debye length according to thermal electron profiles.

    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).

    Returns
    -------
    lambda_D : np.ndarray of shape (n_rho,), dtype float64
        Debye length due to electron profiles in centimeters (cm).

    Notes
    -----
    The calculation used here is for quasineutral ideal fusion plasmas 
    so only electrons are considered.
    """
    n_e_SI = n_e * 1e6 # cm-3 -> m-3
    T_e_SI = T_e * 1e3 * J_PER_EV # keV -> J
    num = EPS_NAUGHT_SI*T_e_SI
    den = n_e_SI * ELEMENTARY_CHARGE**2
    lambda_D_2 = np.divide(
        num, den, out=np.full_like(n_e, np.nan), where=den != 0.0)
    lambda_D = np.sqrt(
        lambda_D_2, out=np.full_like(n_e, np.nan), where=lambda_D_2 >= 0.0)
    return lambda_D * 100 # m -> cm

def calc_reduced_mass(m_1: float, m_2:float) -> float:
    r"""
    Calculate the reduced mass of a two-particle system.

    Parameters
    ----------
    m_1 : float
        Mass or mass number of particle 1.
    m_2 : float
        Mass or mass number of particle 2.

    Returns
    -------
    m_r : float
        Reduced mass of the two-particle system.
    """
    if m_1 + m_2 == 0: return 0

    return m_1 * m_2 / (m_1 + m_2)

def calc_deBroglie_length(A_i: int, v_f: float, A_f: int) -> float:
    r"""
    Calculate the deBroglie wavelength of a fast ion-thermal ion system.
    Assumes T_i << fast ion energy << speed of light.

    Parameters
    ----------
    A_i : int
        Atomic mass number for a thermal ion species.
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.

    Returns
    -------
    lambda_dB : float
        DeBroglie wavelength in centimeters (cm).
    """
    if (v_f == 0) or (A_f == 0) or (A_i == 0): return 0

    v_f_SI = v_f * 1e2 # cm/s -> m/s
    E_f = conv_speed_to_kinetic_energy(v_f_SI, A_f*HYDR_MASS_KG) # J
    A_avg = (A_f + A_i) / 2
    m_f = A_f * HYDR_MASS_KG # kg
    lambda_dB = HBAR_SI * A_avg / np.sqrt(2 * m_f * E_f) / A_i # m
    # m_r = calc_reduced_mass(A_f, A_i) * HYDR_MASS_KG
    # lambda_dB = HBAR_SI / np.sqrt(2 * m_r * E_f_SI)
    return lambda_dB * 100 # m -> cm

def impact_parameter_perp(
    A_i: int,
    Z_i: int,
    v_f: float,
    A_f: int,
    Z_f: int,
) -> float:
    r"""
    Calculate the perpendicular collision impact parameter of a fast ion
    and a thermal ion. Assumes T_i << fast ion energy << speed of light.

    Parameters
    ----------
    A_i : int
        Atomic mass number for a thermal ion species.
    Z_i : int
        Nuclear charge number for a thermal ion species.
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.

    Returns
    -------
    b_90 : float
        Perpendicular impact parameter in centimeters (cm).
    """
    if (v_f == 0) or (A_f + A_i == 0): return 0

    v_f_SI = v_f * 1e2 # cm/s -> m/s
    m_r = calc_reduced_mass(A_f, A_i) * HYDR_MASS_KG # kg
    q_f = Z_f * ELEMENTARY_CHARGE # C
    q_i = Z_i * ELEMENTARY_CHARGE # C
    b_90 = q_f * q_i / \
        (4 * np.pi * EPS_NAUGHT_SI * m_r * v_f_SI**2) # m
    return b_90 * 100 # m -> cm

def impact_parameter(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    A_i: int,
    Z_i: int,
    v_f: float,
    A_f: int,
    Z_f: int
) -> Tuple[
    float,
    Annotated[npt.NDArray[np.float64], ('n_rho')]
]:
    r"""
    Calculate the impact parameter profile for fast ion-thermal ion
    collisions.

    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    A_i : int
        Atomic mass number for a thermal ion species.
    Z_i : int
        Nuclear charge number for a thermal ion species.
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.

    Returns
    -------
    b_min : float
        Minimum impact parameter in centimeters (cm).
    b_max : np.ndarray of shape (n_rho,), dtype float64
        Maximum impact parameter due to electron profiles in centimeters
        (cm).
    """
    b_max = calc_Debye_length(n_e, T_e) # cm
    lambda_dB = calc_deBroglie_length(A_i, v_f, A_f) # cm
    b_90 = impact_parameter_perp(A_i, Z_i, v_f, A_f, Z_f) # cm
    b_min = max(lambda_dB, b_90) # cm
    return b_min, b_max # cm, cm

def _calc_Coulomb_ii(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    A_i: int,
    Z_i: int,
    v_f: float,
    A_f: int,
    Z_f: int
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate the fast ion-thermal ion Coulomb logarithm for one thermal
    ion species.

    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    A_i : int
        Atomic mass number for a thermal ion species.
    Z_i : int
        Nuclear charge number for a thermal ion species.
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.

    Returns
    -------
    log_lambda_ii : np.ndarray of shape (n_rho,), dtype float64
        Ion-ion Coulomb logarithm for fast ion collisions with a thermal
        ion species.
    """
    bmin, bmax = impact_parameter(n_e, T_e, A_i, Z_i, v_f, A_f, Z_f)
    if bmin == 0: return 0
    return np.log(
        bmax / bmin, out=np.zeros_like(bmax), where=(bmax / bmin) > 0)

def calc_Coulomb_ii(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    A_i: Annotated[npt.NDArray[np.int32], ('n_rho')],
    Z_i: Annotated[npt.NDArray[np.int32], ('n_rho')],
    v_f: float,
    A_f: int,
    Z_f: int
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate the fast ion-thermal ion Coulomb logarithm for all thermal
    ion species.

    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    A_i : np.ndarray of shape (n_species,), dtype int32
        Atomic mass numbers for all thermal ion species.
    Z_i : np.ndarray of shape (n_species,), dtype int32
        Nuclear charge number for all thermal ion species.
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.

    Returns
    -------
    log_lambda_ii : np.ndarray of shape (n_rho,), dtype float64
        Ion-ion Coulomb logarithm for fast ion collisions with all
        thermal ion species.
    """
    return np.array([
        _calc_Coulomb_ii(n_e, T_e, A, Z, v_f, A_f, Z_f)
        for A, Z in zip(A_i, Z_i)])

def calc_Coulomb_ie(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate the ion-electron Coulomb logarithm.

    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).

    Returns
    -------
    log_lambda_ie : np.ndarray of shape (n_rho,), dtype float64
        Ion-electron Coulomb logarithm.
    """
    n_e = np.array(n_e)
    n_rho = n_e.size
    sqrt_n_e = np.sqrt(n_e, out=np.full(n_rho, np.nan), where=n_e >= 0)
    
    T_e_eV = np.array(T_e) * 1e3 # keV -> eV
    above10 = T_e_eV > 10
    below10 = T_e_eV < 10
    cbrt_T_e = np.cbrt(
        T_e_eV, out=np.full(n_rho, np.nan), where=T_e_eV >= 0 )

    log_lambda_ie = np.zeros(n_rho)
    if above10.sum():
        above10_frac = np.divide(
            sqrt_n_e[above10],
            T_e_eV[above10],
            out=np.full(above10.sum(), np.nan),
            where=T_e_eV[above10] != 0 )
        log_lambda_ie[above10] = 24 - np.log(
            above10_frac,
            out=np.full(above10.sum(), np.nan),
            where=above10_frac > 0)
    if below10.sum():
        below10_frac = np.divide(
            sqrt_n_e[below10],
            cbrt_T_e[below10],
            out=np.full(below10.sum(), np.nan),
            where=cbrt_T_e[below10] != 0 )
        log_lambda_ie[below10] = 23 - np.log(
            below10_frac,
            out=np.full(below10.sum(), np.nan),
            where=below10_frac > 0)
    return log_lambda_ie

def calc_critical_energy(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    A_i: Annotated[npt.NDArray[np.int32], ('n_rho')],
    Z_i: Annotated[npt.NDArray[np.int32], ('n_rho')],
    v_f: float,
    A_f: int,
    Z_f: int
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate the critical energy of a fast ion due to thermal profiles.

    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    A_i : np.ndarray of shape (n_species,), dtype int32
        Atomic mass numbers for all thermal ion species.
    Z_i : np.ndarray of shape (n_species,), dtype int32
        Nuclear charge number for all thermal ion species.
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.

    Returns
    -------
    E_c : np.ndarray of shape (n_rho,), dtype float64
        Fast ion critical energy due to thermal profiles in
        kiloelectronvolts (keV).

    Notes
    -----
    Below the critical energy, fast ions mainly slow down on thermal
    ions. Above the critical energy, they mainly slow down on thermal
    electrons [1]_.

    References
    ----------
    .. [1] W. W. Heidbrink, G. J. Sadler, "The Behavior of Fast Ions in
        Tokamak Experiments," Nuclear Fusion, vol. 34 pp. 535-615, 1994.
    """
    log_lambda_ie = calc_Coulomb_ie(n_e, T_e)
    log_lambda_ii = calc_Coulomb_ii(n_e, T_e, A_i, Z_i, v_f, A_f, Z_f)
    arg = np.divide(
        log_lambda_ii.sum(0), n_e * log_lambda_ie, out=np.zeros_like(n_e),
        where=(n_e*log_lambda_ie) != 0 )
    return 14.8 * A_f * T_e * arg
