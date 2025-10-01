import numpy as np

ZEFF_0 = 3.0

D_ALPHA = 656.104 # nm
H_ALPHA = 656.279 # nm

HBAR_SI = 1.054571817e-34 # J s
SPEED_OF_LIGHT = 299792458 # m/s
MU_NAUGHT_SI = 4*np.pi*1e-7 # N A-2
EPS_NAUGHT_SI = 1/(MU_NAUGHT_SI*SPEED_OF_LIGHT**2) # C^2 J-1 m-1

ELEMENTARY_CHARGE = 1.602176634e-19 # C 
J_PER_EV = ELEMENTARY_CHARGE # J/eV

DEUT_MASS_KG  = 3.3435837768e-27
DEUT_MASS_G   = DEUT_MASS_KG * 1e3
DEUT_MASS_MEV = DEUT_MASS_KG * SPEED_OF_LIGHT**2 / J_PER_EV / 1e6 # MeV [* c^2]
DEUT_MASS_KEV = DEUT_MASS_MEV * 1000
DEUT_MASS_EV  = DEUT_MASS_KEV * 1000

HYDR_MASS_KG  = 1.6726219260e-27
HYDR_MASS_G   = HYDR_MASS_KG * 1e3
HYDR_MASS_MEV = HYDR_MASS_KG * SPEED_OF_LIGHT**2 / J_PER_EV / 1e6
HYDR_MASS_KEV = HYDR_MASS_MEV * 1000
HYDR_MASS_EV  = HYDR_MASS_KEV * 1000

def convert_kinetic_energy_to_speed(
    energy: float,
    mass: float,
) -> float:
    """
    Convert kinetic energy to speed. Assumes energy and mass are in compatible
    units.

    Parameters
    ----------
    energy : float
        Energy in select units. Double check units are compatible with mass.
    mass : float
        Mass in select_units. Double check units are compatible with energy.

    Returns
    -------
    speed : float
        Speed calculated from particle kinetic energy.

    Notes
    -----
    .. attention::
        For SI units, energy should be in Joules and mass should be in
        kilograms. The resultant speed is then in meters per second.
        If energy is input in kiloelectronvolts and mass is input in
        kiloelectronvolts per speed-of-light-squared then the output
        speed is a percentage of the speed-of-light.
    """
    speed = np.sqrt(2*energy/mass)
    return speed

def convert_kinetic_energy_to_Doppler_shift(
    energy: float,
    mass: float,
    unshifted_wavelength: float,
    blue_shift: bool=False
) -> float:
    """
    Convert kinetic energy to Doppler shifted wavelength. Assumes
    energy and mass are in SI units.

    Parameters
    ----------
    energy : float
        Energy in Joules.
    .. attention::
        Pay attention to units. Energy should be in Joules.
    mass : float
        Mass in kilograms.
    .. attention::
        Pay attention to units. Mass should be in kilograms.
    unshifted_wavelength : float
        Unshifted wavelength in nanometers.
    blue_shift : bool, optional
        Set to `True` if source is moving toward detector.

    Returns
    -------
    shifted_wavelength : float
        Doppler-shifted wavelength of emission from source in 
        nanometers.
    """
    Dopp_sign = 1 if blue_shift else -1
    speed = Dopp_sign * convert_kinetic_energy_to_speed(energy, mass)
    speed_to_c_ratio = speed / SPEED_OF_LIGHT
    shifted_wavelength = unshifted_wavelength * (1 - speed_to_c_ratio)

    return shifted_wavelength

def calc_Debye_length(n_e: np.ndarray, T_e: np.ndarray) -> np.ndarray:
    """
    Calculate the Debye length according to thermal electron profiles.

    Parameters
    ----------
    n_e : np.ndarray
        1-D (n_rho,) thermal electron density profile in inverse cubic
        centimeters (cm-3).
    T_e : np.ndarray
        1-D (n_rho,) thermal electron temperature profile in
        kiloelectronvolts (keV).

    Returns
    -------
    lambda_D : np.ndarray
        1-D (n_rho,) profile of the Debye length in centimeters (cm).

    Notes
    -----
    The calculation used here is for quasineutral ideal fusion plasmas 
    so only electrons are considered.
    """
    n_e_SI = np.ndarray(n_e) * 1e6 # cm-3 -> m-3
    n_rho = n_e_SI.size
    T_e_SI = np.ndarray(T_e) * 1e3 * J_PER_EV # keV -> J
    numerator = (EPS_NAUGHT_SI*T_e_SI)
    denominator = n_e_SI * ELEMENTARY_CHARGE**2
    lambda_D_2 = np.divide(
        numerator, denominator, out=np.full(n_rho, np.nan),
        where=denominator!=0.0)
    lambda_D = np.sqrt(
        lambda_D_2, out=np.full(n_rho, np.nan),where=lambda_D_2 >= 0.0)
    return lambda_D * 100 # m -> cm

def calc_reduced_mass(m_1: float, m_2:float) -> float:
    """
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
    return m_1 * m_2 / (m_1 + m_2)

def calc_deBroglie_length(E_f: float, A_f: int, A_i: int) -> float:
    """
    Calculate the deBroglie wavelength of a fast ion-thermal ion system.
    Assumes T_i << fast ion energy << speed of light.

    Parameters
    ----------
    E_f : float
        Fast ion energy in kiloelectronvolts (keV).
    A_f : int
        Fast ion atomic mass number.
    A_i : int
        Main ion species atomic mass number.

    Returns
    -------
    lambda_dB : float
        DeBroglie wavelength in centimeters (cm).
    """
    E_f_SI = E_f * 1e3 * J_PER_EV # keV -> J
    A_avg = (A_f + A_i) / 2
    m_f = A_f * HYDR_MASS_KG
    lambda_dB = HBAR_SI * A_avg / np.sqrt(2 * m_f * E_f_SI) / A_i
    # m_r = calc_reduced_mass(A_f, A_i) * HYDR_MASS_KG
    # lambda_dB = HBAR_SI / np.sqrt(2 * m_r * E_f_SI)
    return lambda_dB * 100 # m -> cm

def impact_parameter_perp(
    E_f: float,
    A_f: int,
    Z_f: int,
    A_i: int,
    Z_i: int) -> float:
    """
    Calculate the perpendicular collision impact parameter of a fast ion
    and a thermal ion. Assumes T_i << fast ion energy << speed of light.

    Parameters
    ----------
    E_f : float
        Fast ion energy in kiloelectronvolts (keV).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.
    A_i : int
        Main ion species atomic mass number.
    Z_i : int
        Main ion species nuclear charge number.

    Returns
    -------
    b_90 : float
        Perpendicular impact parameter in centimeters (cm).
    """
    E_f_SI = E_f * 1e3 * J_PER_EV # keV -> J
    v_f = convert_kinetic_energy_to_speed(E_f_SI, A_f*HYDR_MASS_KG)
    m_r = calc_reduced_mass(A_f, A_i) * HYDR_MASS_KG
    q_f = Z_f * ELEMENTARY_CHARGE
    q_i = A_i * ELEMENTARY_CHARGE
    b_90 = q_f * q_i / (4 * np.pi * EPS_NAUGHT_SI * m_r * v_f**2)
    return b_90 * 100 # m -> cm

def impact_parameter(
    n_e: np.ndarray,
    T_e: np.ndarray,
    E_f: float,
    A_f: int,
    Z_f: int,
    A_i: int,
    Z_i: int
) -> tuple[float, np.ndarray]:
    """
    Calculate the impact parameter profile for fast ion-thermal ion
    collisions.

    Parameters
    ----------
    n_e : np.ndarray
        1-D (n_rho,) thermal electron density profile in inverse cubic
        centimeters (cm-3).
    T_e : np.ndarray
        1-D (n_rho,) thermal electron temperature profile in
        kiloelectronvolts (keV).
    E_f : float
        Fast ion energy in kiloelectronvolts (keV).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.
    A_i : int
        Main ion species atomic mass number.
    Z_i : int
        Main ion species nuclear charge number.

    Returns
    -------
    b_min : float
        Minimum impact parameter in centimeters (cm).
    b_max : np.ndarray
        1-D (n_rho,) profile of the maximum impact parameter due to
        electron profiles in centimeters (cm).
    """
    b_max = calc_Debye_length(n_e, T_e)

    lambda_dB = calc_deBroglie_length(E_f, A_f, A_i)
    b_90 = impact_parameter_perp(E_f, A_f, Z_f, A_i, Z_i)
    b_min = max(lambda_dB, b_90)

    return b_min, b_max

def calc_Coulomb_ii(
    n_e: np.ndarray,
    T_e: np.ndarray,
    E_f: float,
    A_f: int,
    Z_f: int,
    A_i: int,
    Z_i: int
) -> np.ndarray:
    """
    Calculate the fast ion-thermal ion Coulomb logarithm.

    Parameters
    ----------
    n_e : np.ndarray
        1-D (n_rho,) thermal electron density profile in inverse cubic
        centimeters (cm-3).
    T_e : np.ndarray
        1-D (n_rho,) thermal electron temperature profile in
        kiloelectronvolts (keV).
    E_f : float
        Fast ion energy in kiloelectronvolts (keV).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.
    A_i : int
        Main ion species atomic mass number.
    Z_i : int
        Main ion species nuclear charge number.

    Returns
    -------
    log_lambda_ii : np.ndarray
        1-D (n_rho,) profile of the fast ion-thermal ion Coulomb
        logarithm.
    """
    bmin, bmax = impact_parameter()

def calc_Coulomb_ie(n_e: np.ndarray, T_e: np.ndarray) -> np.ndarray:
    """
    Calculate the ion-electron Coulomb logarithm.

    Parameters
    ----------
    n_e : np.ndarray
        1-D (n_rho,) thermal electron density profile in inverse cubic
        centimeters (cm-3).
    T_e : np.ndarray
        1-D (n_rho,) thermal electron temperature profile in
        kiloelectronvolts (keV).

    Returns
    -------
    log_lambda_ie : np.ndarray
        1-D (n_rho,) profile of the ion-electron Coulomb logarithm.
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
    n_e: np.ndarray,
    T_e: np.ndarray,
    E_f: float,
    A_f: int,
    Z_f: int,
    n_i: np.ndarray=None,
    A_i: np.ndarray=None,
    Z_i: np.ndarray=None,
    Z_eff: np.ndarray=None
) -> np.ndarray:
    """
    Calculate the critical energy of a fast ion due to thermal profiles.

    Parameters
    ----------
    n_e : np.ndarray
        1-D (n_rho,) thermal electron density profile in inverse cubic
        centimeters (cm-3).
    T_e : np.ndarray
        1-D (n_rho,) thermal electron temperature profile in
        kiloelectronvolts (keV).
    E_f : float
        Fast ion birth energy in kiloelectronvolts (keV).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.
    n_i : np.ndarray
        2-D (n_species, n_rho) thermal ion density profiles including
        minority and impurity species in inverse cubic centimeters
        (cm-3).
    A_i : array_like
        1-D (n_species,) sequence of thermal ion mass numbers including
        minority and impurity species.
    Z_i : array_like
        1-D (n_species,) sequence of thermal nuclear charge numbers
        including minority and impurity species.

    Returns
    -------
    E_c : np.ndarray
        1-D (n_rho,) profile of the fast ion critical energy in
        kiloelectronvolts (keV).

    Notes
    -----
    Below the critical energy, fast ions mainly slow down on thermal
    ions. Above the critical energy, they mainly slow down on thermal
    electrons [1]_.

    If thermal ion profiles are not input, the main ion and impurity
    species are assumed to be deuterium and carbon with a uniform
    Z_eff = 3 profile.

    References
    ----------
    .. [1] W. W. Heidbrink, G. J. Sadler, "The Behavior of Fast Ions in
        Tokamak Experiments," Nuclear Fusion, vol. 34 pp. 535-615, 1994.
    """
    n_e = np.array(n_e)
    T_e = np.array(T_e)
    n_rho = n_e.size

    if n_i is None:
        if Z_eff is None: Z_eff = np.full(n_rho, ZEFF_0)
        A_i = np.array([12, 1])
        Z_i = np.array([ 6, 1])
        A_mx = np.array([ [Z**2 for Z in Z_i], [Z for Z in Z_i] ])
        b_vc = np.array([Z_eff * n_e, n_e])
        n_i = np.linalg.solve(A_mx, b_vc)
    else:
        A_i = np.array(A_i)
        Z_i = np.array(Z_i)
        n_species = n_i.shape[0]
        assert n_species == A_i.size and n_species == Z_i.size, (
            f'Input ion parameters must match number of species '
            f'{n_species = }, {A_i.size = }, {Z_i.size}.')
