r"""
Solutions to the Fokker-Planck equation described by
Goldston [1]_.

References
----------
.. [1] Goldston, R.L., Charge-exchange spectra near the injection energy
    in tokamaks equipped with tangential neutral beams - experiment and
    theory, Nuclear Fusion **15** (1975) 651
"""

import numpy as np
import numpy.typing as npt
import os
import sys

from typing import Tuple, Annotated
from scipy.special import legendre
from scipy.integrate import simpson

PKG_PAR_DIR = os.path.abspath('../..')
sys.path.insert(0, PKG_PAR_DIR)
from lhd_ficxs_py.physics import (
    conv_kinetic_energy_to_speed, calc_Coulomb_ie, calc_Coulomb_ii,
    J_PER_EV, HYDR_MASS_KG )

def calc_gaussian(
    x: Annotated[npt.NDArray[np.float64], ('n_x')],
    x0: float,
    e0: float
) -> Annotated[npt.NDArray[np.float64], ('n_x')]:
    r"""
    Calculate a Gaussian function. The initial fast ion birth
    distribution in pitch space is assumed to be a Gaussian.

    Parameters
    ----------
    x : np.ndarray of shape (n_x,), dtype float64
        Dependent variable grid.
    x0 : float
        Center value of Gaussian distribution.
    e0 : float
        Spread of Gaussian distribution.

    Returns
    -------
    gauss : np.ndarray of shape (n_x,), dtype float64
        1-D (n_x,) Gaussian distribution.
    """
    if e0 == 0:
        ix0 = np.abs(x - x0).argmin()
        gauss = np.zeros_like(x)
        gauss[ix0] = x0
        return gauss
    arg = (x - x0)/e0
    coeff = 1/(e0*np.sqrt(2*np.pi))
    gauss = coeff * np.exp(-0.5 * arg**2)

    # Renormalize Gaussian so that integral(gauss) within the boundaries
    # of x still goes to unity.
    return gauss / simpson(gauss, x=x)

def calc_s_ell_from_pitch(
    ell: int,
    pitch: Annotated[npt.NDArray[np.float64], ('n_pitch')],
    source_v_p: Annotated[npt.NDArray[np.float64], ('n_pitch')]
) -> float:
    r"""
    Calculate Legendre coefficients for an expansion of a source
    function in pitch space.

    Parameters
    ----------
    ell : int
        Legendre index.
    pitch : np.ndarray of shape (n_pitch,), dtype float64
        Pitch grid.
    source_v_p : np.ndarray of shape (n_pitch,), dtype float64
        Source grid vs pitch.

    Returns
    -------
    s_ell : float
        Legendre coefficient.

    Notes
    -----
    Suppose a function $f(x)$ is defined over the domain [-1, 1] by a
    series of Legendre polynomials :math`:P_\ell`:
    .. math::
        f(x) = \sum_\ell f_\ell P_\ell
    The solution to :math:`f_\ell` is then:
    .. math::
        f_\ell = (\ell + \frac{1}{2})\int_{-1}^1 f(x) P_\ell(x) dx
    """
    pitch = np.where(np.abs(pitch) > 1, 0, pitch)
    P_ell = legendre(ell)(pitch)
    return (ell+0.5) * simpson(source_v_p*P_ell, x=pitch)


def expand_s_grid_from_gaussian_pitch(
    n_pitch: int,
    s_0: float,
    p_0: float,
    delta_p_0: float,
    n_ell: int=61
) -> Tuple[
    Annotated[npt.NDArray[np.float64], ('n_pitch')],
    Annotated[npt.NDArray[np.float64], ('n_ell', 'n_pitch')]
]:
    r"""
    Legendre expand a source function in pitch space. The pitch
    distribution is modelled by a Gaussian function.

    Parameters
    ----------
    n_pitch : int
        Size of pitch grid.
    s_0 : float
        Fast ion birth rate in units of fast ions per second.
    p_0 : float
        Initial pitch value of fast ion births.
    delta_p_0 : float
        Width of initial pitch distribution.
    n_ell : int, optional
        Number of Legendre coefficients. Default is 61.

    Returns
    -------
    pitch : np.ndarray of shape (n_pitch,), dtype float64
        Pitch grid.
    s_grid : np.ndarray of shape (n_ell, n_pitch), dtype float64
        Legendre-expanded source grid.
        
    Notes
    -----
    Suppose a function $f(x)$ is defined over the domain [-1, 1] by a
    series of Legendre polynomials :math`:P_\ell`:
    .. math::
        f(x) = \sum_\ell f_\ell P_\ell
    The solution to :math:`f_\ell` is then:
    .. math::
        f_\ell = (\ell + \frac{1}{2})\int_{-1}^1 f(x) P_\ell(x) dx
    """
    assert np.abs(p_0) <= 1, f'p_0 ({p_0}) must be within domain [-1, 1]'
    
    pitch = np.linspace(-1, 1, n_pitch)
    source_v_p = calc_gaussian(pitch, p_0, delta_p_0)
    s_grid = np.zeros((n_ell, n_pitch))
    for ell in range(n_ell):
        leg_fn = legendre(ell)(pitch)
        s_ell = calc_s_ell_from_pitch(ell, pitch, source_v_p)
        s_grid[ell] = s_0 * s_ell * leg_fn
    return pitch, s_grid

def expand_s_grid_from_pitch(
    pitch: Annotated[npt.NDArray[np.float64], ('n_pitch')],
    source_v_p: Annotated[npt.NDArray[np.float64], ('n_pitch')],
    n_ell: int=61
) -> Tuple[
    Annotated[npt.NDArray[np.float64], ('n_pitch')],
    Annotated[npt.NDArray[np.float64], ('n_ell', 'n_pitch')]
]:
    r"""
    Legendre expand a source function in pitch space. The pitch
    distribution is modelled by a Gaussian function.

    Parameters
    ----------
    pitch : np.ndarray of shape (n_pitch,), dtype float64
        Pitch grid.
    source_v_p : np.ndarray of shape (n_pitch,), dtype float64
        Source grid vs pitch.
    n_ell : int, optional
        Number of Legendre coefficients. Default is 61.

    Returns
    -------
    s_grid : np.ndarray of shape (n_ell, n_pitch), dtype float64
        Legendre-expanded source grid.
        
    Notes
    -----
    Suppose a function $f(x)$ is defined over the domain [-1, 1] by a
    series of Legendre polynomials :math`:P_\ell`:
    .. math::
        f(x) = \sum_\ell f_\ell P_\ell
    The solution to :math:`f_\ell` is then:
    .. math::
        f_\ell = (\ell + \frac{1}{2})\int_{-1}^1 f(x) P_\ell(x) dx
    """
    assert (
        (pitch.ndim == 1) and
        (source_v_p.ndim == 1) and
        (pitch.size == source_v_p.size)
    ), f'pitch and source_v_p must be 1-D arrays with the same size'

    p_0 = pitch[source_v_p.argmax()]

    s_0 = simpson(source_v_p, x=pitch)
    n_pitch = pitch.size
    s_grid = np.zeros((n_ell, n_pitch))
    for ell in range(n_ell):
        leg_fn = legendre(ell)(pitch)
        s_ell = calc_s_ell_from_pitch(ell, pitch, source_v_p)
        s_grid[ell] = s_0 * s_ell * leg_fn
    return s_grid


def calc_Gamma(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    A_f: int,
    Z_f: int
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calclate :math:`\Gamma`.
    
    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.

    Returns
    -------
    Gamma : np.ndarray of shape (n_rho,), dtype float64
        :math:`\Gamma` factor profile.
    """
    return 2.39e11 * (Z_f/A_f)**2 * calc_Coulomb_ie(n_e, T_e)

def calc_A_bar(
    n_i: Annotated[np.ndarray[np.float64], ('n_species', 'n_rho')],
    A_i: Annotated[np.ndarray[np.int32], ('n_species')]
) -> Annotated[np.ndarray[np.float64], ('n_rho')]:
    r"""
    Calculate :math:`\overline{A}_i`.

    Parameters
    ----------
    n_i : np.ndarray of shape (n_species, n_rho), dtype float64
        Thermal ion densities in inverse cubic centimeters (cm-3). 
    A_i : np.ndarray of shape (n_species,), dtype int32
        Thermal ion mass numbers.

    Returns
    -------
    A_bar : np.ndarray of shape (n_rho,), dtype float64
        Density-weighted average ion mass number profile.
    """
    if n_i.shape[0] != A_i.size:
        raise Exception(
            f'Shape mismatch between '
            f'ni_s ({n_i.shape}) and Ai_s ({A_i.shape})')
    
    n_i_sum = n_i.sum(0)
    A_bar = np.divide(
        np.dot(A_i, n_i).sum(), n_i_sum, out=np.zeros(n_i.shape[1]),
        where=n_i_sum != 0 )
    return A_bar

def calc_Z_box(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    n_i: Annotated[np.ndarray[np.float64], ('n_species', 'n_rho')],
    A_i: Annotated[np.ndarray[np.int32], ('n_species')],
    Z_i: Annotated[np.ndarray[np.int32], ('n_species')],
    v_f: float,
    A_f: int,
    Z_f: int,
    A_bar: Annotated[npt.NDArray[np.float64], ('n_rho')]=None
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate collisional effective nuclear charge.
    
    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    n_i : np.ndarray of shape (n_species, n_rho), dtype float64
        Thermal ion densities in inverse cubic centimeters (cm-3).
    A_i : np.ndarray of shape (n_species,), dtype int32
        Thermal ion mass numbers.
    Z_i : np.ndarray of shape (n_species,), dtype int32
        Thermal ion charge numbers.
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.
    A_bar : np.ndarray of shape (n_rho,), dtype float64
        Density-weighted average ion mass number profile. If `None`,
        then `A_bar` is calculated using `calc_A_bar(n_i, A_i)`. Default
        is `None`.

    Returns
    -------
    Z_box : np.ndarray of shape (n_rho,), dtype float64
        Collisional effective nuclear charge profile.
    """
    if A_bar is None: A_bar = calc_A_bar(n_i, A_i) # (n_rho,)
    log_lambda_ie = calc_Coulomb_ie(n_e, T_e) # (n_rho,)
    log_lambda_ii = calc_Coulomb_ii(
        n_e, T_e, A_i, Z_i, v_f, A_f, Z_f) # (n_rho,)

    coeff = np.divide(
        A_bar, n_e*log_lambda_ie, out=np.zeros_like(A_bar),
        where=n_e*log_lambda_ie != 0 ) # (n_rho,)
    
    Z_i_2 = Z_i**2
    Z_i_2_over_A_i = np.divide(
        Z_i_2, A_i, out=np.zeros(Z_i.size, dtype=float),
        where=A_i != 0 ) # (n_species,)
    arg = np.dot(Z_i_2_over_A_i, n_i*log_lambda_ii) # (n_rho,)
    return coeff * arg

def calc_Z_avg(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    n_i: Annotated[np.ndarray[np.float64], ('n_species', 'n_rho')],
    A_i: Annotated[np.ndarray[np.int32], ('n_species')],
    Z_i: Annotated[np.ndarray[np.int32], ('n_species')],
    v_f: float,
    A_f: int,
    Z_f: int
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate collisional average nuclear charge.
    
    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    n_i : np.ndarray of shape (n_species, n_rho), dtype float64
        Thermal ion densities in inverse cubic centimeters (cm-3). 
    A_i : np.ndarray of shape (n_species,), dtype int32
        Thermal ion mass numbers.
    Z_i : np.ndarray of shape (n_species,), dtype int32
        Thermal ion charge numbers.
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.

    Returns
    -------
    Z_avg : np.ndarray of shape (n_rho,), dtype float64
        Collisional average nuclear charge profile.
    """
    log_lambda_ie = calc_Coulomb_ie(n_e, T_e)
    log_lambda_ii = calc_Coulomb_ii(n_e, T_e, A_i, Z_i, v_f, A_f, Z_f)
    coeff = np.divide(
        1, log_lambda_ie*n_e, out=np.zeros_like(n_e),
        where=log_lambda_ie*n_e != 0)
    Z_i_2 = Z_i**2
    arg = np.dot(Z_i_2, n_i*log_lambda_ii)
    return coeff * arg

def calc_Z_eff(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    n_i: Annotated[np.ndarray[np.float64], ('n_species', 'n_rho')],
    Z_i: Annotated[np.ndarray[np.int32], ('n_species')]
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate effective nuclear charge.

    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    n_i : np.ndarray of shape (n_species, n_rho), dtype float64
        Thermal ion densities in inverse cubic centimeters (cm-3).
    Z_i : np.ndarray of shape (n_species,), dtype int32
        Thermal ion charge numbers.

    Returns
    -------
    Z_eff : np.ndarray of shape (n_rho,), dtype float64
        Effective nuclear charge profile.
    """
    nZ_i_2 = np.dot(Z_i**2, n_i)
    Z_eff = np.divide(
        nZ_i_2, n_e, out=np.zeros_like(n_e), where=n_e != 0 )
    return Z_eff
    
def calc_alpha_ell(
    ell: int,
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    v_f: float,
    A_f: int,
    Gamma: Annotated[npt.NDArray[np.float64], ('n_rho')],
    Z_eff: Annotated[npt.NDArray[np.float64], ('n_rho')],
    tau_cx: Annotated[npt.NDArray[np.float64], ('n_rho')]=None
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate $\alpha_\ell$.

    Parameters
    ----------
    ell : int
        Legendre index.
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.
    Gamma : np.ndarray of shape (n_rho,), dtype float64
        :math:`\Gamma` factor profile.
    Z_eff : np.ndarray of shape (n_rho,), dtype float64
        Effective nuclear charge profile. Calculated using
        `calc_Z_eff(n_e, n_i, Z_i)` or
        `calc_Z_avg(n_e, T_e, n_i, A_i, Z_i, v_f, A_f, Z_f)`.
    tau_cx : np.ndarray of shape (n_rho,), dtype float64
        Charge-exchange loss rate of fast ions. Default is None.

    Returns
    -------
    alpha_ell : np.ndarray of shape (n_rho,), dtype float64
        Velocity-space decay/growth profile.
    """
    inv_tau_cx = 0 if ((tau_cx is None) or (tau_cx == 0)) else 1/tau_cx

    v_f_3 = v_f**3
    pitch_angle_scattering = ell*(ell+1)*Z_eff*Gamma * np.divide(
        n_e, 2*v_f_3, out=np.zeros_like(n_e), where=v_f_3 != 0 )
    
    T_e_eV = T_e * 1e3 # keV -> eV
    T_e_1_5 = T_e_eV**1.5
    velocity_compression = -1.99e-20*Gamma*A_f * np.divide(
        n_e, T_e_1_5, np.zeros_like(n_e), where=T_e_1_5 != 0 )
    
    return inv_tau_cx + pitch_angle_scattering + velocity_compression

def calc_beta(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    v_f: float,
    A_f: int,
    A_bar: Annotated[npt.NDArray[np.float64], ('n_rho')],
    Gamma: Annotated[npt.NDArray[np.float64], ('n_rho')],
    Z_box: Annotated[npt.NDArray[np.float64], ('n_rho')]
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate :math:`\beta`.

    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    A_bar : np.ndarray of shape (n_rho,), dtype float64
        Density-weighted average ion mass number profile. Calculated
        using `calc_A_bar(n_i, A_i)`.
    Gamma : np.ndarray of shape (n_rho,), dtype float64
        :math:`\Gamma` factor profile.
    Z_box : np.ndarray of shape (n_rho,), dtype float64
        Collisional effective nuclear charge profile.
        Calculated using `calc_Z_box(n_e, T_e, n_i, A_i, Z_i, v_f, A_f,
        Z_f, A_bar=A_bar)`.

    Returns
    -------
    beta : np.ndarray of shape (n_rho,), dtype float64
        Thermal friction profile.
    """
    coeff = n_e * A_f * Gamma
    
    T_e_eV = T_e * 1e3 # keV -> eV
    T_e_1_5 = T_e_eV**1.5
    electron_friction = 6.62e-21 * np.divide(
        v_f, T_e_1_5, out=np.zeros_like(T_e), where=T_e_1_5 != 0 )
    
    v_f_2 = v_f**2
    ion_friction = np.divide(
        Z_box, A_bar*v_f_2, out=np.zeros_like(Z_box), where=A_bar*v_f_2 != 0 )
    
    return coeff * (electron_friction + ion_friction)
    
def calc_gamma(
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_i: Annotated[npt.NDArray[np.float64], ('n_rho')],
    v_f: float,
    A_bar: Annotated[npt.NDArray[np.float64], ('n_rho')],
    Gamma: Annotated[npt.NDArray[np.float64], ('n_rho')],
    Z_box: Annotated[npt.NDArray[np.float64], ('n_rho')]
) -> Annotated[npt.NDArray[np.float64], ('n_rho')]:
    r"""
    Calculate :math:`\gamma`.

    Parameters
    ----------
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    T_i : np.ndarray of shape (n_rho,), dtype float64
        Thermal ion temperature profile in kiloelectronvolts (keV).
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    A_bar : np.ndarray of shape (n_rho,), dtype float64
        Density-weighted average ion mass number profile. Calculated
        using `calc_A_bar(n_i, A_i)`.
    Gamma : np.ndarray of shape (n_rho,), dtype float64
        :math:`\Gamma` factor profile.
    Z_box : np.ndarray of shape (n_rho,), dtype float64
        Collisional effective nuclear charge profile.
        Calculated using `calc_Z_box(n_e, T_e, n_i, A_i, Z_i, v_f, A_f,
        Z_f, A_bar=A_bar)`.

    Returns
    -------
    gamma : np.ndarray of shape (n_rho,), dtype float64
        Thermal diffusion profile.
    """
    coeff = n_e * Gamma

    T_e_eV = T_e * 1e3 # keV -> eV
    sqrt_T_e = np.sqrt(T_e_eV, out=np.zeros_like(T_e), where=T_e_eV >= 0)
    electron_diffusion = np.divide(
        6.34e-9, sqrt_T_e, out=np.zeros_like(T_e), where=sqrt_T_e != 0)

    v_f_3 = v_f**3
    T_i_eV = T_i * 1e3 # keV -> eV
    ion_diffusion = 9.58e11 * T_i_eV * np.divide(
        Z_box, A_bar*v_f_3, out=np.zeros_like(Z_box), where=A_bar*v_f_3 != 0 )
    
    return coeff * (electron_diffusion + ion_diffusion)

def calc_steady_state_a_ell(
    velocity: Annotated[npt.NDArray[np.float64], ('n_velocity')],
    v_f: float,
    alpha_ell: Annotated[npt.NDArray[np.float64], ('n_rho')],
    beta: Annotated[npt.NDArray[np.float64], ('n_rho')],
    gamma: Annotated[npt.NDArray[np.float64], ('n_rho')],
    s_ell: float
) -> Annotated[npt.NDArray[np.float64], ('n_velocity', 'n_rho')]:
    r"""
    Calculate :math:`a_\ell` term for Goldston's steady-state solution
    to the Fokker-Planck equation.

    Parameters
    ----------
    velocity : np.ndarray of shape (n_velocity,), dtype float64
        Velocity grid in centimeters per second (cm/s).
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    alpha_ell : np.ndarray of shape (n_rho,), dtype float64
        Velocity-space decay/growth profile.
    beta : np.ndarray of shape (n_rho,), dtype float64
        Thermal friction profile.
    gamma : np.ndarray of shape (n_rho,), dtype float64
        Thermal diffusion profile.
    s_ell : float
        Legendre coefficient.

    Returns
    -------
    a_ell : np.ndarray of shape (n_velocity, n_rho), dtype float64
        Legendre coefficient for the steady-state solution in velocity
        space of the Fokker-Planck equation.
    """
    beta_2 = beta**2
    Delta_ell_arg = 1 + 4*alpha_ell*np.divide(
        gamma, beta_2, out=np.zeros_like(gamma), where=beta_2 != 0 ) # (n_rho,)
    Delta_ell = np.sqrt(
        Delta_ell_arg, out=np.zeros_like(Delta_ell_arg),
        where=Delta_ell_arg >= 0 ) # (n_rho,)
    n_rho = alpha_ell.size
    Delta_v = np.repeat((velocity - v_f)[:, np.newaxis], n_rho, axis=1) # (n_velocity, n_rho) cm/s

    n_velocity = velocity.size
    v_f_2 = v_f**2
    coeff_arg = 4 * np.pi * v_f_2 * beta * Delta_ell # (n_rho,)
    coeff = np.divide(
        1, coeff_arg, out=np.zeros_like(coeff_arg), where=coeff_arg != 0 ) # (n_rho,)
    coeff_grid = np.repeat(coeff[np.newaxis, :], n_velocity, axis=0) # (n_velocity, n_rho)
    
    Delta_ell_grid = np.repeat(Delta_ell[np.newaxis, :], n_velocity, axis=0) # (n_velocity, n_rho)
    beta_grid = np.repeat(beta[np.newaxis, :], n_velocity, axis=0) # (n_velocity, n_rho)
    gamma_grid = np.repeat(gamma[np.newaxis, :], n_velocity, axis=0) # (n_velocity, n_rho)
    arg = -(1 + np.sign(Delta_v)*Delta_ell_grid) * beta_grid * np.divide(
        Delta_v, 2*gamma_grid, out=np.zeros_like(Delta_v),
        where=gamma_grid != 0 ) # (n_velocity, n_rho)
    
    return s_ell * coeff_grid * np.exp(arg)

def calc_steady_state(
    pitch: Annotated[npt.NDArray[np.float64], ('n_pitch')],
    a_ell_grid: Annotated[
        npt.NDArray[np.float64], ('n_ell', 'n_velocity', 'n_rho')],
) -> Annotated[npt.NDArray[np.float64], ('n_velocity', 'n_pitch', 'n_rho')]:
    r"""
    Calculate Goldston's steady-state fast-ion distribution due to the
    Fokker-Planck equation.

    Parameters
    ----------
    pitch : np.ndarray of shape (n_pitch,), dtype float64
        Pitch grid.
    a_ell_grid : np.ndarray of shape (n_ell, n_velocity, n_rho), dtype float64
        Grid of Legendre coefficients for the steady-state solution in
        velocity space of the Fokker-Planck equation.

    Returns
    -------
    f_fastion : np.ndarray of shape (n_velocity, n_pitch, n_rho), dtype float64
        Fast-ion distribution.
    """
    n_ell = a_ell_grid.shape[0]

    leg_grid = np.array([
        legendre(ell)(pitch) for ell in range(n_ell) ]) # (n_ell, n_pitch)
    
    f_fastion = np.dot(leg_grid.T, a_ell_grid.swapaxes(0, 1)).swapaxes(0, 1)
    return f_fastion

def calc_f_fastion_vP(
    velocity: Annotated[npt.NDArray[np.float64], ('n_velocity')],
    pitch: Annotated[npt.NDArray[np.float64], ('n_pitch')],
    source_v_p: Annotated[npt.NDArray[np.float64], ('n_pitch')],
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    n_i: Annotated[np.ndarray[np.float64], ('n_species', 'n_rho')],
    A_i: Annotated[np.ndarray[np.int32], ('n_species')],
    Z_i: Annotated[np.ndarray[np.int32], ('n_species')],
    T_i: Annotated[npt.NDArray[np.float64], ('n_rho')],
    v_f: float,
    A_f: int,
    Z_f: int,
    n_ell: int=61,
    tau_cx: Annotated[npt.NDArray[np.float64], ('n_rho')]=None
) -> Annotated[npt.NDArray[np.float64], ('n_velocity', 'n_pitch', 'n_rho')]:
    r"""
    Calculate Goldston's steady-state fast-ion distribution due to the
    Fokker-Planck equation in velocity-pitch space.

    Parameters
    ----------
    velocity : np.ndarray of shape (n_velocity,), dtype float64
        Velocity grid in centimeters per second (cm/s).
    pitch : np.ndarray of shape (n_pitch,), dtype float64
        Pitch grid.
    source_v_p : np.ndarray of shape (n_pitch,), dtype float64
        Source grid vs pitch.
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    n_i : np.ndarray of shape (n_species, n_rho), dtype float64
        Thermal ion densities in inverse cubic centimeters (cm-3).
    A_i : np.ndarray of shape (n_species,), dtype int32
        Thermal ion mass numbers.
    Z_i : np.ndarray of shape (n_species,), dtype int32
        Thermal ion charge numbers.
    T_i : np.ndarray of shape (n_rho,), dtype float64
        Thermal ion temperature profile in kiloelectronvolts (keV).
    v_f : float
        Fast ion speed in centimeters per second (cm/s).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.
    n_ell : int, optional
        Number of Legendre coefficients. Default is 61.
    tau_cx : np.ndarray of shape (n_rho,), dtype float64
        Charge-exchange loss rate of fast ions. Default is None.

    Returns
    -------
    f_fastion_vP : np.ndarray of shape (n_velocity, n_pitch, n_rho), dtype float64
        Fast ion distribution as a function of velocity, pitch, and rho.
    """
    n_velocity = velocity.size
    n_rho = n_e.size

    Gamma = calc_Gamma(n_e, T_e, A_f, Z_f)
    Z_eff = calc_Z_avg(n_e, T_e, n_i, A_i, Z_i, v_f, A_f, Z_f)
    A_bar = calc_A_bar(n_i, A_i)
    Z_box = calc_Z_box(n_e, T_e, n_i, A_i, Z_i, v_f, A_f, Z_f, A_bar=A_bar)
    beta = calc_beta(n_e, T_e, v_f, A_f, A_bar, Gamma, Z_box)
    gamma = calc_gamma(n_e, T_e, T_i, v_f, A_bar, Gamma, Z_box)

    a_ell_grid = np.zeros((n_ell, n_velocity, n_rho))
    for ell in range(n_ell):
        s_ell = calc_s_ell_from_pitch(ell, pitch, source_v_p)
        alpha_ell = calc_alpha_ell(
            ell, n_e, T_e, v_f, A_f, Gamma, Z_eff, tau_cx=tau_cx)
        a_ell_grid[ell] = calc_steady_state_a_ell(
            velocity, v_f, alpha_ell, beta, gamma, s_ell)

    f_fastion_vP = calc_steady_state(pitch, a_ell_grid)
    f_fastion_vP[np.isnan(f_fastion_vP)] = 0.0
    return f_fastion_vP

def calc_f_fastion_EP(
    energy: Annotated[npt.NDArray[np.float64], ('n_energy')],
    pitch: Annotated[npt.NDArray[np.float64], ('n_pitch')],
    source_v_p: Annotated[npt.NDArray[np.float64], ('n_pitch')],
    n_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    T_e: Annotated[npt.NDArray[np.float64], ('n_rho')],
    n_i: Annotated[np.ndarray[np.float64], ('n_species', 'n_rho')],
    A_i: Annotated[np.ndarray[np.int32], ('n_species')],
    Z_i: Annotated[np.ndarray[np.int32], ('n_species')],
    T_i: Annotated[npt.NDArray[np.float64], ('n_rho')],
    E_f: float,
    A_f: int,
    Z_f: int,
    n_ell: int=61,
    tau_cx: Annotated[npt.NDArray[np.float64], ('n_rho')]=None
) -> Annotated[npt.NDArray[np.float64], ('n_energy', 'n_pitch', 'n_rho')]:
    r"""
    Calculate Goldston's steady-state fast-ion distribution due to the
    Fokker-Planck equation in velocity-pitch space.

    Parameters
    ----------
    energy : np.ndarray of shape (n_energy,), dtype float64
        Energy grid in kiloelectronvolts (keV).
    pitch : np.ndarray of shape (n_pitch,), dtype float64
        Pitch grid.
    source_v_p : np.ndarray of shape (n_pitch,), dtype float64
        Source grid vs pitch.
    n_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron density profile in inverse cubic centimeters
        (cm-3).
    T_e : np.ndarray of shape (n_rho,), dtype float64
        Thermal electron temperature profile in kiloelectronvolts (keV).
    n_i : np.ndarray of shape (n_species, n_rho), dtype float64
        Thermal ion densities in inverse cubic centimeters (cm-3).
    A_i : np.ndarray of shape (n_species,), dtype int32
        Thermal ion mass numbers.
    Z_i : np.ndarray of shape (n_species,), dtype int32
        Thermal ion charge numbers.
    T_i : np.ndarray of shape (n_rho,), dtype float64
        Thermal ion temperature profile in kiloelectronvolts (keV).
    E_f : float
        Fast ion energy in kiloelectronvolts (keV).
    A_f : int
        Fast ion atomic mass number.
    Z_f : int
        Fast ion nuclear charge number.
    n_ell : int, optional
        Number of Legendre coefficients. Default is 61.
    tau_cx : np.ndarray of shape (n_rho,), dtype float64
        Charge-exchange loss rate of fast ions. Default is None.

    Returns
    -------
    f_fastion_vP : np.ndarray of shape (n_velocity, n_pitch, n_rho), dtype float64
        Fast ion distribution as a function of velocity, pitch, and rho.
    """
    velocity = conv_kinetic_energy_to_speed(
        energy*1e3*J_PER_EV, A_f*HYDR_MASS_KG) * 100 # m/s -> cm/s
    v_f = conv_kinetic_energy_to_speed(
        E_f*1e3*J_PER_EV, A_f*HYDR_MASS_KG ) * 100 # m/s -> cm/s
    
    f_fastion_vP = calc_f_fastion_vP(
        velocity, pitch, source_v_p, n_e, T_e, n_i, A_i, Z_i, T_i, v_f,
        A_f, Z_f, n_ell=n_ell, tau_cx=tau_cx)
    f_fastion_EP = velocity[:, np.newaxis, np.newaxis] * f_fastion_vP
    return f_fastion_EP