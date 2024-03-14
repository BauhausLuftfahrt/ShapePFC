"""Calculate ISA standard atmosphere based on altitude.

Author:  A. Habermann
"""

import warnings
from typing import Union, Tuple

import numpy as np
from CoolProp.CoolProp import PropsSI
from numpy.core.multiarray import ndarray
from scipy import constants as const


def atmosphere(alt: Union[ndarray, float], dt_isa: Union[ndarray, float] = 0.) -> Tuple[ndarray, ndarray]:
    """Sets the ICAO standard atmosphere properties to the input conditions

    Authors: Hagen Kellermann & Anais Habermann

     Args:
        alt:       [m]     altitude
        dt_isa:    [K]     isa temperature deviation

    Returns:
        p_s:        [Pa]    local fluid pressure
        t_s:        [K]     local fluid temperature

    Sources:
        [1] ICAO: Manual of the ICAO Standard Atmosphere, 3rd edition, 1993
    """

    # Check for input out of range or of faulty dimension
    h_max = 84.852
    if np.any(alt / 1e3 > h_max) or np.any(alt < 0):
        raise ValueError('Altitude out of range')
    if np.ndim(np.squeeze(alt)) > 1 or np.ndim(np.squeeze(dt_isa)) > 1:
        raise ValueError('Input number of dim > 1: shape(alt) = ' + str(np.shape(alt)) + '; shape(dt_isa) = ' + str(
            np.shape(dt_isa)))
    if np.shape(alt) != np.shape(dt_isa) and not (np.size(alt) == 1 or np.size(dt_isa) == 1):
        raise ValueError(
            'Input dimension missmatch: size(alt) = ' + str(np.size(alt)) + '; size(dt_isa): ' + str(np.size(dt_isa)))

    dim1 = np.max([np.size(alt), np.size(dt_isa)])

    # Definition of constants
    g_0 = const.g  # gravitational acceleration [m/s^2]
    r_0 = const.R  # universal gas constant

    # Air properties (ideal gas)
    mol = PropsSI("M", "air") * 1000  # molecular weight of air [g/mol] required for matching later calculation units
    # /ref: APA, HKe, AS

    # Sea Level Ambient Conditions:
    p_0 = 101325  # Standard Pressure at Sevel Level [Pa]

    # International Standard Atmosphere Key Data (1976 Std. Atmosphere):
    h_tab = np.array([0., 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, h_max])  # Altitude Start Values for Atmosphere [m]
    t_tab = np.array(
        [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])  # Temperature Start Values for Atmosphere[K]
    p_tab = np.array([1.0, 0.2233611, 0.024561, 0.0085666784, 0.0010945601, 0.00066063531, 0.000039046834,
                      0.000000368501])  # Ambient Pressure Start Values for Atmosphere [Pa]
    g_tab = np.array([-6.5, 0., 1., 2.8, 0., -2.8, -2., 0.])  # Temperature Gradients for Atmosphere Layers [K/m]

    alt = np.squeeze(alt) / 1000  # altitude conversion to [km]

    # Preallocate arrays
    k = np.empty(shape=(dim1, 7))  # Boolean array to choose correct t_s & dp for each element of alt
    t_s = np.empty(shape=(dim1, 7))
    d_p = np.empty(shape=(dim1, 7))

    # Sometimes runtime warnings occur in power. Negligible because cases are filtered out by k
    warnings.simplefilter("ignore", RuntimeWarning)

    for i in range(1, len(h_tab)):
        k[:, i - 1] = np.logical_and(alt >= h_tab[i - 1], alt < h_tab[i])

        # calculate local temperature
        t_s[:, i - 1] = t_tab[i - 1] + dt_isa + g_tab[i - 1] * (alt - h_tab[i - 1])

        # calculate pressure ratio
        if g_tab[i - 1] == 0:
            d_p[:, i - 1] = p_tab[i - 1] * np.exp(-g_0 * mol / r_0 * (alt - h_tab[i - 1]) / t_tab[i - 1])
        else:
            d_p[:, i - 1] = p_tab[i - 1] * (1 + g_tab[i - 1] / t_tab[i - 1] * (alt - h_tab[i - 1])) ** (
                    -g_0 * mol / r_0 / (g_tab[i - 1]))

    # Ensure that no relevant runtime warning was ignored
    if np.any(k[np.isnan(d_p)]):
        raise RuntimeError("NaN calculated on relevant position in d_p array")

    d_p[np.isnan(d_p)] = 0
    t_s = np.sum(t_s * k, 1)
    d_p = np.sum(d_p * k, 1)

    # calculate local pressure (source: APA, Arne Seitz. Bit intricate but due to choice of p_tab correct)
    p_s = d_p * p_0

    return t_s, p_s


def std_atmosphere(alt: Union[ndarray, float], dt_isa: Union[ndarray, float] = 0.) -> Tuple[
    ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Sets the ICAO standard atmosphere properties to the input conditions

        Authors: Florian Troeltsch

         Args:
            alt:       [m]     1-D array altitude
            dt_isa:    [K]     1-D array isa temperature deviation

        Returns:
            rho:      [kg/m³]   1-D array ISA density
            sos:      [m/s]     1-D array ISA speed of sound
            t_s:      [K]       1-D array ISA temperature
            p_s:      [Pa]      1-D array ISA pressure
            mue:      [Pa*s]    1-D array ISA dynamic viscosity
            nue:      [m^2/s]   1-D array ISA kinematic viscosity
    """

    # Calculate Air properties with Cool Props Library
    t_s, p_s = atmosphere(alt, dt_isa)
    sos = PropsSI('A', 'T', t_s, 'P', p_s, 'Air')
    mue = PropsSI('V', 'T', t_s, 'P', p_s, 'Air')
    rho = PropsSI('D', 'T', t_s, 'P', p_s, 'Air')
    nue = mue / rho
    return rho, sos, t_s, p_s, mue, nue


if __name__ == "__main__":
    ts, ps = atmosphere(np.linspace(0, 10000, 10))
    print('ts: ' + str(ts))
    print('ps: ' + str(ps))
