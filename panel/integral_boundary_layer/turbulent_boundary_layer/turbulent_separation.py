"""Determine existence of point of turbulent separation.

Author:  A. Habermann

 Args:
    thetapl_in      [m]     Planar momentum thickness at the first turbulent point
    Hpl_in          [-]     Planar shape factor at the first turbulent point
    Vx_e            [-]     Dimensionless X-component of the edge velocity (rectangular coordinates, divided by u_inf)
    Vy_e            [-]     Dimensionless Y-component of the edge velocity (rectangular coordinates, divided by u_inf)
    u_e             [-]     1-D array Dimensionless edge velocity (divided by u_inf)
    p_e             [Pa]    1-D array Static pressure at the edge of the boundary layer
    rho_e           [kg/m^3]    1-D array Density at the edge of the boundary layer
    M_e             [-]     1-D array Mach number at the edge of the boundary layer
    Xs              [m]     1-D array X-coordinate of discretized nodes
    r_0             [m]     1-D array Y-coordinate of discretized nodes (local transverse radius)
    S               [m]     1-D array Segment sizes
    phi             [rad]   1-D array Segment angle w.r.t symmetry axis
    eps             [-]     Relative Tolerance for convergence check
    end             [-]     index of last calculation point
    counter         [-]     Number of viscid/inviscid iterations
    r_f             [-]     Temperature recovery factor
    Air_prop        [-]     Tuple air properties
    M_inf           [-]     Freestream Mach number
    phi_d           [rad]   1-D array Flow angle w.r.t symmetry axis
    filled up to transition point:
    Theta:          [m^2]     1-D array Momentum deficit area
    H               [-]     1-D array Shape factor
    delta           [m]     1-D array Boundary layer thickness
    C_f             [-]     1-D array Friction coefficient
    n               [-]     1-D array Exponent of velocity profile power-law
    delta_starPhys  [m]     1-D array Displacement thickness
    p_s             [Pa]    1-D array Static pressure at body's surface
    theta           [m]     1-D array Momentum thickness
    Delta_star      [m^2]     1-D array Displacement area
    C_e             [-]     1-D array Entrainment coefficient

Returns:
    Theta:          [m^2]     1-D array Momentum deficit area
    H               [-]     1-D array Shape factor
    delta           [m]     1-D array Boundary layer thickness
    C_f             [-]     1-D array Friction coefficient
    n               [-]     1-D array Exponent of velocity profile power-law
    delta_starPhys  [m]     1-D array Displacement thickness
    p_s             [Pa]    1-D array Static pressure at body's surface
    theta           [m]     1-D array Momentum thickness
    Delta_star      [m^2]   1-D array Displacement area

Sources:
    [1] Olson, L.E. and Dvorak, F.A. Viscous/potential flow about multi-element two-dimensional and infinite-span swept 
    wings - Theory and experiment. 14tj Aerospace Sciences Meeting. 1976.
"""

# Built-in/Generic Imports
import numpy as np
import warnings


def turbulent_separation(Theta, H, delta, delta_starPhys, p_s, theta, Delta_star, Xs, k, u_e):
    # identify point of separation of turbulent boundary layer
    if np.any(H[k + 1:] > 2.8):
        idx_sep = np.where(H > 2.8)
        idx_turbsep = idx_sep[0][idx_sep[0] > k]
        x_turbsep = Xs[idx_turbsep[0]]
        xc_turb = x_turbsep / Xs[-1]
        # Asspt.: const. static pressure in separated flow -> const. u_e
        p_s[idx_turbsep[0] + 1:] = p_s[idx_turbsep[0]]
        u_e[idx_turbsep[0] + 1:] = u_e[idx_turbsep[0]]
        H[idx_turbsep[0]:] = 2.8
        # lin. extrapolation of displacement thickness/area and boundary layer thickness in separated flow from point
        # of separation
        delta_starPhys[idx_turbsep[0] + 1:] = [delta_starPhys[idx_turbsep[0]] +
                                               (delta_starPhys[idx_turbsep[0]] - delta_starPhys[idx_turbsep[0] - 1]) *
                                               (Xs[i] - Xs[idx_turbsep[0]]) / (
                                                           Xs[idx_turbsep[0]] - Xs[idx_turbsep[0] - 1])
                                               for i in range(idx_turbsep[0] + 1, len(delta_starPhys))]
        Delta_star[idx_turbsep[0] + 1:] = [Delta_star[idx_turbsep[0]] +
                                           (Delta_star[idx_turbsep[0]] - Delta_star[idx_turbsep[0] - 1]) *
                                           (Xs[i] - Xs[idx_turbsep[0]]) / (Xs[idx_turbsep[0]] - Xs[idx_turbsep[0] - 1])
                                           for i in range(idx_turbsep[0] + 1, len(Delta_star))]
        delta[idx_turbsep[0] + 1:] = [delta[idx_turbsep[0]] +
                                      (delta[idx_turbsep[0]] - delta[idx_turbsep[0] - 1]) *
                                      (Xs[i] - Xs[idx_turbsep[0]]) / (Xs[idx_turbsep[0]] - Xs[idx_turbsep[0] - 1])
                                      for i in range(idx_turbsep[0] + 1, len(delta))]
        # calculation of momentum thickness/area
        theta[idx_turbsep[0] + 1:] = [delta_starPhys[i] / H[i] for i in range(idx_turbsep[0] + 1, len(theta))]
        Theta[idx_turbsep[0] + 1:] = [delta_starPhys[i] / H[i] for i in range(idx_turbsep[0] + 1, len(Theta))]

        warnings.warn(f"Separation in turbulent flow region detected at x/c={xc_turb}.")

    return Theta, H, delta, delta_starPhys, p_s, theta, Delta_star
