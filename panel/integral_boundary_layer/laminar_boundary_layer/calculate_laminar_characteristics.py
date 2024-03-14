"""Obtains laminar boundary layer characteristics

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, A. Habermann

 Args:
    Air_prop        [-]     1-D array air properties
    M_inf           [-]     Free stream Mach number
    gamma           [-]     Specific heat ratio
    tr              [-]     Transition option (user input or apply model)
    Vx_e            [-]     Dimensionless X-component of the edge velocity (rectangular coordinates, divided by u_inf)
    Vy_e            [-]     Dimensionless Y-component of the edge velocity (rectangular coordinates, divided by u_inf)
    u_e             [-]     Dimensionless edge velocity (divided by u_inf)
    p_e             [Pa]    Static pressure at the edge of the boundary layer
    rho_e           [kg/m^3]    Density at the edge of the boundary layer
    M_e             [-]     Mach number at the edge of the boundary layer
    Xs              [m]     1-D array X-coordinate of discretized nodes
    r_0             [m]     1-D array Y-coordinate of discretized nodes (local transverse radius)
    S               [m]     1-D array Segment sizes
    phi             [rad]   1-D array Segment angle w.r.t symmetry axis

Returns: theta, Re_theta, dudx, lambda_, Xtr, C_f, H, delta, Re_x
    theta           [m]     Momentum thickness
    Re_theta        [-]     Reynolds number based on momentum thickness
    dudx            [1/s]   Rate of the edge velocity in the x-direction
    lambda_         [-]     Pressure gradient parameter
    Xtr             [-]     x/L position of transition (node position)
    C_f             [-]     Friction coefficient
    H               [-]     Shape factor
    delta           [m]     Boundary layer thickness
    Re_x            [-]     Reynolds number based on running length

Sources:
    [6] Rott, N. & Crabtree, L. F.: Simplifed Laminar Boundary-Layer Calculations
        for Bodies of Revolution and Yawed Wings. Journal of the Aeronautical Sciences
        19 (1952), 553-565.
    [7] Cebeci, T. & Bradshaw, P.: Physical and Computational Aspects of Convective
        Heat Transfer. Springer Berlin Heidelberg 1984, ISBN 978-3-662-02413-3.
    [8] Kays, W. & Crawford, M.: Convective Heat and Mass Transfer. McGraw-Hill
        series in mechanical engineering, McGraw-Hill 1993, ISBN 9780070337213.
"""

# Built-in/Generic Imports
import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import interpolate


def laminarCharacteristics(atmos, potentialSolution, surface, i):
    rho = atmos.ext_props['rho']  # Density [kg/m³]
    T_s = atmos.temperature  # Static temperature
    mu = atmos.ext_props['mue']  # Dynamic viscosity [Pa.s]
    nu = atmos.ext_props['nue']  # Kinematic viscosity [m²/s]
    u_inf = atmos.ext_props['u']
    gamma = atmos.ext_props['gamma']
    M_inf = atmos.ext_props['mach']
    u_e = potentialSolution[2] * u_inf
    rho_e = potentialSolution[4]
    M_e = potentialSolution[5]
    Xs = surface[0]
    r_0 = surface[1]
    S = surface[2]
    T_t = T_s * (1 + 0.5 * (gamma - 1) * M_inf ** 2)  # Stagnation temperature [K]
    T_e = T_t / (1 + 0.5 * (gamma - 1) * M_e ** 2)  # Temperature at edge of BL [K]
    nu_t = nu * (1 + 0.5 * (gamma - 1) * M_inf ** 2) ** (1 / (gamma - 1))  # Stagnation viscosity [m²/s]

    # [6], equ. 4.17
    theta = (0.45 * nu_t * (r_0[i] ** (-2)) * (u_e[i] ** (-6)) * (T_t / T_e[i]) ** 3 * (
        integ.simps(laminarIntegral(r_0[0:i + 1], u_e[0:i + 1], T_t, T_e[0:i + 1]), Xs[0:i + 1]))) ** 0.5
    Re_theta = u_e[i] * theta * rho / mu
    Re_x = u_e[i] * Xs[i] / nu_t

    dudx = (u_e[i] - u_e[i - 1]) / S[i]
    lambda_ = (theta ** 2 / nu) * dudx  # modified Pohlhausen parameter, pres. grad. lambda
    Xtr = Xs[i]
    delta = 4.64 * (nu * Xs[i] / u_e[i]) ** 0.5
    if lambda_ >= 0:  # Shape factor according to Thwaites (1960)
        if Re_theta == 0:
            C_f = 2 * (0.225 + 1.61 * lambda_ - 3.75 * lambda_ ** 2 + 5.24 ** lambda_ ** 3) / (
                    u_e[i] * Xs[i] / nu_t)
        else:
            C_f = 2 * (0.225 + 1.61 * lambda_ - 3.75 * lambda_ ** 2 + 5.24 ** lambda_ ** 3) / Re_theta
        H_i = 2.61 - 3.75 * lambda_ + 5.24 * lambda_ ** 2
    else:
        if Re_theta == 0:
            C_f = 2 * (0.225 + 1.472 * lambda_ + (0.0147 * lambda_) / (0.107 + lambda_)) / (
                    u_e[i] * Xs[i] / nu_t)
        else:
            C_f = 2 * (0.225 + 1.472 * lambda_ + (0.0147 * lambda_) / (0.107 + lambda_)) / Re_theta
        H_i = 0.0147 / (0.107 + lambda_) + 2.472
    # [6], equ. 4.14
    H = (T_t / T_e[i]) * H_i + (T_t / T_e[i]) - 1

    return theta, Re_theta, dudx, lambda_, Xtr, C_f, H, delta, Re_x


def laminarIntegral(r, u, T_t, T_e):
    return ((T_e / T_t) ** 1.5) * (r ** 2) * u ** 5
