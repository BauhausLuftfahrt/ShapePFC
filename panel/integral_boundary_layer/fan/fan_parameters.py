import numpy as np
import scipy.integrate as integ
from scipy import interpolate
from scipy.optimize import fsolve


def rotorAveraged(x_rot, d_rot, u_inf, p_inf, M_inf, delta, u_e, p_e, rho_e, Xs, r_0, phi, n, p_s, end, gamma, R, c):
    """Compute mass-averaged boundary layer characteristics at rotor inlet
    Author:  Nikolaus Romanow
     Args:
        x_rot       [m]         X-position rotor
        d_rot       [m]         Rotor blade height
        u_inf       [m/s]       Freestream velocity
        p_inf       [Pa]        Freestream pressure
        M_inf       [-]         Freestream Mach number
        delta       [m]         Boundary layer thickness
        u_e         [-]         Dimensionless edge velocity (divided by u_inf)
        p_e         [Pa]        Static pressure at the edge of the boundary layer
        rho_e       [kg/m^3]    Density at the edge of the boundary layer
        Xs          [m]         X-coordinate of discretized nodes
        r_0         [m]         Y-coordinate of discretized nodes (local transverse radius)
        phi         [rad]       Segment angle w.r.t symmetry axis
        n           [-]         Exponent of velocity profile power-law
        p_s         [Pa]        Static pressure at body's surface
        end         [-]         Index of last calculation point
        gamma       [-]         Specific heat ratio
        R           [J kg^-1 K^-1]  Specific gas constant
        c           [m/s]       Speed of sound
    Returns:
        u_ma        [m/s]       Mass flow averaged velocity at rotor inlet
        mdot        [kg/s]      Mass flow at rotor inlet
        Tt_ma       [K]         Mass flow averaged total temperature at rotor inlet
        pt_ma       [K]         Mass flow averaged total pressure at rotor inlet
        T_ma        [K]         Mass flow averaged static temperature at rotor inlet
        p_ma        [Pa]        Mass flow averaged static pressure at rotor inlet
        rho_ma      [Pa]        Density at rotor inlet
        Ma_ma       [-]         Mass flow averaged Mach number at rotor inlet
    """

    delta_ip = interpolate.interp1d(Xs[0:end - 1], delta[0:end - 1], fill_value="extrapolate")
    ue_ip = interpolate.interp1d(Xs[0:end - 1], u_e[0:end - 1], fill_value="extrapolate")
    n_ip = interpolate.interp1d(Xs[0:end - 1], n[0:end - 1], fill_value="extrapolate")
    rhoe_ip = interpolate.interp1d(Xs[0:end - 1], rho_e[0:end - 1], fill_value="extrapolate")
    r0_ip = interpolate.interp1d(Xs[0:end - 1], r_0[0:end - 1], fill_value="extrapolate")
    alpha = phi * 1  # continuous body contour angle
    for j in range(len(Xs)):
        if phi[j] >= (3 / 2) * np.pi:
            alpha[j] = phi[j] - 2 * np.pi
        else:
            alpha[j] = phi[j]
    phi_ip = interpolate.interp1d(Xs[0:end - 1], alpha[0:end - 1], fill_value="extrapolate")
    pe_ip = interpolate.interp1d(Xs[0:end - 1], p_e[0:end - 1], fill_value="extrapolate")
    ps_ip = interpolate.interp1d(Xs[0:end - 1], p_s[0:end - 1], fill_value="extrapolate")

    if d_rot > delta_ip(x_rot):
        y = np.linspace(0, delta_ip(x_rot), 100)
        ud = ue_ip(x_rot) * (y / delta_ip(x_rot)) ** (1 / n_ip(x_rot))                          # Velocity power law
        pd = pe_ip(x_rot) + (ps_ip(x_rot) - pe_ip(x_rot)) * (1 - (y / delta_ip(x_rot)) ** 2)    # Pressure profile Patel
        y = np.append(y, d_rot)
        ud = np.append(ud, ud[-1])
        pd = np.append(pd, pd[-1])
    else:
        y = np.linspace(0, d_rot, 100)
        ud = ue_ip(x_rot) * (y / delta_ip(x_rot)) ** (1 / n_ip(x_rot))
        pd = pe_ip(x_rot) + (ps_ip(x_rot) - pe_ip(x_rot)) * (1 - (y / delta_ip(x_rot)) ** 2)

    Td = pd / (R * rhoe_ip(x_rot))

    # Calculate mass flow entering fan face (annular area from body up to nacelle height)
    mdot = rhoe_ip(x_rot) * 2 * np.pi * integ.simps((r0_ip(x_rot) + y * np.cos(phi_ip(x_rot))) * ud, y)

    # Calculate total pressure and temperature profile at fan face
    p_tot = (pd * (1 + 0.5 * (gamma - 1) * (ud / np.sqrt(gamma*R*Td)) ** 2) ** (
            gamma / (gamma - 1)))
    #rho_tot = (rhoe_ip(x_rot) * (1 + 0.5 * (gamma - 1) * (ud / np.sqrt(gamma*R*Td)) ** 2) ** (
            #1 / (gamma - 1)))
    T_tot = (Td * (1 + 0.5 * (gamma - 1) * (ud / np.sqrt(gamma*R*Td)) ** 2))

    # Calculate mass-averaged fan-face properties
    mdot_loc = rhoe_ip(x_rot) * 2*np.pi * (r0_ip(x_rot) + y * np.cos(phi_ip(x_rot))) * ud    # local mass flow rate
    u_ma = integ.simps(mdot_loc * ud, y) / integ.simps(mdot_loc, y)
    Ma_ma = u_ma/c

    pt_ma = integ.simps(mdot_loc * p_tot, y) / integ.simps(mdot_loc, y)
    p_ma = integ.simps(mdot_loc * pd, y) / integ.simps(mdot_loc, y)

    #rhot_ma = integ.simps(mdot_loc * rho_tot, y) / integ.simps(mdot_loc, y)
    rho_ma = rhoe_ip(x_rot)

    Tt_ma = integ.simps(mdot_loc * T_tot, y) / integ.simps(mdot_loc, y)
    T_ma = integ.simps(mdot_loc * Td, y) / integ.simps(mdot_loc, y)

    return u_ma, mdot, Tt_ma, pt_ma, T_ma, p_ma, rho_ma, Ma_ma


def thermoSys(mdot, Tt, pt, A, gamma, R, init):
    """Solve thermodynamic system of equations (to compute boundary layer characteristics at stator outlet)
    Author:  Nikolaus Romanow
     Args:
        mdot        [kg/s]      Mass flow (at stator outlet)
        Tt          [K]         Total temperature (at stator outlet)
        pt          [Pa]        Total pressure (at stator outlet)
        A           [m^2]       (Stator) area
        gamma       [-]         Specific heat ratio
        R           [J kg^-1 K^-1]  Specific gas constant
        init                    Initial guess vector for numerical solver (boundary layer characteristics at rotor inlet)
    Returns:
        x_sol                   Solution vector (Static temperature, static pressure, density, velocity, Mach number at stator outlet)
    """

    # system of equations
    def func1(x):
        out1= [Tt - x[0] * (1 + ((gamma-1)/2) * x[4]**2),
               pt - x[1] * (1 + ((gamma-1)/2) * x[4]**2)**(gamma/(gamma-1)),
               mdot - x[2] * x[3] * A,
               x[2] - (x[1]/(R*x[0])),
               x[4] - (x[3]/(gamma*R*x[0])**0.5)
         ]
        return out1

    # solve system of equations numerically with initial guess
    x_sol = fsolve(func1, init, xtol=1e-12, factor=0.1, maxfev=10000)
    #print("T, p, rho, u, Ma:", x_sol)

    return x_sol
