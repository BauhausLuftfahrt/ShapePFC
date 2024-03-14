"""Compute parameters on inlet of engine

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, (Nikolaus Romanow)

 Args:
    geom:                   Name of computations under Results/ to be post-processed
    idx:        [-]         index of node where the nacelle is located
    nac_h:      [m]         height of nacelle above body surface to compute inlet parameters
    delta       [m]         Boundary layer thickness
    u_inf       [m/s]       Freestream velocity
    u_e         [-]         Dimensionless edge velocity (divided by u_inf)
    n           [-]         Exponent of velocity profile power-law
    rho_e       [kg/m^3]    Density at the edge of the boundary layer
    r_0         [m]         Y-coordinate of discretized nodes (local transverse radius)
    gamma       [-]         Specific heat ratio
    c           [m/s]       Speed of sound
    phi         [rad]       Segment angle w.r.t symmetry axis
    p_inf       [Pa]        Freestream pressure
    M_inf       [-]         Freestream Mach number
    p_e         [Pa]        Static pressure at the edge of the boundary layer
    Theta:      [m^2]       Momentum deficit area
    d_theta:    [rad]       Circumferential angle for interpolation (elliptic fuselage)


Returns:
    mdot        [kg/s]      Mass flow across inlet
    momdef      [%]         Percentage of momentum thickness ingested
    y           [m]         1-D array Points across boundary layer to plot velocity profile
    ud          [m/s]       1-D array Velocity profile across the boundary layer
    PRP         [-]         1-D array radial distortion intensity [21]
    ang         [rad]       1-D array linear angular distribution to plot pressure contour
    r           [m]         1-D array linear radial distribution to plot pressure contour
    values      [Pa]        2-D array pressure distribution in inlet to plot contour
    p_tinf      [Pa]        Total or stagnation pressure
    pfav        [-]         Dimensionless inlet average total pressure (divided by p_tinf)
    pfma        [-]         Dimensionless inlet mass-averaged total pressure (divided by p_tinf)

Sources:
    [21] SAE International: Gas Turbine Engine Inlet Flow Distortion Guidelines.
        Aerospace Recommended Practice ARP1420 (2017).
"""

# Built-in/Generic Imports
import os
import numpy as np
import scipy.integrate as integ
from scipy import interpolate


#Axisymmetric body
def inletProfile(geom,idx,nac_h,delta,u_inf,u_e,n,rho_e,r_0,gamma,c,phi,p_inf,M_inf,p_e,Theta,filename):
    # Check if solution refers to CFD or integral computation and loads/calculates velocity profile
    if os.path.exists('results/%s/' %filename  + geom + '/' + geom + 'uProf0' + '.txt'):
        ud = np.loadtxt('results/%s/' %filename  + geom + '/' + geom + 'uProf' + str(idx) + '.txt')
        if nac_h > delta[idx]:
            ud = np.append(ud, ud[-1])
            y = np.linspace(0, delta[idx], 1000)
            y = np.append(y, nac_h)
        else:
            rat = nac_h / delta[idx]
            num = int(rat * 1000)
            ud = ud[0:num]
            y = np.linspace(0, nac_h, num)
    else:
        if nac_h > delta[idx]:
            y = np.linspace(0, delta[idx], 100)
            ud = u_inf * u_e[idx] *(y / delta[idx]) ** (1 / n[idx])
            y = np.append(y, nac_h)
            ud = np.append(ud, ud[-1])
        else:
            y = np.linspace(0, nac_h, 100)
            ud = u_inf * u_e[idx] * (y / delta[idx]) ** (1 / n[idx])

    # Calculate mass flow entering inlet (anular area from body up to nacelle height)
    mdot = rho_e[idx] * 2 * np.pi * integ.simps((r_0[idx] + y * np.cos(phi[idx])) * ud, y)

    # Calculate momentum deficit entering inlet (anular area from body up to nacelle height)
    momdef = integ.simps((r_0[idx] + y * np.cos(phi[idx])) * (ud/(u_inf * u_e[idx])) * (1 - (ud/(u_inf * u_e[idx]))), y) / Theta[idx]

    # Calculate pressure profile at inlet
    p_tinf = (p_inf * (1 + 0.5 * (gamma - 1) * (M_inf) ** 2) ** (gamma / (gamma - 1)))
    p_tot = (p_e[idx] * (1 + 0.5 * (gamma - 1) * (ud / c) ** 2) ** (
                gamma / (gamma - 1))) / p_tinf

    # Create array to plot pressure contour
    angle = np.radians(np.linspace(0, 360, 50))
    r, ang = np.meshgrid(y + r_0[idx], angle)
    values = np.ones((len(angle), len(y)))

    # Calculate inlet mass-averaged total pressure
    mdot_loc = rho_e[idx] * 2*np.pi * (r_0[idx] + y * np.cos(phi[idx])) * ud    # local mass flow rate
    pfma = integ.simps(mdot_loc * p_tot, y) / integ.simps(mdot_loc, y)

    # Calculate area weighed pressure average
    pfav = 2 * np.pi * integ.simps((r_0[idx] + y) * p_tot, y) / (np.pi * ((y[-1] + r_0[idx]) ** 2 - r_0[idx] ** 2))
    PRP = np.ones((len(y)))
    # Compute radial distortion according to SAE ARP1420 and fill values (for contour)
    for ii in range(0, len(y)):
        values[:, ii] = p_tot[ii] * values[:, ii]
        PRP[ii] = (pfav - p_tot[ii]) / pfav

    return mdot, momdef, y, ud, PRP, ang, r, values, p_tinf, pfav, pfma


# Elliptic body
def inletProfileElliptic(nac_h,delta,u_inf,u_e,n,rho_e,r_0,gamma,c,phi,p_inf,M_inf,p_e,Theta,d_theta):

    p_tinf = (p_inf * (1 + 0.5 * (gamma - 1) * (M_inf) ** 2) ** (gamma / (gamma - 1)))

    N_y = 100
    y = np.linspace(0, nac_h, N_y)

    mdot_section = np.zeros(len(d_theta))
    momdef = np.zeros(len(d_theta))
    ud_samples = np.zeros((N_y, len(d_theta)))
    p_tot_samples = np.zeros((N_y, len(d_theta)))

    for i in range(0, len(d_theta)):
        ud = u_inf * u_e[i] * (y / delta[i]) ** (1 / n[i])
        for ii in range(0, N_y):
            if y[ii] > delta[i]:
                ud[ii] = u_inf * u_e[i] * 1 ** (1 / n[i])           # maximum value of velocity profile at boundary edge
        ud_samples[:, i] = ud * 1

        mdot_section[i] = integ.simps(rho_e[i] * (r_0 + y * np.cos(phi)) * ud, y)

        # Calculate momentum deficit entering inlet (anular area from body up to nacelle height)
        momdef[i] = integ.simps((r_0 + y * np.cos(phi)) * (ud / (u_inf * u_e[i])) * (1 - (ud / (u_inf * u_e[i]))), y) / Theta[i]

        p_tot_samples[:, i] = (p_e[i] * (1 + 0.5 * (gamma - 1) * (ud / c) ** 2) ** (gamma / (gamma - 1))) / p_tinf


    # Calculate mass flow entering inlet (anular area from body up to nacelle height)
    mdot_total = integ.simps(mdot_section, d_theta)

    # Calculate pressure profile at inlet
    p_tot = np.zeros(N_y)
    mdot_loc = np.zeros(N_y)
    for i in range(0, N_y):
        p_tot[i] = integ.simps((p_tot_samples[i, :]), d_theta) / (2 * np.pi)
        int_test = (r_0 + y[i] * np.cos(phi)) * ud_samples[i, :]
        mdot_loc[i] = 2 * np.pi * integ.simps(rho_e * int_test, d_theta)  # local mass flow rate

    # Create array to plot pressure contour
    angle = np.linspace(0, 360, 1000)
    angle = angle + 90                      # semi-major axis should be in vertical direction (fuselage height)
    for i in range(len(angle)):
        if angle[i] > 360:
            angle[i] = angle[i] - 360
    angle = np.radians(angle)
    r, ang = np.meshgrid(y + r_0, angle)
    values = np.ones((len(angle), len(y)))

    # Calculate mass-averaged fan-face total pressure
    pfma = integ.simps(mdot_loc * p_tot, y) / integ.simps(mdot_loc, y)

    # Calculate area weighed pressure average
    pfav = 2 * np.pi * integ.simps((r_0 + y) * p_tot, y) / (np.pi * ((y[-1] + r_0) ** 2 - r_0 ** 2))
    PRP = np.ones(N_y)

    # Compute radial distortion according to SAE ARP1420 and fill values (for contour)
    for ii in range(0, N_y):
        for i in range(0, len(d_theta)):
            values[i, ii] = p_tot_samples[ii, i] * values[i, ii]
        PRP[ii] = (pfav - p_tot[ii]) / pfav

    return mdot_total, momdef, y, ud_samples, PRP, ang, r, values, p_tinf, pfav, pfma


# Function made only to plot profiles from waisted body experiment
def experimentalProfile(pos,Uprofexp,delta_exp,nac_h,idx,rho_e,u_e,u_inf,r_0,phi):
    if len(Uprofexp) <= 2:
        return 0,0,0
    if np.isclose(pos, 0.4, rtol=2e-2):
        flg = 1
        up = np.asarray(Uprofexp[0])
        del_exp = delta_exp[0, 1]
    elif np.isclose(pos, 0.475, rtol=2e-2):
        flg = 1
        up = np.asarray(Uprofexp[1])
        del_exp = delta_exp[1, 1]
    elif np.isclose(pos, 0.55, rtol=5e-2):
        flg = 1
        up = np.asarray(Uprofexp[2])
        del_exp = delta_exp[2, 1]
    elif np.isclose(pos, 0.7, rtol=5e-2):
        flg = 1
        up = np.asarray(Uprofexp[3])
        del_exp = delta_exp[3, 1]
    elif np.isclose(pos, 0.83, rtol=5e-2):
        flg = 1
        up = np.asarray(Uprofexp[4])
        del_exp = delta_exp[4, 1]
    elif np.isclose(pos, 0.98, rtol=5e-2):
        flg = 1
        up = np.asarray(Uprofexp[5])
        del_exp = delta_exp[5, 1]
    else:
        flg = 0
        return 0, flg, 0

    if nac_h > del_exp:
        up = np.vstack([up, [up[-1, 0], nac_h]])
    else:
        rat = nac_h / del_exp
        num = int(rat * len(up[:, 0]))
        up = up[0:num, :]
        dp = interpolate.interp1d(up[:, 1], up[:, 0], fill_value="extrapolate")
        inter = dp(nac_h)
        up = np.vstack([up, [inter, nac_h]])
    mdot = rho_e[idx] * u_inf * u_e[idx] * 2 * np.pi * integ.simps((r_0[idx] + up[:, 1] * np.cos(phi[idx])) * up[:, 0],
                                                                   up[:, 1])

    return up, flg, mdot