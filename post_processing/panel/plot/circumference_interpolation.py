
""" Interpolate boundary layer characteristics

Author:  Nikolaus Romanow

 Args:
    idx:        [-]         index of node where to interpolate
    a_ell       [m]         Semi-major axis
    b_ell       [m]         Semi-minor axis
    d_theta:    [rad]       Circumferential positions for interpolation (elliptic fuselage)
    
    Boundary layer characteristics at semi-major ('_a') and semi-minor axis ('_b')
        delta       [m]         Boundary layer thickness
        rhoe        [kg/m^3]    Density at the edge of the boundary layer
        ue          [-]         Dimensionless edge velocity (divided by u_inf)
        n           [-]         Exponent of velocity profile power-law
        Theta:      [m^2]       Momentum deficit area
        pe          [Pa]        Static pressure at the edge of the boundary layer
        dS          [m]         Arc length function of body contour
        tauw        [Pa]        Wall shear stress
        ps          [Pa]        Static pressure at body's surface
        alpha       [rad]       Segment angle w.r.t symmetry axis

Returns:
    Interpolated boundary layer characteristics ('_samples')

"""


# Built-in/Generic Imports
import numpy as np
from scipy import interpolate

def circInterp(idx, delta_a,delta_b, rhoe_a,rhoe_b, ue_a,ue_b, n_a,n_b, Theta_a,Theta_b, pe_a, pe_b,
               a_ell,b_ell, dS_a,dS_b, tauw_a,tauw_b, ps_a,ps_b, alpha_a,alpha_b,
               d_theta):

    # Initialize sample arrays
    dS_samples = np.zeros((len(d_theta), len(dS_a)))
    tauw_samples = np.zeros((len(d_theta), len(tauw_a)))
    ps_samples = np.zeros((len(d_theta), len(ps_a)))
    alpha_samples = np.zeros((len(d_theta), len(alpha_a)))

    for j in range(0, len(dS_a)):
        dr_ell = ((a_ell[j] ** 2) * (np.cos(d_theta) ** 2) + (b_ell[j] ** 2) * (np.sin(d_theta) ** 2)) ** 0.5
        for k in range(0, len(dr_ell)):     # Exclude rounding errors (dr_ell cannot be outside of [b, a] due to interpolation)
            if dr_ell[k] < b_ell[j]:
                dr_ell[k] = b_ell[j]*1
            if dr_ell[k] > a_ell[j]:
                dr_ell[k] = a_ell[j]*1

        # Find interpolation function for every "surface ring" along body length
        dS_ip = interpolate.interp1d([a_ell[j], b_ell[j], a_ell[j], b_ell[j], a_ell[j]], [dS_a[j], dS_b[j], dS_a[j], dS_b[j], dS_a[j]], kind='linear')
        tauw_ip = interpolate.interp1d([a_ell[j], b_ell[j], a_ell[j], b_ell[j], a_ell[j]], [tauw_a[j], tauw_b[j], tauw_a[j], tauw_b[j], tauw_a[j]], kind='linear')
        ps_ip = interpolate.interp1d([a_ell[j], b_ell[j], a_ell[j], b_ell[j], a_ell[j]], [ps_a[j], ps_b[j], ps_a[j], ps_b[j], ps_a[j]], kind='linear')
        alpha_ip = interpolate.interp1d([a_ell[j], b_ell[j], a_ell[j], b_ell[j], a_ell[j]], [alpha_a[j], alpha_b[j], alpha_a[j], alpha_b[j], alpha_a[j]], kind='linear')

        # Find interpolated values for every "surface ring" along body length
        dS_samples[:, j] = dS_ip(dr_ell)
        ps_samples[:, j] = ps_ip(dr_ell)
        tauw_samples[:, j] = tauw_ip(dr_ell)
        alpha_samples[:, j] = alpha_ip(dr_ell)

        if j == idx:
            dr_idx = dr_ell[:]

    # Find interpolation function for "ring" at idx (x-position of nacelle)
    a_idx = a_ell[idx]
    b_idx = b_ell[idx]
    delta_ip = interpolate.interp1d([a_idx, b_idx, a_idx, b_idx, a_idx], [delta_a, delta_b, delta_a, delta_b, delta_a], kind='linear')
    rhoe_ip = interpolate.interp1d([a_idx, b_idx, a_idx, b_idx, a_idx], [rhoe_a, rhoe_b, rhoe_a, rhoe_b, rhoe_a], kind='linear')
    ue_ip = interpolate.interp1d([a_idx, b_idx, a_idx, b_idx, a_idx], [ue_a, ue_b, ue_a, ue_b, ue_a], kind='linear')
    n_ip = interpolate.interp1d([a_idx, b_idx, a_idx, b_idx, a_idx], [n_a, n_b, n_a, n_b, n_a], kind='linear')
    Theta_ip = interpolate.interp1d([a_idx, b_idx, a_idx, b_idx, a_idx], [Theta_a, Theta_b, Theta_a, Theta_b, Theta_a], kind='linear')
    pe_ip = interpolate.interp1d([a_idx, b_idx, a_idx, b_idx, a_idx], [pe_a, pe_b, pe_a, pe_b, pe_a], kind='linear')

    # Find interpolated values for "ring" at idx
    delta_samples = delta_ip(dr_idx)
    rhoe_samples = rhoe_ip(dr_idx)
    ue_samples = ue_ip(dr_idx)
    n_samples = n_ip(dr_idx)
    Theta_samples = Theta_ip(dr_idx)
    pe_samples = pe_ip(dr_idx)


    return delta_samples, rhoe_samples, ue_samples, n_samples, Theta_samples, pe_samples, \
           dS_samples, ps_samples, tauw_samples, alpha_samples