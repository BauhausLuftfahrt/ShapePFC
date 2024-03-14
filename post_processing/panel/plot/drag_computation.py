"""Compute body's pressure and viscous drag
    (for elliptic body: drag of a meridional section)

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, (Nikolaus Romanow)

 Args:
    tau_w           [Pa]        1-D array Wall shear stress
    p_s             [Pa]        1-D array Static pressure at body's surface
    alpha           [rad]       1-D array Segment angle w.r.t symmetry axis
    dS              [m]         1-D array Cumulative arc length of body contour
    r_0             [m]         1-D array Y-coordinate of discretized nodes (local transverse radius)  
    p_stag          [Pa]        Pressure at stagnation point (fuselage nose)
    end             [-]         index of last calculation point
    
    a               [m]         1-D array Semi-major axis of discretized nodes (for elliptic body)
    b               [m]         1-D array Semi-minor axis of discretized nodes (for elliptic body)
    theta           [rad]       Circumferential position (for elliptic body)

Returns:
    D_v                 [N]     Viscous drag
    D_p                 [N]     Pressure drag

"""

# Built-in/Generic Imports
import numpy as np
import scipy.integrate as integ


# Axisymmetric body
def dragBody(rho_inf, v_inf, p_inf, alpha, dS, C_p=None, p_s=None, C_f=None, tau=None):
    if C_p is not None:
        D_p = np.sum(np.sin(alpha)*C_p*rho_inf*v_inf**2*0.5*dS)
    elif p_s is not None:
        D_p = np.sum(np.sin(alpha)*(p_s-p_inf)*dS)
    else:
        raise Exception('Pressure drag cannot be calculated')

    if C_f is not None:
        D_v = np.sum(np.array(C_f)*rho_inf*v_inf**2*0.5*dS)
    elif tau is not None:
        D_v = np.sum(tau*dS)
    else:
        raise Exception('Skin friction drag cannot be calculated')

    return D_v, D_p


# Elliptic body
def dragSection(a, b, tau_w, p_s, alpha, dS, p_stag, theta):

    # Append values at stagnation point
    tau_dummy = tau_w[0] * 1
    a = np.append(0, a)
    b = np.append(0, b)
    tau_w = np.append(tau_dummy, tau_w)
    p_s = np.append(p_stag, p_s)
    alpha = np.append(np.pi / 2, alpha)
    dS = np.append(0, dS)

    # Viscous and pressure drag (sections)
    D_v_section = integ.trapz(((a**2)*(np.sin(theta)**2)+(b**2)*(np.cos(theta)**2))**0.5 * tau_w * np.cos(np.arcsin(np.sin(alpha))), dS)
    D_p_section = integ.trapz(((a**2)*(np.sin(theta)**2)+(b**2)*(np.cos(theta)**2))**0.5 * p_s * np.sin(np.arcsin(np.sin(alpha))), dS)

    return D_v_section, D_p_section