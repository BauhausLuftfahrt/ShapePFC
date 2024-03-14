"""Create an Atmosphere object from the BHL Python Toolbox. Has to be replaced by own atmosphere model if used without
BHLPythonToolbox.

Author:  A. Habermann
"""

from bhlpythontoolbox.atmosphere.atmosphere import Atmosphere


def create_atmos(alt, mach, l_ref, dt_isa):
    # Atmospheric and Free-stream definitions
    atmos = Atmosphere(alt, dt_isa, ext_props_required=['rho', 'sos', 'mue', 'nue'], backend='bhl') # Standard atmospheric properties
    rho_inf = atmos.ext_props['rho']  # Density [kg/m³]
    c = atmos.ext_props['sos']  # Speed of sound [m/s]
    T_inf = atmos.temperature  # Static temperature
    p_inf = atmos.pressure  # Static pressure on free-stream [Pa]
    u_inf = mach * c  # Free-stream velocity [m/s]
    mu = atmos.ext_props['mue']  # Dynamic viscosity [Pa.s]
    nu = atmos.ext_props['nue']  # Kinematic viscosity [m²/s]
    gamma = 1.40  # Specific heat ratio [-]
    R = 287.058 # Specific gas constant [J kg^-1 K^-1]
    c_p = 1004.5
    r_f = 1  # Temperature recovery factor [-]
    p_t = p_inf * (1 + 0.5 * (gamma - 1) * mach ** 2) ** (gamma / (gamma - 1))
    # Reynolds number
    L_ref = l_ref
    Re_L = u_inf * L_ref / nu
    T_t_inf = T_inf * (1 + 0.5 * (gamma - 1) * mach ** 2)  # Total temperature [K]
    rho_t_inf = rho_inf * (1 + 0.5 * (gamma - 1) * mach ** 2) ** (1 / (gamma - 1))  # Total density [kg/m³]
    atmos.ext_props['mach'] = mach
    atmos.ext_props['R'] = R
    atmos.ext_props['gamma'] = gamma
    atmos.ext_props['re_l'] = Re_L
    atmos.ext_props['u'] = u_inf
    atmos.ext_props['p_t'] = p_t
    atmos.ext_props['T_t'] = T_t_inf
    atmos.ext_props['rho_t'] = rho_t_inf
    atmos.ext_props['c_p'] = c_p

    return atmos
