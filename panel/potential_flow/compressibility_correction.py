"""
Compressibility correction of velocity array according to Karman-Tsien.

Author:  A. Habermann

v_x, v_y    incompressible velocity x- and y- component arrays, on which compressibility correction should be applied
u_e         local boundary layer edge velocity
atmos       atmospheric (freestream) conditions

"""

import numpy as np


def karman_tsien_compr_corr(atmos, v, v_x, v_y):
    p_t_inf = atmos.ext_props['p_t']
    gamma = atmos.ext_props['gamma']
    c_inf = atmos.ext_props['sos']
    rho_inf = atmos.ext_props['rho']
    u_inf = atmos.ext_props['u']
    p_inf = atmos.pressure
    p_i = p_t_inf * (1 + 0.5 * (gamma - 1) * (v / c_inf) ** 2) ** (-gamma / (gamma - 1))
    Cp_i = (p_i - p_inf) / (0.5 * rho_inf * u_inf ** 2)
    M_i = v / c_inf  # for Karman-Tsien
    if np.all(M_i < 1):
        Cp_c = Cp_i / ((1 - M_i ** 2) ** 0.5 + (M_i ** 2) * (Cp_i / 2) / (
                1 + (1 - M_i ** 2) ** 0.5))
    else:
        Cp_c = np.zeros(np.shape(v))
        Cp_c[M_i < 1] = Cp_i / ((1 - M_i ** 2) ** 0.5 + (M_i ** 2) * (Cp_i / 2) / (1 + (1 - M_i ** 2) ** 0.5))
        Cp_c[M_i >= 1] = Cp_i / (np.abs(1 - M_i ** 2)) ** 0.5  # Prandtl-Glauert transformation, valid for 0.7<Ma<1.3
    p_c = (0.5 * rho_inf * u_inf ** 2) * Cp_c + p_inf  # for Karman-Tsien
    v_c = c_inf * ((2 / (gamma - 1)) * ((p_c / p_t_inf) ** (-(gamma - 1) / gamma) - 1)) ** 0.5  # for Karman-Tsien
    v_x_c = v_c * v_x / v
    v_y_c = v_c * v_y / v

    return v_x_c, v_y_c
