"""Calculate freestream characteristics.

Author:  A. Habermann
"""

import numpy as np


def calc_freestream(atmos, l_ref, wall_res):
    # h_duct = geometry.h_duct
    # mach = atmos.ext_props['mach']
    altitude = atmos.altitude
    L_ref = l_ref  # geometry.x[0][-1]-geometry.x[0][0]
    Int = 0.01  # turbulent intensity [-] https://www.cfd-online.com/Wiki/Turbulence_intensity
    L_mix = 1e-5  # mixing length / turbulent length scale [m] https://www.cfd-online.com/Wiki/Turbulence_length_scale

    if wall_res == 'wall_funcs':
        y_plus = 100
    elif wall_res == 'yplus':
        y_plus = 0.95

    X = [[0, L_ref]]
    C_mue = 0.09  # turbulence model constant

    u_inf = U_mag = atmos.ext_props['u']
    p_inf = atmos.pressure
    T_inf = atmos.temperature
    gamma_inf = atmos.ext_props['gamma']
    rho_inf = atmos.ext_props['rho']
    nue_inf = atmos.ext_props['nue']
    mue_inf = atmos.ext_props['mue']

    c_p = 1.006  # specific heat capacity [kJ/kg/K] (ca. const.)

    Re_x = rho_inf * u_inf * L_ref / mue_inf  # White. Fluid Mechanics. p. 467
    C_f = 0.026 / (Re_x ** (1 / 7))
    tau_w = C_f * rho_inf * u_inf ** 2 / 2
    u_fric = np.sqrt(tau_w / rho_inf)
    h_first_wall = y_plus * mue_inf / (u_fric * rho_inf)

    nue_tilda_inf = 3 * nue_inf  # 3*nue_inf < nue_tilda_inf < 5*nue_inf
    k_inf = 1.5 * (
                Int * U_mag) ** 2  # turbulent kinetic energy [J/kg] https://www.openfoam.com/documentation/guides/latest/doc/guide-turbulence-ras-k-omega-sst.html
    omega_inf = np.sqrt(k_inf) / (
                C_mue ** 0.25 * L_mix)  # specific turbulent dissipation rate [1/s] https://www.openfoam.com/documentation/guides/latest/doc/guide-turbulence-ras-k-omega-sst.html

    alpha_t = 0  # turbulent thermal diffusivity [m2/s]
    C_v1 = 7.1
    Xi = nue_tilda_inf / nue_inf
    f_v1 = Xi ** 3 / (Xi ** 3 + C_v1 ** 3)
    nue_t = nue_tilda_inf * f_v1  # required for Spalart-Allmaras turbulence model. turbulenct eddy viscosity https://www.cfd-online.com/Wiki/Spalart-Allmaras_model

    return k_inf, omega_inf, alpha_t, nue_t
