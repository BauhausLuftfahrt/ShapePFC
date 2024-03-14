"""Calculate flow characteristics (velocity and temperature profiles) at the PM/FVM interface using the results of the PM.

Author:  A. Habermann
"""

import numpy as np
from scipy import interpolate
from panel.potential_flow.compressibility_correction import karman_tsien_compr_corr
from post_processing.panel.boundary_layer_profiles import calc_velocity_profile, calc_temperature_profile
from post_processing.panel.velocity_potential_interface import calculate_vel_pot_line


class Interface:

    def __init__(self, ibl_sol, surface, j_s, sigma, pot_sol, x_panel_simplified, y_panel_simplified, atmos, l_cent_f,
                 hdomain):
        self.atmos = atmos
        self.ibl = ibl_sol
        self.pot = pot_sol
        self.surface = surface
        self.j_s = j_s
        self.sigma = sigma
        self.vel_x_int = None
        self.vel_y_int = None
        self.x_int = None
        self.y_int = None
        self.y = None
        self.idx_int = None
        self.X_fuselage = x_panel_simplified
        self.Y_fuselage = y_panel_simplified
        self.l_cent_f = l_cent_f
        self.hdom = hdomain

    def interface_location(self, interface_loc):
        self.x_int = interface_loc * self.l_cent_f
        self.y_int = max(self.Y_fuselage)
        self.idx_int = np.where(self.X_fuselage < self.x_int)[0][-1]

    def profiles(self, path):
        y_blnondim = np.logspace(0, 1, 1000)
        delta_int = interpolate_bl(self.x_int, self.X_fuselage, self.idx_int, self.ibl[1][0])
        delta_starPhys_int = interpolate_bl(self.x_int, self.X_fuselage, self.idx_int, self.ibl[0][0])
        u_e_int = self.atmos.ext_props['u'] * interpolate_bl(self.x_int, self.X_fuselage, self.idx_int, self.pot[2])
        M_e_int = interpolate_bl(self.x_int, self.X_fuselage, self.idx_int, self.pot[5])
        n_int = interpolate_bl(self.x_int, self.X_fuselage, self.idx_int, self.ibl[8])
        C_f_int = interpolate_bl(self.x_int, self.X_fuselage, self.idx_int, self.ibl[9])
        dp_e_dx_int = (self.pot[3][self.idx_int + 1] - self.pot[3][self.idx_int]) / (
                self.X_fuselage[self.idx_int + 1] - self.X_fuselage[self.idx_int])
        tau_w = (C_f_int * self.atmos.ext_props['rho'] * (u_e_int) ** 2) / 2

        # assumption: freestream temperature at BL edge
        T_e_int = self.atmos.temperature
        # identify edge of boundary layer (physical boundary)
        y_ble = self.y_int + delta_int

        y_bl = (y_blnondim - y_blnondim[0]) / (y_blnondim[-1] - y_blnondim[0]) * (y_ble - self.y_int) + self.y_int

        # calculate velocity and temperature profile at interface
        u_prof, _, _ = calc_velocity_profile(u_e_int, delta_int, self.y_int, C_f_int, self.atmos.ext_props['nue'], y_bl,
                                             'granville_spalding', n_int, delta_starPhys_int, tau_w, dp_e_dx_int)

        t_prof = calc_temperature_profile(u_prof, u_e_int, M_e_int, T_e_int)

        # extend velocity and temperature profiles to edge of FV domain
        y_ext = np.linspace(y_bl[-1], self.hdom, 1000)[2:]

        # calculate velocity potential derivatives at interface from panel method solution in potential flow region
        pot_vz, pot_vr = calculate_vel_pot_line(self.surface[0], self.j_s, self.sigma, self.X_fuselage,
                                                self.Y_fuselage, self.atmos.ext_props['u'], [self.x_int] *
                                                len(y_ext), y_ext)

        pot_v = [np.sqrt(pot_vz[i] ** 2 + pot_vr[i] ** 2) for i in range(0, len(pot_vz))]

        # Karman-Tsien compressibility correction
        pot_vz_c, pot_vr_c = karman_tsien_compr_corr(self.atmos, pot_v, pot_vz, pot_vr)

        pot_vz = pot_vz_c
        pot_vr = pot_vr_c

        # ensure that "below" surface temperature and velocity values are not extrapolated to crazy values
        y_pre = np.linspace(y_bl[0] - 1, y_bl[0], 11)
        y_interface = np.concatenate((y_pre[:-1], y_bl, y_ext))

        u_pre = [u_prof[0]] * 10
        ux_interface = u_pre + u_prof + list(pot_vz)
        uz_interface = [0] * (len(u_prof) + 10) + list(pot_vr)

        t_interface = [t_prof[0][0]] * 10 + [i[0] for i in t_prof] + [self.atmos.temperature[0] for i in
                                                                      range(0, len(y_ext))]

        with open(f'{path}//u_interface_data.csv', 'w') as f:
            f.write("\"y\",\"u_x\",\"u_z\"\n")
            for i in range(0, len(y_interface)):
                f.write(f'{y_interface[i]}, {ux_interface[i]}, {uz_interface[i]}\n')

        with open(f'{path}//t_interface_data.csv', 'w') as f:
            f.write("\"y\",\"t\"\n")
            for i in range(0, len(y_interface)):
                f.write(f'{y_interface[i]}, {t_interface[i]}\n')

        F_bl_ux = interpolate.UnivariateSpline(y_interface, ux_interface, s=0)
        F_bl_uz = interpolate.UnivariateSpline(y_interface, uz_interface, s=0)
        F_bl_t = interpolate.UnivariateSpline(y_interface, t_interface, s=0)

        return F_bl_ux, F_bl_uz, F_bl_t


def interpolate_bl(x_point, x_curve, idx_curve, bl_charac):
    return (x_point - x_curve[idx_curve]) / (x_curve[idx_curve + 1] - x_curve[idx_curve]) * \
           (bl_charac[idx_curve + 1] - bl_charac[idx_curve]) + bl_charac[idx_curve]
