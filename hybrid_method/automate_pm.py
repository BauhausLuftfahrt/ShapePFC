"""Automate panel method solution for PFC study with hybrid method.

Author:  A. Habermann
"""

import csv
from panel.solve_potential_flow_pfc_hybrid import PotentialFlow
from panel.solve_panel_method_pfc_hybrid import BoundaryLayerCalculation


def process_pfchybrid_ibl(Xn_fuse, Yn_fuse, Fm_fuse, arc_length_fuse, atmos, flags, eps, eps_2, pm_max_it,
                          save_results=True, resultpath=None):
    counter = 0  # Counts number of interactions between viscid and inviscid flow
    flags[7] = 0
    fuselage_panel_pot_init = PotentialFlow([Xn_fuse], [Yn_fuse], [Fm_fuse], atmos, flags, 0)
    pot_init, surface_init, sigma_init, j_s_init, j_v_init = fuselage_panel_pot_init.calculate_potential_flow()

    # calculate potential flow with IBL interaction
    flags[7] = 1
    fuselage_panel_ibl = BoundaryLayerCalculation([Xn_fuse], [Yn_fuse], [Fm_fuse], [arc_length_fuse], atmos,
                                                  flags, counter,  # eps=eps, eps2=eps_2)#
                                                  pot_init, surface_init, sigma_init, j_s_init, j_v_init, eps,
                                                  eps_2, pm_max_it)
    pot_final, surface_final, sigma_final, j_s_final, j_v_final, bl_final, p_s, C_f, x_tr_rel, _, pm_conv = \
        fuselage_panel_ibl.calculateIBL()

    if save_results is True:
        """SAVE PANEL METHOD RESULTS"""
        geokeys = ['panel_X_node_fuse', 'panel_r_node_fuse', 'panel_length_fuse', 'panel_angle_fuse',
                   'panel_start_end_x_fuse', 'panel_start_end_y_fuse']
        geo_data_rows = list(
            zip(surface_init[0][0], surface_init[0][1], surface_init[0][2], surface_init[0][3], surface_init[0][4],
                surface_init[0][5]))

        potkeys = ['Vx_e / u_inf', 'Vy_e / u_inf', 'u_e / u_inf', 'p_e', 'rho_e', 'M_e', 'Cp_e']
        pot_data_rows = list(zip(pot_final[0], pot_final[1], pot_final[2], pot_final[3], pot_final[4],
                                 pot_final[5], pot_final[6]))

        iblkeys = ['delta_star_phys', 'delta', 'delta_star_phys_BC', 'ue_BC', 'Theta', 'H', 'theta', 'Delta_star',
                   'n',
                   'Cf', 'p_s']
        ibl_data_rows = list(
            zip(bl_final[0][0], bl_final[1][0], bl_final[2][0], bl_final[3][0], bl_final[4], bl_final[5],
                bl_final[6], bl_final[7], bl_final[8], bl_final[9], bl_final[10]))

        with open(f"{resultpath}//panel_geo.csv", 'w', newline="") as geofile:
            writergeo = csv.writer(geofile)
            writergeo.writerow(geokeys)
            writergeo.writerows(geo_data_rows)

        with open(f"{resultpath}//pot_sol.csv", 'w', newline="") as potfile:
            writerpan = csv.writer(potfile)
            writerpan.writerow(potkeys)
            writerpan.writerows(pot_data_rows)

        with open(f"{resultpath}//ibl_sol.csv", 'w', newline="") as iblfile:
            writeribl = csv.writer(iblfile)
            writeribl.writerow(iblkeys)
            writeribl.writerows(ibl_data_rows)

    return pm_conv, surface_init, surface_final, j_s_final, sigma_final, pot_final, bl_final
