"""Generate a sample for a PFC sensitivity study.

Author:  A. Habermann
"""

import numpy as np
from misc_functions.helpers.find_nearest import find_nearest_replace


class Sampling:
    def __init__(self, samples_per_param: int):
        self.samples_per_param = samples_per_param

    def create_sample(self):
        x_ff = 62.3785
        r_cent_f = 6.09 / 2
        l_cent_f = 50.9
        rmax = 1.415925
        f_xmax = 0.118
        lnac = 2.53018
        h_duct = 0.56
        f_r12 = 0.51003
        f_lint = 0.3795
        teta_f_cone = 13.59
        f_rho_le = 1.0
        f_l_nose = 0.1975
        ahi_athr = 1.304
        athr_a12 = 0.994
        a18_a13 = 0.745
        f_xthr = 0.225
        delta_beta_te = 13.
        beta_te_low = 0.
        f_r18hub = 1.134
        f_rthrtip = 0.898
        teta_int_in = 10.5
        beta_ff_in = -4.
        teta_ff_in = 0.
        # 'parameter_name': [min_value, max_value, default_value]
        param_dict = {'rmax': [rmax * 0.99, rmax * 1.01, rmax],
                      # [0.465*0.995,0.465*1.005,0.465],        # own estimation
                      'x_ff': [0.99 * x_ff, 1.01 * x_ff, x_ff],
                      # [0.99*0.92822,1.01*0.92822,0.92822],     # own estimation
                      'r_cent_f': [r_cent_f * 0.995, r_cent_f * 1.005, r_cent_f],
                      # based on exisiting AC, fuselage_geometries.xlsx
                      'l_cent_f': [0.99 * l_cent_f, l_cent_f * 1.01, l_cent_f],
                      # based on exisiting AC, fuselage_geometries.xlsx
                      'f_xmax': [f_xmax * 0.99, f_xmax * 1.01, f_xmax],  # own estimation; maybe upper limit rather 0.5
                      'lnac': [lnac * 0.99, lnac * 1.01, lnac],  # [4.341*0.99,4.341*1.01,4.341],
                      'h_duct': [h_duct * 0.995, h_duct * 1.005, h_duct],
                      'f_r12': [f_r12 * 0.995, f_r12 * 1.005, f_r12],
                      'f_lint': [0.99 * f_lint, f_lint * 1.01, f_lint],
                      'teta_f_cone': [teta_f_cone - 2, teta_f_cone + 2, teta_f_cone],
                      'f_rho_le': [0.9, 1.1, f_rho_le],
                      'f_l_nose': [0.99 * f_l_nose, f_l_nose * 1.01, f_l_nose],
                      # based on exisiting AC, fuselage_geometries.xlsx
                      'ahi_athr': [0.99 * ahi_athr, ahi_athr * 1.00, ahi_athr],
                      'athr_a12': [athr_a12 * 0.99, 1.0, athr_a12],
                      'a18_a13': [0.99 * a18_a13, a18_a13 * 1.01, a18_a13],  # look at realistic engine data
                      'f_xthr': [0.99 * f_xthr, f_xthr * 1.01, f_xthr],  # look at realistic engine data
                      'delta_beta_te': [0.99 * delta_beta_te, delta_beta_te * 1.01, delta_beta_te],
                      'beta_te_low': [beta_te_low - 10, beta_te_low + 10, beta_te_low],
                      'f_r18hub': [0.99 * f_r18hub, f_r18hub * 1.01, f_r18hub],
                      'f_rthrtip': [0.99 * f_rthrtip, f_rthrtip * 1.00, f_rthrtip],
                      'teta_int_in': [teta_int_in * 0.99, teta_int_in * 1.01, teta_int_in],
                      'teta_ff_in': [teta_ff_in - 2, teta_ff_in + 2, teta_ff_in],
                      'beta_ff_in': [beta_ff_in - 2, beta_ff_in + 2, beta_ff_in],
                      'mach_number': [0.76, 0.86, 0.82],
                      'altitude': [7620., 12192., 10680.]}  # as proxy for Re number - FL [250,400,350]}

        # ATTENTION: For the sensitivity study, we don't use relative values for
        # nacelle length (f_lnac -> lnac),
        # max. nacelle radius (f_rmax -> rmax),
        # ff position (f_x_ff -> x_ff)

        samples_dict = {}

        for key, value in param_dict.items():
            param_sample = generate_sample(value[0], value[1], value[2], self.samples_per_param)
            samples_dict[key] = param_sample

        individual_sims = []

        # default case
        individual_sims.append({key: value[2] for key, value in param_dict.items()})
        individual_sims[0]['omega_rot'] = 293.

        # add all sensitivity study cases
        for key, value in samples_dict.items():
            for i in value:
                if i != param_dict[key][2]:
                    individual_sims.append({key: value[2] for key, value in param_dict.items()})
                    individual_sims[-1][key] = i
                individual_sims[-1]['omega_rot'] = 293.

        param_sample_plot = dict(samples_dict)
        param_sample_for_plotting = dict(samples_dict)
        for key, value in samples_dict.items():
            if isinstance(value, np.ndarray):
                unique_value = np.unique(value)
                param_sample_plot[key] = unique_value
                param_sample_for_plotting[key] = unique_value
                if key == "ahi_athr" or key == "f_rthrtip":
                    param_sample_plot[key] = np.delete(param_sample_plot[key], 4)
                elif key == "altitude":
                    param_sample_plot[key] = np.delete(param_sample_plot[key], 3)
                else:
                    param_sample_plot[key] = np.delete(param_sample_plot[key], 2)

        return individual_sims, param_sample_plot, param_sample_for_plotting


def generate_sample(min, max, nominal, number_of_samples):
    orig_sample = np.linspace(min, max, number_of_samples)
    target_sample = find_nearest_replace(orig_sample, [nominal])
    return target_sample
