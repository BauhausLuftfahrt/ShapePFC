"""Generate a design of experiment for a PFC multi-parameter study using Maximin Latin Hypercube Sampling.

Author:  A. Habermann
"""

from pyDOE2 import doe_lhs
import matplotlib.pyplot as plt
from geometry_generation.finite_volume_geometry.generate_fv_icst_geometry import GenerateGeomICST
import csv
from doegen.doegen import eval_extarray, read_setup
import numpy as np
from misc_functions.air_properties.create_atmos import create_atmos
from finite_volume.open_foam.prepare_body_force_model.prepare_centreline_fan_stage.scale_rot_speed import \
    scale_rotor_rotational_speed


def multidimensional_study(task):
    if task == 'create_lhs':

        # 'parameter_name': [min_value, max_value, default_value]
        param_dict = {'f_x_ff': [0.85, 0.95, 0.9283],  # own estimation
                      'r_cent_f': [1.75, 4.0, 3.045],  # based on exisiting AC, fuselage_geometries.xlsx
                      'l_cent_f': [20, 65, 50.9],  # based on exisiting AC, fuselage_geometries.xlsx
                      'f_rmax': [1., 1.3, 1.052814],
                      'f_xmax': [0.1, 0.5, 0.118],
                      # lower limit outcome of invalid geometries; own estimation; maybe upper limit rather 0.5
                      'f_lnac': [10, 2, 4.341],
                      'h_duct': [0.2, 0.8, 0.56],
                      'f_r12': [0.3, 0.7, 0.51],
                      'f_lint': [0.1, 0.5, 0.3795],
                      'teta_f_cone': [5, 20, 13.57],
                      'f_rho_le': [0.5, 1.5, 1.0],
                      'f_l_nose': [0.1, 0.4, 0.1975],  # based on exisiting AC, fuselage_geometries.xlsx
                      'ahi_athr': [1.0, 1.5, 1.304],
                      'athr_a12': [0.95, 1.0, 0.994],
                      'a18_a13': [0.6, 1.0, 0.78],  # look at realistic engine data
                      'f_xthr': [0.02, 0.4, 0.243],  # look at realistic engine data
                      'delta_beta_te': [0, 20, 13],
                      'beta_te_low': [-10, 20, 0],
                      'f_r18hub': [1.0, 1.5, 1.134],
                      'f_rthrtip': [0.8, 0.975, 0.898],  # upper limit outcome of invalid geometries;
                      'teta_int_in': [0.0, 20.0, 10.5],
                      'teta_ff_in': [-10, 4, -4],
                      'beta_ff_in': [-10, 4, -4],
                      'mach_number': [0.76, 0.86, 0.82],
                      'altitude': [7620, 12192, 10680]}  # as proxy for Re number - FL [250,400,350]}

        # rule of thumb, e.g. Loepky et al. 2009 (10 times the number of parameters -> here, 250 would be required)
        # actually, 300 samples are required, because 15% of geometries don't run through IBL PM simulation,
        # because there are shocks near the fuselage nose
        number_of_samples = round(73.2 * len(param_dict))

        Y = doe_lhs.lhs(len(param_dict), criterion='maximin', samples=number_of_samples)

        LHS_matrix = []
        individual_sims = []

        for i in range(0, number_of_samples):
            row = []
            for j in range(0, len(param_dict)):
                row.append(Y[i][j] * (list(param_dict.values())[j][1] - list(param_dict.values())[j][0]) +
                           list(param_dict.values())[j][0])
            LHS_matrix.append(row)

        samples_dict = {}

        i = 0
        for key, value in param_dict.items():
            param_sample = [LHS_matrix[j][i] for j in range(0, len(LHS_matrix))]
            samples_dict[key] = param_sample
            i += 1

        for j in range(0, len(LHS_matrix)):
            individual = {}
            for key, value in samples_dict.items():
                individual.update({key: samples_dict[key][j]})
            individual_sims.append(individual)

        lhs_header = ['Nexp']
        lhs_header.extend(list(param_dict.keys()))
        lhs_for_eval = np.hstack((np.array([np.arange(0, number_of_samples)]).T, Y))

        with open('doe_generation_data/LHS_orig.csv', 'w', newline='') as output_file:
            dict_writer = csv.writer(output_file)
            dict_writer.writerow(lhs_header)
            dict_writer.writerows(lhs_for_eval)

        with open('doe_generation_data/LHS_orig_eval.csv', 'w', newline='') as output_file:
            dict_writer = csv.writer(output_file)
            dict_writer.writerows(lhs_for_eval)

        setup = read_setup('Experiment_setup.xlsx')
        eval_orig_lhs = eval_extarray(setup, '', 'LHS_orig_eval.csv')

    elif task == 'read_lhs':
        individual_sims = []
        with open('doe_generation_data/LHS_orig.csv', 'r', newline='') as input_file:
            dict_reader = csv.DictReader(input_file)
            for row in dict_reader:
                float_dict = {key: float(value) for key, value in row.items()}
                individual_sims.append(float_dict)

    invalid = 0
    valid_samples = []
    invalid_samples = []
    if task == 'create_lhs':
        valid_lhs = []

    for i in range(0, len(individual_sims)):
        try:
            Geometry = GenerateGeomICST(f_x_ff=individual_sims[i]['f_x_ff'], r_cent_f=individual_sims[i]['r_cent_f'],
                                        l_cent_f=individual_sims[i]['l_cent_f'], f_rmax=individual_sims[i]['f_rmax'],
                                        f_xmax=individual_sims[i]['f_xmax'], f_lnac=individual_sims[i]['f_lnac'],
                                        h_duct=individual_sims[i]['h_duct'], f_r12=individual_sims[i]['f_r12'],
                                        f_lint=individual_sims[i]['f_lint'],
                                        teta_f_cone=individual_sims[i]['teta_f_cone'],
                                        f_rho_le=individual_sims[i]['f_rho_le'],
                                        f_l_nose=individual_sims[i]['f_l_nose'],
                                        ahi_athr=individual_sims[i]['ahi_athr'],
                                        athr_a12=individual_sims[i]['athr_a12'],
                                        a18_a13=individual_sims[i]['a18_a13'],
                                        f_xthr=individual_sims[i]['f_xthr'],
                                        delta_beta_te=individual_sims[i]['delta_beta_te'],
                                        beta_te_low=individual_sims[i]['beta_te_low'],
                                        r_te_hub=0.0, f_r18hub=individual_sims[i]['f_r18hub'],
                                        f_rthrtip=individual_sims[i]['f_rthrtip'],
                                        teta_int_in=individual_sims[i]['teta_int_in'],
                                        teta_ff_in=individual_sims[i]['teta_ff_in'],
                                        beta_ff_in=individual_sims[i]['beta_ff_in'], plot=False, samplevars=False)

            fuselage, nacelle_top, nacelle_bottom, rotor_inlet, rotor_outlet, stator_inlet, stator_outlet, l_fuse, f_slr, tc_max, \
            tc_max_x, c_nac, i_nac, teta_f_aft, Astar_A2, _, x_12, x_13, rotor_le_coeffs, stator_le_coeffs, \
            h_duct_in, h_duct_out, A_12, A_13, A_18, r_12_tip, x_nac_max_tot, ar_nose, fuselage_panel \
                = Geometry.build_geometry()

            plt.close()
            plt.title(i)
            plt.plot([i[0] for i in fuselage], [i[1] for i in fuselage])
            plt.plot([i[0] for i in nacelle_top], [i[1] for i in nacelle_top])
            plt.plot([i[0] for i in nacelle_bottom], [i[1] for i in nacelle_bottom])
            plt.plot([i[0] for i in rotor_inlet], [i[1] for i in rotor_inlet])
            plt.plot([i[0] for i in rotor_outlet], [i[1] for i in rotor_outlet])
            plt.plot([i[0] for i in stator_inlet], [i[1] for i in stator_inlet])
            plt.plot([i[0] for i in stator_outlet], [i[1] for i in stator_outlet])
            plt.xlim(individual_sims[i]['l_cent_f'])
            # plt.xlim(50,80)
            # plt.legend()
            plt.savefig(f'./multidimstudy/{i}')
            # plt.show()

            # calculate omega_rot
            atmos = create_atmos(individual_sims[i]['altitude'], individual_sims[i]['mach_number'], l_fuse, 0)
            individual_sims[i]['omega_rot'] = scale_rotor_rotational_speed(2 * r_12_tip, atmos.ext_props['gamma'],
                                                                           atmos.ext_props['R'], atmos.ext_props['T_t'])

            valid_samples.append(individual_sims[i])

            if task == 'create_lhs':
                valid_lhs.append(Y[i])

        except Exception as a:
            # print(f'No solution for index {i}. {individual_sims[i]}')
            # print(a)

            invalid_samples.append([i, a])
            invalid += 1

    if task == 'create_lhs':
        lhs_new_for_eval = np.hstack((np.array([np.arange(0, len(valid_lhs))]).T, valid_lhs))

        samplekeys = ['index', 'f_x_ff', 'r_cent_f', 'l_cent_f', 'f_rmax', 'f_xmax', 'f_lnac', 'h_duct', 'f_r12',
                      'f_lint',
                      'teta_f_cone', 'f_rho_le', 'f_l_nose', 'ahi_athr', 'athr_a12', 'a18_a13', 'f_xthr',
                      'delta_beta_te',
                      'beta_te_low', 'f_r18hub', 'f_rthrtip', 'teta_int_in', 'teta_ff_in', 'beta_ff_in', 'mach_number',
                      'altitude', 'omega_rot']

        with open('doe_generation_data/LHS_new.csv', 'w', newline='') as output_file:
            dict_writer = csv.writer(output_file)
            dict_writer.writerow(samplekeys)
            dict_writer.writerows(lhs_new_for_eval)

        with open('doe_generation_data/LHS_new_eval.csv', 'w', newline='') as output_file:
            dict_writer = csv.writer(output_file)
            dict_writer.writerows(lhs_new_for_eval)

        with open('doe_generation_data/individual_samples.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=samplekeys[1:])
            writer.writeheader()
            for row in valid_samples:
                writer.writerow(row)

        eval_new_lhs = eval_extarray(setup, '', 'LHS_new_eval.csv')

    plt.close()
    plt.scatter([i['f_rmax'] for i in individual_sims], [i['f_xmax'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['f_rmax'] for i in valid_samples], [i['f_xmax'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/f_rmax__f_x_max')

    plt.close()
    plt.scatter([i['r_cent_f'] for i in individual_sims], [i['l_cent_f'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['r_cent_f'] for i in valid_samples], [i['l_cent_f'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/r_cent_f__l_cent_f')

    plt.close()
    plt.scatter([i['f_rmax'] for i in individual_sims], [i['f_xmax'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['f_rmax'] for i in valid_samples], [i['f_xmax'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/f_rmax__f_x_max')

    plt.close()
    plt.scatter([i['f_lnac'] for i in individual_sims], [i['h_duct'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['f_lnac'] for i in valid_samples], [i['h_duct'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/f_lnac__h_duct')

    plt.close()
    plt.scatter([i['f_r12'] for i in individual_sims], [i['f_lint'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['f_r12'] for i in valid_samples], [i['f_lint'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/f_r12__f_lint')

    plt.close()
    plt.scatter([i['teta_f_cone'] for i in individual_sims], [i['f_x_ff'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['teta_f_cone'] for i in valid_samples], [i['f_x_ff'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/teta_f_cone__f_x_ff')

    plt.close()
    plt.scatter([i['f_rho_le'] for i in individual_sims], [i['ahi_athr'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['f_rho_le'] for i in valid_samples], [i['ahi_athr'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/f_rho_le__ahi_athr')

    plt.close()
    plt.scatter([i['f_l_nose'] for i in individual_sims], [i['l_cent_f'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['f_l_nose'] for i in valid_samples], [i['l_cent_f'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/f_l_nose__l_cent_f')

    plt.close()
    plt.scatter([i['ahi_athr'] for i in individual_sims], [i['athr_a12'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['ahi_athr'] for i in valid_samples], [i['athr_a12'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/ahi_athr__athr_a12')

    plt.close()
    plt.scatter([i['a18_a13'] for i in individual_sims], [i['f_r18hub'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['a18_a13'] for i in valid_samples], [i['f_r18hub'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/a18_a13__f_r18hub')

    plt.close()
    plt.scatter([i['f_xthr'] for i in individual_sims], [i['f_rthrtip'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['f_xthr'] for i in valid_samples], [i['f_rthrtip'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/f_xthr__f_rthrtip')

    plt.close()
    plt.scatter([i['delta_beta_te'] for i in individual_sims], [i['beta_te_low'] for i in individual_sims], color='0.8',
                s=2)
    plt.scatter([i['delta_beta_te'] for i in valid_samples], [i['beta_te_low'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/delta_beta_te__beta_te_low')

    plt.close()
    plt.scatter([i['f_rthrtip'] for i in individual_sims], [i['f_r18hub'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['f_rthrtip'] for i in valid_samples], [i['f_r18hub'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/f_rthrtip__f_r18hub')

    plt.close()
    plt.scatter([i['teta_int_in'] for i in individual_sims], [i['f_x_ff'] for i in individual_sims], color='0.8', s=2)
    plt.scatter([i['teta_int_in'] for i in valid_samples], [i['f_x_ff'] for i in valid_samples], color='r', s=3)
    plt.savefig(f'./multidimstudy/teta_int_in__f_x_ff')

    print(f'{len(individual_sims) - invalid} out of {len(individual_sims)} samples valid '
          f'({(len(individual_sims) - invalid) / len(individual_sims) * 100} %).')


if __name__ == "__main__":
    multidimensional_study('create_lhs')
