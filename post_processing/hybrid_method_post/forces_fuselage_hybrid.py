"""Calculate the pressure and skin friction forces of fuselage and nacelle of a PFC geometry from panel method and
finite volume method skin friction coefficient and pressure coefficient results.

Author:  A. Habermann
"""

from scipy import interpolate

import csv
import numpy as np
import itertools
from post_processing.panel.plot.drag_computation import dragBody


def int_forces_smooth(casepath, atmos, surface, pot, ibl, int, geo, type='full'):
    if type == 'hybrid_method_post':
        fuse_parts = ['fuse_center', 'fuse_hub_gap', 'fuse_hub_rotor', 'fuse_hub_stator', 'fuse_hub_inlet',
                      'fuse_hub_nozzle', 'fuse_sweep', 'fuse_tail']
        fuse_parts_rest = ['fuse_center', 'fuse_sweep', 'fuse_tail']
    else:
        fuse_parts = ['fuse_center', 'fuse_hub_gap', 'fuse_hub_rotor', 'fuse_hub_stator', 'fuse_hub_inlet',
                      'fuse_hub_nozzle', 'fuse_nose', 'fuse_sweep', 'fuse_tail']
        fuse_parts_rest = ['fuse_center', 'fuse_nose', 'fuse_sweep', 'fuse_tail']
    nac_parts_top = ['nac_cowling']
    nac_parts_bottom = ['nac_inlet', 'nac_nozzle', 'nac_rotor', 'nac_gap', 'nac_stator']
    fuse_forces = {}
    fuse_forces['total'] = []
    nac_forces = {}
    nac_forces['total'] = []
    nac_forces['top'] = []
    nac_forces['bottom'] = []

    c_p_fus_fvm = []
    c_f_fus_fvm = []
    x_fus_fvm = []
    z_fus_fvm = []
    for i in fuse_parts:
        c_p_part = []
        c_f_part = []
        x_part = []
        z_part = []
        with open(f'{casepath}/{i}_data.csv', mode='r') as file:
            temp = csv.DictReader(file)
            for line in temp:
                c_p_part.append(float(line['C_p']))
                c_f_part.append(float(line['C_f']))
                x_part.append(float(line['Points:0']))
                z_part.append(float(line['Points:2']))

        c_p_fus_fvm.append(c_p_part)
        c_f_fus_fvm.append(c_f_part)
        x_fus_fvm.append(x_part)
        z_fus_fvm.append(z_part)

    c_p_fus_fvm = list(itertools.chain(*c_p_fus_fvm))
    c_f_fus_fvm = list(itertools.chain(*c_f_fus_fvm))
    x_fus_fvm = list(itertools.chain(*x_fus_fvm))
    z_fus_fvm = list(itertools.chain(*z_fus_fvm))
    x_fus_fvm, z_fus_fvm, c_p_fus_fvm, c_f_fus_fvm = sort_lists_exclude_duplicates(x_fus_fvm, z_fus_fvm, c_p_fus_fvm,
                                                                                   c_f_fus_fvm)
    alpha_fus_fvm, dA_fus_fvm = calc_inc_areas(x_fus_fvm, z_fus_fvm)

    c_p_fus_fvm_rest = []
    c_f_fus_fvm_rest = []
    x_fus_fvm_rest = []
    z_fus_fvm_rest = []
    for i in fuse_parts_rest:
        c_p_part = []
        c_f_part = []
        x_part = []
        z_part = []
        with open(f'{casepath}/{i}_data.csv', mode='r') as file:
            temp = csv.DictReader(file)
            for line in temp:
                c_p_part.append(float(line['C_p']))
                c_f_part.append(float(line['C_f']))
                x_part.append(float(line['Points:0']))
                z_part.append(float(line['Points:2']))

        c_p_fus_fvm_rest.append(c_p_part)
        c_f_fus_fvm_rest.append(c_f_part)
        x_fus_fvm_rest.append(x_part)
        z_fus_fvm_rest.append(z_part)

    c_p_fus_fvm_rest = list(itertools.chain(*c_p_fus_fvm_rest))
    c_f_fus_fvm_rest = list(itertools.chain(*c_f_fus_fvm_rest))
    x_fus_fvm_rest = list(itertools.chain(*x_fus_fvm_rest))
    z_fus_fvm_rest = list(itertools.chain(*z_fus_fvm_rest))
    x_fus_fvm_rest, z_fus_fvm_rest, c_p_fus_fvm_rest, c_f_fus_fvm_rest = sort_lists_exclude_duplicates(x_fus_fvm_rest,
                                                                                                       z_fus_fvm_rest,
                                                                                                       c_p_fus_fvm_rest,
                                                                                                       c_f_fus_fvm_rest)
    alpha_fus_fvm_rest, dA_fus_fvm_rest = calc_inc_areas(x_fus_fvm_rest, z_fus_fvm_rest)

    c_p_nac_top = []
    c_f_nac_top = []
    x_nac_top = []
    z_nac_top = []
    for i in nac_parts_top:
        c_p_part = []
        c_f_part = []
        x_part = []
        z_part = []
        with open(f'{casepath}/{i}_data.csv', mode='r') as file:
            temp = csv.DictReader(file)
            for line in temp:
                c_p_part.append(float(line['C_p']))
                c_f_part.append(float(line['C_f']))
                x_part.append(float(line['Points:0']))
                z_part.append(float(line['Points:2']))

        c_p_nac_top.append(c_p_part)
        c_f_nac_top.append(c_f_part)
        x_nac_top.append(x_part)
        z_nac_top.append(z_part)

    c_p_nac_top = list(itertools.chain(*c_p_nac_top))
    c_f_nac_top = list(itertools.chain(*c_f_nac_top))
    x_nac_top = list(itertools.chain(*x_nac_top))
    z_nac_top = list(itertools.chain(*z_nac_top))
    x_nac_top, z_nac_top, c_p_nac_top, c_f_nac_top = sort_lists_exclude_duplicates(x_nac_top, z_nac_top, c_p_nac_top,
                                                                                   c_f_nac_top)
    alpha_nac_top, dA_nac_top = calc_inc_areas(x_nac_top, z_nac_top)

    c_p_nac_bottom = []
    c_f_nac_bottom = []
    x_nac_bottom = []
    z_nac_bottom = []
    for i in nac_parts_bottom:
        c_p_part = []
        c_f_part = []
        x_part = []
        z_part = []
        with open(f'{casepath}/{i}_data.csv', mode='r') as file:
            temp = csv.DictReader(file)
            for line in temp:
                c_p_part.append(float(line['C_p']))
                c_f_part.append(float(line['C_f']))
                x_part.append(float(line['Points:0']))
                z_part.append(float(line['Points:2']))

        c_p_nac_bottom.append(c_p_part)
        c_f_nac_bottom.append(c_f_part)
        x_nac_bottom.append(x_part)
        z_nac_bottom.append(z_part)

    c_p_nac_bottom = list(itertools.chain(*c_p_nac_bottom))
    c_f_nac_bottom = list(itertools.chain(*c_f_nac_bottom))
    x_nac_bottom = list(itertools.chain(*x_nac_bottom))
    z_nac_bottom = list(itertools.chain(*z_nac_bottom))
    x_nac_bottom, z_nac_bottom, c_p_nac_bottom, c_f_nac_bottom = sort_lists_exclude_duplicates(x_nac_bottom,
                                                                                               z_nac_bottom,
                                                                                               c_p_nac_bottom,
                                                                                               c_f_nac_bottom)
    alpha_nac_bottom, dA_nac_bottom = calc_inc_areas(x_nac_bottom, z_nac_bottom)

    alpha_nac_bottom = [-i for i in alpha_nac_bottom]
    # calculate nacelle forces from FVM results
    ff_nactop, fp_nactop = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_nac_top,
                                    dA_nac_top,
                                    C_p=c_p_nac_top, p_s=None, C_f=c_f_nac_top, tau=None)

    ff_nacbot, fp_nacbot = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_nac_bottom,
                                    dA_nac_bottom,
                                    C_p=c_p_nac_bottom, p_s=None, C_f=c_f_nac_bottom, tau=None)

    # calculate fuselage forces from c_p and c_f distribution of PM and FVM
    # assumptions:
    # 1) front of fuselage: assumption "thin boundary layer", i.e. use of edge pressure
    # 2) rear fuselage (delta > 0.1*r_cent,f): assumption "thick boundary layer", i.e. use of wall pressure calculated
    # from Patel's method
    # 3) weird behavior of skin friction in FV domain right behin interface. -> "smoothen" c_f distribution, i.e.
    # use panel method result until end of center section

    cf_w_pm = ibl[9]  # skin friction coefficient at wall as calculated with PM
    cp_e_pm = pot[6]  # pressure coefficient at boundary layer edge calc. with PM
    cp_w_pm = [(i - atmos.pressure[0]) / (0.5 * atmos.ext_props['rho'][0] * atmos.ext_props['u'][0] ** 2) for i in
               ibl[10]]  # pressure coefficient at wall calc. with PM
    Xs = surface[0][0]
    Xn = surface[0][4]

    delta = ibl[1]  # boundary layer thickness

    # Compute pressure at stagnation point with Karman-Tsien compressibility correction
    Cp_i_stag = 1  # Bernoulli incompressible
    Cp_stag = Cp_i_stag / ((1 - atmos.ext_props['mach'] ** 2) ** 0.5 + (atmos.ext_props['mach'] ** 2) *
                           (Cp_i_stag / 2) / (1 + (1 - atmos.ext_props['mach'] ** 2) ** 0.5))

    # compute location at which thin BL turns into thick BL
    r_max = max(z_fus_fvm)
    idx_thick = [index for index, value in enumerate(delta[0]) if value > 0.1 * r_max][0]

    # combine edge and wall pressure of PM for thin and thick BL region and add stagnation pressure at stagnation point
    cp_pm_comb = list(cp_e_pm[:idx_thick]) + cp_w_pm[idx_thick:]

    if Cp_stag > cp_pm_comb[0]:
        cp_pm_comb = [Cp_stag] + cp_pm_comb
    else:
        F_cp_pm_pre = interpolate.interp1d(Xs, cp_pm_comb, fill_value='extrapolate')
        cp_pm_comb = [F_cp_pm_pre(0).tolist()] + cp_pm_comb

    Xs = [0] + list(Xs)
    cf_pm_comb = [0] + list(cf_w_pm)

    # use PM cp results until interface at 0.8*l_cent,f
    Xn_cp_pm = np.linspace(min(Xn), int.x_int, 3000)
    Xs_cp_pm = np.array([Xn_cp_pm[i] + 0.5 * (Xn_cp_pm[i + 1] - Xn_cp_pm[i]) for i in range(0, len(Xn_cp_pm) - 1)])

    F_cp_pm = interpolate.UnivariateSpline(Xs, cp_pm_comb, s=0)
    F_cf_pm = interpolate.UnivariateSpline(Xs, cf_pm_comb, s=0)

    # use PM cf results until 0.9*l_cent,f
    Xn_cf_pm = np.concatenate((Xn_cp_pm[:-1], np.linspace(int.x_int, 0.9 * geo['l_cent_f'], 600)))
    Xs_cf_pm = np.array([Xn_cf_pm[i] + 0.5 * (Xn_cf_pm[i + 1] - Xn_cf_pm[i]) for i in range(0, len(Xn_cf_pm) - 1)])

    # pressure and skin friction distributions to be used for fuselage force calculations
    cf_pm = F_cf_pm(Xs_cf_pm)
    cp_pm = F_cp_pm(Xs_cp_pm)

    # add FVM results for fuselage
    F_cp_fvm = interpolate.UnivariateSpline(x_fus_fvm, c_p_fus_fvm, s=0)
    F_cf_fvm = interpolate.UnivariateSpline(x_fus_fvm, c_f_fus_fvm, s=0)

    Xn_cf_fvm = np.linspace(0.9 * geo['l_cent_f'], max(x_fus_fvm), 1500)
    Xn_cp_fvm = np.linspace(int.x_int, max(x_fus_fvm), 2100 - 1)

    Xn_cf_fvm_cent = np.linspace(0.9 * geo['l_cent_f'], geo['l_cent_f'], 200)
    Xn_cf_fvm_rear = np.linspace(geo['l_cent_f'], max(x_fus_fvm), 1300)

    Xs_cf_fvm = np.array([Xn_cf_fvm[i] + 0.5 * (Xn_cf_fvm[i + 1] - Xn_cf_fvm[i]) for i in range(0, len(Xn_cf_fvm) - 1)])
    Xs_cp_fvm = np.array([Xn_cp_fvm[i] + 0.5 * (Xn_cp_fvm[i + 1] - Xn_cp_fvm[i]) for i in range(0, len(Xn_cp_fvm) - 1)])

    Xs_cf_fvm_cent = np.array([Xn_cf_fvm_cent[i] + 0.5 * (Xn_cf_fvm_cent[i + 1] - Xn_cf_fvm_cent[i]) for i in
                               range(0, len(Xn_cf_fvm_cent) - 1)])
    Xs_cf_fvm_rear = np.array([Xn_cf_fvm_rear[i] + 0.5 * (Xn_cf_fvm_rear[i + 1] - Xn_cf_fvm_rear[i]) for i in
                               range(0, len(Xn_cf_fvm_rear) - 1)])

    cf_fvm = F_cf_fvm(Xs_cf_fvm)
    cp_fvm = F_cp_fvm(Xs_cp_fvm)

    cf = list(cf_pm) + list(cf_fvm)
    cp = list(cp_pm) + list(cp_fvm)

    F_cf = interpolate.UnivariateSpline(np.concatenate((Xs_cf_pm, Xs_cf_fvm)), cf, s=0)
    F_cp = interpolate.UnivariateSpline(np.concatenate((Xs_cp_pm, Xs_cp_fvm)), cp, s=0)

    # calculate continuous body contour angle and incremental surface areas
    F_z_contour = interpolate.UnivariateSpline([i[0] for i in geo['fuselage']], [i[1] for i in geo['fuselage']], s=0)
    Xs_tot = np.concatenate((Xs_cf_pm, Xs_cf_fvm))
    Xn_tot = np.concatenate((Xn_cf_pm[:-1], Xn_cf_fvm))
    Zn_tot = F_z_contour(Xn_tot)
    Zn_ibl = F_z_contour(Xn_cf_pm)
    Zn_fvm = F_z_contour(Xn_cf_fvm)
    Zn_fvm_cent = F_z_contour(Xn_cf_fvm_cent)
    Zn_fvm_rear = F_z_contour(Xn_cf_fvm_rear)

    alpha = [np.arctan((Zn_tot[i + 1] - Zn_tot[i]) / (Xn_tot[i + 1] - Xn_tot[i])) for i in
             range(0, len(Xn_tot) - 1)]

    dA = [np.pi * (Zn_tot[i + 1] + Zn_tot[i]) * np.sqrt(
        (Xn_tot[i + 1] - Xn_tot[i]) ** 2 + (Zn_tot[i + 1] - Zn_tot[i]) ** 2) for i in
          range(0, len(Xn_tot) - 1)]

    alpha_ibl = [np.arctan((Zn_ibl[i + 1] - Zn_ibl[i]) / (Xn_cf_pm[i + 1] - Xn_cf_pm[i])) for i in
                 range(0, len(Xn_cf_pm) - 1)]

    dA_ibl = [np.pi * (Zn_ibl[i + 1] + Zn_ibl[i]) * np.sqrt(
        (Xn_cf_pm[i + 1] - Xn_cf_pm[i]) ** 2 + (Zn_ibl[i + 1] - Zn_ibl[i]) ** 2) for i in
              range(0, len(Xn_cf_pm) - 1)]

    alpha_fvm = [np.arctan((Zn_fvm[i + 1] - Zn_fvm[i]) / (Xn_cf_fvm[i + 1] - Xn_cf_fvm[i])) for i in
                 range(0, len(Xn_cf_fvm) - 1)]

    dA_fvm = [np.pi * (Zn_fvm[i + 1] + Zn_fvm[i]) * np.sqrt(
        (Xn_cf_fvm[i + 1] - Xn_cf_fvm[i]) ** 2 + (Zn_fvm[i + 1] - Zn_fvm[i]) ** 2) for i in
              range(0, len(Xn_cf_fvm) - 1)]

    alpha_fvm_cent = [np.arctan((Zn_fvm_cent[i + 1] - Zn_fvm_cent[i]) / (Xn_cf_fvm_cent[i + 1] - Xn_cf_fvm_cent[i])) for
                      i in
                      range(0, len(Xn_cf_fvm_cent) - 1)]

    dA_fvm_cent = [np.pi * (Zn_fvm_cent[i + 1] + Zn_fvm_cent[i]) * np.sqrt(
        (Xn_cf_fvm_cent[i + 1] - Xn_cf_fvm_cent[i]) ** 2 + (Zn_fvm_cent[i + 1] - Zn_fvm_cent[i]) ** 2) for i in
                   range(0, len(Xn_cf_fvm_cent) - 1)]

    alpha_fvm_rear = [np.arctan((Zn_fvm_rear[i + 1] - Zn_fvm_rear[i]) / (Xn_cf_fvm_rear[i + 1] - Xn_cf_fvm_rear[i])) for
                      i in
                      range(0, len(Xn_cf_fvm_rear) - 1)]

    dA_fvm_rear = [np.pi * (Zn_fvm_rear[i + 1] + Zn_fvm_rear[i]) * np.sqrt(
        (Xn_cf_fvm_rear[i + 1] - Xn_cf_fvm_rear[i]) ** 2 + (Zn_fvm_rear[i + 1] - Zn_fvm_rear[i]) ** 2) for i in
                   range(0, len(Xn_cf_fvm_rear) - 1)]

    ff_fus_tot, fp_fus_tot = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha, dA,
                                      C_p=F_cp(Xs_tot), p_s=None, C_f=F_cf(Xs_tot), tau=None)

    ff_ibl, fp_ibl = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_ibl, dA_ibl,
                              C_p=F_cp(Xs_cf_pm), p_s=None, C_f=F_cf(Xs_cf_pm), tau=None)

    ff_fvm_cent, fp_fvm_cent = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_fvm_cent,
                                        dA_fvm_cent,
                                        C_p=F_cp(Xs_cf_fvm_cent), p_s=None, C_f=F_cf(Xs_cf_fvm_cent), tau=None)

    ff_fvm_rear, fp_fvm_rear = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_fvm_rear,
                                        dA_fvm_rear,
                                        C_p=F_cp(Xs_cf_fvm_rear), p_s=None, C_f=F_cf(Xs_cf_fvm_rear), tau=None)

    ff_fvm, fp_fvm = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_fvm, dA_fvm,
                              C_p=F_cp(Xs_cf_fvm), p_s=None, C_f=F_cf(Xs_cf_fvm), tau=None)

    ft_nac_bot = ff_nacbot + fp_nacbot
    ft_nac_top = ff_nactop + fp_nactop
    ft_fus = ff_fus_tot + fp_fus_tot
    fuse_forces['total'] = [ff_fus_tot, fp_fus_tot, ft_fus]
    nac_forces['total'] = [ff_nactop + ff_nacbot, fp_nactop + fp_nacbot, ft_nac_top + ft_nac_bot]
    nac_forces['top'] = [ff_nactop, fp_nactop, ft_nac_top]
    nac_forces['bottom'] = [ff_nacbot, fp_nacbot, ft_nac_bot]

    all_forces = {
        'nac': {
            'top': {
                'tot': ff_nactop + fp_nactop,
                'visc': ff_nactop,
                'pres': fp_nactop
            },
            'bot': {
                'tot': ff_nacbot + fp_nacbot,
                'visc': ff_nacbot,
                'pres': fp_nacbot
            },
            'tot': {
                'tot': ff_nactop + fp_nactop + ff_nacbot + fp_nacbot,
                'visc': ff_nactop + ff_nacbot,
                'pres': fp_nactop + fp_nacbot
            }
        },
        'fuse': {
            'fvm': {
                'cent': {
                    'tot': ff_fvm_cent + fp_fvm_cent,
                    'visc': ff_fvm_cent,
                    'pres': fp_fvm_cent
                },
                'rear': {
                    'tot': ff_fvm_rear + fp_fvm_rear,
                    'visc': ff_fvm_rear,
                    'pres': fp_fvm_rear
                },
                'tot': {
                    'tot': ff_fvm + fp_fvm,
                    'visc': ff_fvm,
                    'pres': fp_fvm
                }
            },
            'ibl': {
                'tot': ff_ibl + fp_ibl,
                'visc': ff_ibl,
                'pres': fp_ibl
            },
            'tot': {
                'tot': ff_fus_tot + fp_fus_tot,
                'visc': ff_fus_tot,
                'pres': fp_fus_tot
            }
        },
        'tot': {
            'tot': ft_nac_bot + ft_nac_top + ft_fus,
            'visc': ff_fus_tot + ff_nactop + ff_nacbot,
            'pres': fp_fus_tot + fp_nactop + fp_nacbot
        }
    }

    return fuse_forces, nac_forces, all_forces


def calc_inc_areas(x, z):
    x_new = [x[i - 1] + 0.5 * (x[i] - x[i - 1]) if 0 < i < len(x) - 1 else val for i, val in enumerate(x)]
    z_new = [z[i - 1] + 0.5 * (z[i] - z[i - 1]) if 0 < i < len(z) - 1 else val for i, val in enumerate(z)]
    dA = [np.pi * (z_new[i + 1] + z_new[i]) * np.sqrt((z_new[i + 1] - z_new[i]) ** 2 + (x_new[i + 1] - x_new[i]) ** 2)
          for i in
          range(0, len(x_new) - 1)]
    dA.append(0)
    alpha = [np.arctan((z_new[i + 1] - z_new[i]) / (x_new[i + 1] - x_new[i])) for i in range(0, len(x_new) - 1)]
    alpha.append(0)
    return alpha, dA


def sort_lists_exclude_duplicates(main_list, *other_lists):
    # Create a list of tuples where the first element is the main list element and the second is an index
    indexed_list = [(main_list[i], i) for i in range(len(main_list))]
    # Remove duplicates by converting to set and back to list
    unique_values = list(set(main_list))

    # Sort the unique values
    unique_values.sort()

    # Initialize a set to keep track of processed values
    processed_values = set()

    # Initialize lists to store the sorted elements
    sorted_indices = []
    sorted_main_list = []
    sorted_other_lists = [[] for _ in other_lists]

    # Process the unique values
    for value in unique_values:
        # Find all indices with the current value
        indices = [index for val, index in indexed_list if val == value]

        # If the value occurs more than once, exclude it
        if len(indices) > 1 or value in processed_values:
            continue

        # Add the index and value to processed set
        processed_values.add(value)

        # Append the index and corresponding elements to the sorted lists
        sorted_indices.extend(indices)
        sorted_main_list.extend([main_list[i] for i in indices])
        for i, other_list in enumerate(other_lists):
            sorted_other_lists[i].extend([other_list[i] for i in indices])

    return sorted_main_list, *[sorted_other_list for sorted_other_list in sorted_other_lists]


def sort_lists_by_value(main_list, *other_lists):
    # Create a list of tuples where the first element is the main list element and the second is an index
    indexed_list = [(main_list[i], i) for i in range(len(main_list))]
    # Sort the indexed list by the first element of the tuples (the values)
    indexed_list.sort(key=lambda x: x[0])

    # Unpack the sorted indices
    sorted_indices = [index for value, index in indexed_list]

    # Sort the main list
    sorted_main_list = [main_list[i] for i in sorted_indices]

    # Sort the other lists based on the same indices
    sorted_other_lists = [[other[i] for i in sorted_indices] for other in other_lists]

    return sorted_main_list, *sorted_other_lists
