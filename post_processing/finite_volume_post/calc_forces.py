"""Calculate the pressure and skin friction forces of fuselage and nacelle of a PFC geometry from finite volume method
skin friction coefficient and pressure coefficient results.

Author:  A. Habermann
"""

import csv
import numpy as np
import itertools
from post_processing.panel.plot.drag_computation import dragBody


def calc_forces(casepath, atmos, type='full'):
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
    fuse_forces['fuselage_fan'] = []
    fuse_forces['fuselage_rest'] = []
    nac_forces = {}
    nac_forces['total'] = []
    nac_forces['top'] = []
    nac_forces['bottom'] = []

    c_p_fus = []
    c_f_fus = []
    x_fus = []
    z_fus = []
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

        c_p_fus.append(c_p_part)
        c_f_fus.append(c_f_part)
        x_fus.append(x_part)
        z_fus.append(z_part)

    c_p_fus = list(itertools.chain(*c_p_fus))
    c_f_fus = list(itertools.chain(*c_f_fus))
    x_fus = list(itertools.chain(*x_fus))
    z_fus = list(itertools.chain(*z_fus))
    x_fus, z_fus, c_p_fus, c_f_fus = sort_lists_exclude_duplicates(x_fus, z_fus, c_p_fus, c_f_fus)
    alpha_fus, dA_fus = calc_inc_areas(x_fus, z_fus)

    c_p_fus_rest = []
    c_f_fus_rest = []
    x_fus_rest = []
    z_fus_rest = []
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

        c_p_fus_rest.append(c_p_part)
        c_f_fus_rest.append(c_f_part)
        x_fus_rest.append(x_part)
        z_fus_rest.append(z_part)

    c_p_fus_rest = list(itertools.chain(*c_p_fus_rest))
    c_f_fus_rest = list(itertools.chain(*c_f_fus_rest))
    x_fus_rest = list(itertools.chain(*x_fus_rest))
    z_fus_rest = list(itertools.chain(*z_fus_rest))
    x_fus_rest, z_fus_rest, c_p_fus_rest, c_f_fus_rest = sort_lists_exclude_duplicates(x_fus_rest, z_fus_rest,
                                                                                       c_p_fus_rest, c_f_fus_rest)
    alpha_fus_rest, dA_fus_rest = calc_inc_areas(x_fus_rest, z_fus_rest)

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

    ff_nactop, fp_nactop = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_nac_top,
                                    dA_nac_top,
                                    C_p=c_p_nac_top, p_s=None, C_f=c_f_nac_top, tau=None)

    ff_nacbot, fp_nacbot = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_nac_bottom,
                                    dA_nac_bottom,
                                    C_p=c_p_nac_bottom, p_s=None, C_f=c_f_nac_bottom, tau=None)

    ff_fus, fp_fus = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_fus, dA_fus,
                              C_p=c_p_fus, p_s=None, C_f=c_f_fus, tau=None)

    ff_fus_rest, fp_fus_rest = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_fus_rest,
                                        dA_fus_rest,
                                        C_p=c_p_fus_rest, p_s=None, C_f=c_f_fus_rest, tau=None)

    ft_nac_bot = ff_nacbot + fp_nacbot
    ft_nac_top = ff_nactop + fp_nactop
    ft_fus = ff_fus + fp_fus
    ft_fus_rest = ff_fus_rest + fp_fus_rest
    fuse_forces['fuselage_rest'] = [ff_fus_rest, fp_fus_rest, ft_fus_rest]
    fuse_forces['fuselage_fan'] = [ff_fus - ff_fus_rest, fp_fus - fp_fus_rest, ft_fus - ft_fus_rest]
    fuse_forces['total'] = [ff_fus, fp_fus, ft_fus]
    nac_forces['total'] = [ff_nactop + ff_nacbot, fp_nactop + fp_nacbot, ft_nac_top + ft_nac_bot]
    nac_forces['top'] = [ff_nactop, fp_nactop, ft_nac_top]
    nac_forces['bottom'] = [ff_nacbot, fp_nacbot, ft_nac_bot]

    return fuse_forces, nac_forces


def calc_inc_areas(x, z):
    x_new = x
    x_new = [x[i - 1] + 0.5 * (x[i] - x[i - 1]) if 0 < i < len(x) - 1 else val for i, val in enumerate(x)]
    z_new = z
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
