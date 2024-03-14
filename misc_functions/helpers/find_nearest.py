"""Find the nearest element of an array (and its index), which is closest to a given value.

Author:  A. Habermann
"""

import numpy as np


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest_replace(init_array, values: list):
    init_array = np.asarray(init_array)
    if init_array[-1] > init_array[0]:
        sort_idx = 0
    else:
        sort_idx = 1
    for i in range(0, len(values)):
        idx = (np.abs(init_array - values[i])).argmin()
        if init_array[idx] not in values:
            init_array[idx] = values[i]
        else:
            init_array = np.append(init_array, values[i])
    if sort_idx == 0:
        out_array = np.sort(init_array)
    else:
        out_array = np.sort(init_array)
        out_array = out_array[::-1]
    return out_array
