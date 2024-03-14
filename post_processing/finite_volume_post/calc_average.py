"""Calculate the average value for FV simulation results for a number of timesteps.

Author:  A. Habermann
"""

import numpy as np
from numpy import ndarray as ndarray


def calc_average(var_arr: list or ndarray, sample_no: int):
    return np.sum(var_arr[-sample_no:])/sample_no
