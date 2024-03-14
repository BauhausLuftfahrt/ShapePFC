"""Scale the rotational speed of the CENTRELINE fuselage fan rotor.

Author:  A. Habermann
"""

import numpy as np


def scale_rotor_rotational_speed(d_rot_tip, gamma, R, T_tot):
    omega_rot_orig = 5518 * 0.1047198  # rpm to rad/s
    d_rot_tip_orig = 0.5795 * 2
    gamma_orig = 1.4
    R_orig = 287.058
    T_tot_orig = 255.2
    return (omega_rot_orig * d_rot_tip_orig / d_rot_tip) * np.sqrt(
        (gamma * R * T_tot) / (gamma_orig * R_orig * T_tot_orig))
