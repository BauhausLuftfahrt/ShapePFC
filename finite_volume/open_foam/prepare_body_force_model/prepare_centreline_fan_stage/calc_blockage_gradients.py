"""
Calculate blockage gradients required for body force model.

Author:  A. Habermann
"""

import numpy as np
from scipy.interpolate import griddata


def calc_blockage_gradients(rot_camber_x, rot_camber_z, stat_camber_x, stat_camber_z, rot_blockage, stat_blockage):
    extension_rotor_x = 0.0
    extension_stator_x = 0.0
    new_x_r, new_z_r = np.mgrid[np.min(rot_camber_x) - extension_rotor_x:np.max(rot_camber_x) + extension_rotor_x:200j,
                       np.min(rot_camber_z) - extension_rotor_x:np.max(rot_camber_z) + extension_rotor_x:200j]
    new_x_s, new_z_s = np.mgrid[
                       np.min(stat_camber_x) - extension_stator_x:np.max(stat_camber_x) + extension_stator_x:200j,
                       np.min(stat_camber_z) - extension_stator_x:np.max(stat_camber_z) + extension_stator_x:200j]

    blockage_cubic_r = griddata(tuple((rot_camber_x, rot_camber_z)),
                                np.array(rot_blockage), (new_x_r, new_z_r), method='linear')
    blockage_cubic_s = griddata(tuple((stat_camber_x, stat_camber_z)),
                                np.array(stat_blockage), (new_x_s, new_z_s), method='linear')

    blockage_cubic_r[blockage_cubic_r > 1.] = 1.
    blockage_cubic_s[blockage_cubic_s > 1.] = 1.

    blockage_cubic_r[blockage_cubic_r < 0.] = 0.
    blockage_cubic_s[blockage_cubic_s < 0.] = 0.

    grad_x_r, grad_z_r = np.gradient(blockage_cubic_r, new_x_r[:, 0], new_z_r[0, :])
    grad_x_s, grad_z_s = np.gradient(blockage_cubic_s, new_x_s[:, 0], new_z_s[0, :])

    return blockage_cubic_r, blockage_cubic_s, [grad_x_r, grad_z_r], [grad_x_s, grad_z_s], [new_x_r, new_z_r], [new_x_s,
                                                                                                                new_z_s]
