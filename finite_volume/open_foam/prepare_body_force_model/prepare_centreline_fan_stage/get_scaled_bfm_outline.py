"""
Calculates outline of prepare_centreline_fan_stage Fan B rotor and stator blades. Required for geometry generation.

Author:  A. Habermann
"""

import os
import numpy as np
from misc_functions.geometry.cylindrical_cartesian_coord_transformation import cylindrical_to_cartesian_rotate


def get_bfm_outline(h_duct, z_hub, x_rot_in, plot=False):
    rel_path = os.path.abspath(__file__)
    outline = np.array(np.loadtxt(os.path.join(os.path.dirname(rel_path),
                                               'scaled_fan_stage/CENTRELINE_fanstage_outline_scaled.csv'),
                                  delimiter=",",
                                  skiprows=1))
    theta_rot_in = 0.15844784  # 0.158621
    theta_rot_tip_in = outline[-1][2]
    theta_rot_hub_in = outline[0][2]
    r_scale = ((z_hub + h_duct) / np.cos(theta_rot_tip_in) - z_hub / np.cos(theta_rot_hub_in))
    delta_r = z_hub / np.cos(theta_rot_hub_in)
    # r_scale = h_duct / np.cos(theta_rot_in)
    z_rotor_in = outline[:, 0] * r_scale + x_rot_in
    r_rotor_in = outline[:, 1] * r_scale + delta_r
    z_rotor_out = outline[:, 3] * r_scale + x_rot_in
    r_rotor_out = outline[:, 4] * r_scale + delta_r
    z_stator_in = outline[:, 6] * r_scale + x_rot_in
    r_stator_in = outline[:, 7] * r_scale + delta_r
    z_stator_out = outline[:, 9] * r_scale + x_rot_in
    r_stator_out = outline[:, 10] * r_scale + delta_r

    rotor_in_coords = [cylindrical_to_cartesian_rotate(r_rotor_in[i], outline[i, 2], z_rotor_in[i]) for i in
                       range(0, len(r_rotor_in))]
    rotor_out_coords = [cylindrical_to_cartesian_rotate(r_rotor_out[i], outline[i, 5], z_rotor_out[i]) for i in
                        range(0, len(r_rotor_out))]
    stator_in_coords = [cylindrical_to_cartesian_rotate(r_stator_in[i], outline[i, 8], z_stator_in[i]) for i in
                        range(0, len(r_stator_in))]
    stator_out_coords = [cylindrical_to_cartesian_rotate(r_stator_out[i], outline[i, 11], z_stator_out[i]) for i in
                         range(0, len(r_stator_out))]


    return [[i[0] for i in rotor_in_coords], [i[2] for i in rotor_in_coords]], \
           [[i[0] for i in rotor_out_coords], [i[2] for i in rotor_out_coords]], \
           [[i[0] for i in stator_in_coords], [i[2] for i in stator_in_coords]], \
           [[i[0] for i in stator_out_coords], [i[2] for i in stator_out_coords]]
