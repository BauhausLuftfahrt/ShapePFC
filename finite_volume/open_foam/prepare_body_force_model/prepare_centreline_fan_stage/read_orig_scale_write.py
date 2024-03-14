"""
Scales and writes blade data files (blockage, blockage gradients, body_force_model), which are required for OpenFOAM
simulations. Based on fan stage geometry prepare_centreline_fan_stage Fan B provided by Alejandro Castillo Pardo

Author:  A. Habermann
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from misc_functions.geometry.cylindrical_cartesian_coord_transformation import cylindrical_to_cartesian_rotate
from misc_functions.body_force_model.calculate_metall_blockage import blockage
from finite_volume.open_foam.prepare_body_force_model.prepare_centreline_fan_stage.calc_blockage_gradients import \
    calc_blockage_gradients
from scipy.interpolate import interp1d
from misc_functions.geometry.calculate_normals_tangents import calculate_normals


def read_csv_file(path):
    stations = []
    with open(path, 'r') as f:
        all_lines = f.readlines()
        all_lines = [row.split(' ') for row in all_lines]
        all_lines = [[float(k) for k in row] for row in all_lines]
        for i in range(0, int(len(all_lines) / 500)):
            stations.append(all_lines[i * 500:(i + 1) * 500])
    return stations


def calc_camber(top, bottom):
    top_int_y = interp1d(np.array([i[0] for i in top]), np.array([i[1] for i in top]), fill_value='extrapolate',
                         bounds_error=False)
    top_int_z = interp1d(np.array([i[0] for i in top]), np.array([i[2] for i in top]), fill_value='extrapolate',
                         bounds_error=False)
    bot_int_y = interp1d(np.array([i[0] for i in bottom]), np.array([i[1] for i in bottom]), fill_value='extrapolate',
                         bounds_error=False)
    bot_int_z = interp1d(np.array([i[0] for i in bottom]), np.array([i[2] for i in bottom]), fill_value='extrapolate',
                         bounds_error=False)
    x_new = np.linspace(max(min(np.array([i[0] for i in top])), min(np.array([i[0] for i in bottom]))),
                        min(max(np.array([i[0] for i in top])), max(np.array([i[0] for i in bottom]))), 500)
    thickness = top_int_y(x_new) - bot_int_y(x_new)
    thickness[0] = 0.0
    thickness[-1] = 0.0
    camber_y = bot_int_y(x_new) + 0.5 * (top_int_y(x_new) - bot_int_y(x_new))
    camber = [x_new, camber_y, top_int_z(x_new)]
    return camber, thickness


def scale_cylindrical_coordinates(r, theta, x, delta_r, x_rot_in, scale_fac):
    return r * scale_fac + delta_r, theta, x * scale_fac + x_rot_in


def scale_cart_coordinates(coords, scale_factor, z_hub):
    x_cent = coords[0][0][0] + 0.5 * (coords[-1][-1][0] - coords[0][0][0])
    z_cent = z_hub + 0.5 * (max(max([[j[2] for j in i] for i in coords])) - z_hub)
    x_coords = [[(j[0] - x_cent) * scale_factor + x_cent for j in i] for i in coords]
    z_coords = [[(j[2] - z_cent) * scale_factor + z_cent for j in i] for i in coords]
    return [[(x_coords[i][j], coords[i][j][1], z_coords[i][j]) for j in range(0, len(coords[i]))] for i in
            range(0, len(coords))]


def write_scaled_blade_data(path, h_duct, z_hub, x_rot_in, n_rot, n_stat, plot=False):
    # read normalized cylindrical coordinates
    rel_path = os.path.abspath(__file__)
    rotor_top = read_csv_file(os.path.join(os.path.dirname(rel_path),
                                           'scaled_fan_stage/CENTRELINE_rotor_crosssections_top_scaled.csv'))
    rotor_bot = read_csv_file(os.path.join(os.path.dirname(rel_path),
                                           'scaled_fan_stage/CENTRELINE_rotor_crosssections_bot_scaled.csv'))
    stator_top = read_csv_file(os.path.join(os.path.dirname(rel_path),
                                            'scaled_fan_stage/CENTRELINE_stator_crosssections_top_scaled.csv'))
    stator_bot = read_csv_file(os.path.join(os.path.dirname(rel_path),
                                            'scaled_fan_stage/CENTRELINE_stator_crosssections_bot_scaled.csv'))

    theta_rot_hub_in = rotor_top[0][0][1]
    theta_rot_tip_in = rotor_top[-1][0][1]
    r_scale = ((z_hub + h_duct) / np.cos(theta_rot_tip_in) - z_hub / np.cos(
        theta_rot_hub_in))  # (h_duct) / np.cos(theta_rot_tip_in)
    r_hub_cyl = z_hub / np.cos(theta_rot_hub_in)
    h_duct_scale = r_scale

    # scale cylindrical coordinates
    coord_rot_top = [
        [scale_cylindrical_coordinates(i[0], i[1], i[2], r_hub_cyl, x_rot_in, h_duct_scale) for i in rotor_top[j]]
        for j in range(0, len(rotor_top))]
    coord_rot_bot = [
        [scale_cylindrical_coordinates(i[0], i[1], i[2], r_hub_cyl, x_rot_in, h_duct_scale) for i in rotor_bot[j]]
        for j in range(0, len(rotor_bot))]
    coord_stat_top = [
        [scale_cylindrical_coordinates(i[0], i[1], i[2], r_hub_cyl, x_rot_in, h_duct_scale) for i in stator_top[j]]
        for j in range(0, len(stator_top))]
    coord_stat_bot = [
        [scale_cylindrical_coordinates(i[0], i[1], i[2], r_hub_cyl, x_rot_in, h_duct_scale) for i in stator_bot[j]]
        for j in range(0, len(stator_bot))]

    # transform cylindrical to cartesian coordinates
    cart_coord_rot_top = [[cylindrical_to_cartesian_rotate(i[0], i[1], i[2]) for i in coord_rot_top[j]]
                          for j in range(0, len(coord_rot_top))]
    cart_coord_rot_bot = [[cylindrical_to_cartesian_rotate(i[0], i[1], i[2]) for i in coord_rot_bot[j]]
                          for j in range(0, len(coord_rot_bot))]
    cart_coord_stat_top = [[cylindrical_to_cartesian_rotate(i[0], i[1], i[2]) for i in coord_stat_top[j]]
                           for j in range(0, len(coord_stat_top))]
    cart_coord_stat_bot = [[cylindrical_to_cartesian_rotate(i[0], i[1], i[2]) for i in coord_stat_bot[j]]
                           for j in range(0, len(coord_stat_bot))]

    # enlargement factor (ensure that rotor/stator field is a little bigger than actual field so that at the edges of
    # the rotor/stator regions no weird things happen to blockage and blockage gradients
    f_enl = 1.02
    z_hub_stat = cart_coord_stat_top[0][0][2]
    z_rot_hub_scale = z_hub - ((f_enl - 1.0) / 2 * z_hub)
    z_stat_hub_scale = z_hub_stat - ((f_enl - 1.0) / 2 * z_hub_stat)

    cart_coord_rot_top = scale_cart_coordinates(cart_coord_rot_top, f_enl, z_rot_hub_scale)
    cart_coord_rot_bot = scale_cart_coordinates(cart_coord_rot_bot, f_enl, z_rot_hub_scale)
    cart_coord_stat_top = scale_cart_coordinates(cart_coord_stat_top, f_enl, z_stat_hub_scale)
    cart_coord_stat_bot = scale_cart_coordinates(cart_coord_stat_bot, f_enl, z_stat_hub_scale)
    #
    # for k in range(0,len(coord_rot_top)):
    #     plt.plot([[i[0] for i in coord_rot_top[j]]
    #                  for j in range(0,len(coord_rot_top))][k], [[i[1] for i in coord_rot_top[j]]
    #                  for j in range(0,len(coord_rot_top))][k])
    #     plt.plot([[i[0] for i in coord_rot_bot[j]]
    #                  for j in range(0,len(coord_rot_bot))][k], [[i[1] for i in coord_rot_bot[j]]
    #                  for j in range(0,len(coord_rot_bot))][k])
    #     plt.plot([[i[0] for i in coord_stat_top[j]]
    #                  for j in range(0,len(coord_stat_top))][k], [[i[1] for i in coord_stat_top[j]]
    #                  for j in range(0,len(coord_stat_top))][k])
    #     plt.plot([[i[0] for i in coord_stat_bot[j]]
    #                  for j in range(0,len(coord_stat_bot))][k], [[i[1] for i in coord_stat_bot[j]]
    #                  for j in range(0,len(coord_stat_bot))][k])
    # plt.show()
    #
    # for k in range(0,len(cart_coord_rot_top)):
    #     plt.plot([[i[0] for i in cart_coord_rot_top[j]]
    #                  for j in range(0,len(cart_coord_rot_top))][k], [[i[1] for i in cart_coord_rot_top[j]]
    #                  for j in range(0,len(cart_coord_rot_top))][k])
    #     plt.plot([[i[0] for i in cart_coord_rot_bot[j]]
    #                  for j in range(0,len(cart_coord_rot_bot))][k], [[i[1] for i in cart_coord_rot_bot[j]]
    #                  for j in range(0,len(cart_coord_rot_bot))][k])
    #     plt.plot([[i[0] for i in cart_coord_stat_top[j]]
    #                  for j in range(0,len(cart_coord_stat_top))][k], [[i[1] for i in cart_coord_stat_top[j]]
    #                  for j in range(0,len(cart_coord_stat_top))][k])
    #     plt.plot([[i[0] for i in cart_coord_stat_bot[j]]
    #                  for j in range(0,len(cart_coord_stat_bot))][k], [[i[1] for i in cart_coord_stat_bot[j]]
    #                  for j in range(0,len(cart_coord_stat_bot))][k])
    # plt.show()

    rot_camber = []
    rot_thickness = []
    stat_camber = []
    stat_thickness = []

    # calculate body_force_model
    for i in range(0, len(cart_coord_rot_top)):
        camber_r, thickness_r = calc_camber(cart_coord_rot_top[i], cart_coord_rot_bot[i])
        rot_camber.append(camber_r)
        rot_thickness.append(thickness_r)

        # plt.plot(rot_camber[i][0], rot_camber[i][1])
        # plt.plot(rot_camber[i][0], rot_thickness[i])

    for i in range(0, len(cart_coord_stat_top)):
        camber_s, thickness_s = calc_camber(cart_coord_stat_top[i], cart_coord_stat_bot[i])
        stat_camber.append(camber_s)
        stat_thickness.append(thickness_s)

    #     plt.plot(stat_camber[i][0],stat_camber[i][1])
    #     plt.plot(stat_camber[i][0],stat_thickness[i])
    #
    # plt.axis('equal')
    # plt.show()

    from copy import deepcopy

    for i in range(0, 2):
        rot_camber.insert(0, deepcopy(rot_camber[0][:]))
        stat_camber.insert(0, deepcopy(stat_camber[0][:]))
        rot_thickness.insert(0, deepcopy(rot_thickness[0][:]))
        stat_thickness.insert(0, deepcopy(stat_thickness[0][:]))
        rot_camber[0][2] -= 0.01
        stat_camber[0][2] -= 0.01

    # calculate body_force_model normals
    rot_normals = [calculate_normals(rot_camber[i][0], rot_camber[i][1]) for i in range(0, len(rot_camber))]
    stat_normals = [calculate_normals(stat_camber[i][0], stat_camber[i][1]) for i in range(0, len(stat_camber))]

    # fix problem with rotor leading edge
    rot_normals[43][0][0] = rot_normals[43][1][0]
    rot_normals[43][0][1] = rot_normals[43][1][1]
    rot_normals[43][0][2] = rot_normals[43][1][2]
    rot_normals[44][0][0] = rot_normals[44][1][0]
    rot_normals[44][0][1] = rot_normals[44][1][1]
    rot_normals[44][0][2] = rot_normals[44][1][2]

    # calculate blockage
    rot_blockage = [blockage(n_rot, rot_thickness[i], rot_camber[i][2]) for i in range(0, len(rot_camber))]
    stat_blockage = [blockage(n_stat, stat_thickness[i], stat_camber[i][2]) for i in range(0, len(stat_camber))]

    for i in range(0, len(rot_blockage)):
        rot_blockage[i][0] = 1.0
        rot_blockage[i][-1] = 1.0

    for i in range(0, len(stat_blockage)):
        stat_blockage[i][0] = 1.0
        stat_blockage[i][-1] = 1.0

    rot_blockage = [np.where(arr > 1.0, 1.0, arr) for arr in rot_blockage]
    rot_blockage = [np.where(arr < 0.0, 0.0, arr) for arr in rot_blockage]
    stat_blockage = [np.where(arr > 1.0, 1.0, arr) for arr in stat_blockage]
    stat_blockage = [np.where(arr < 0.0, 0.0, arr) for arr in stat_blockage]

    # fix problem with rotor leading edge
    rot_camber[43][1][0] = rot_camber[43][1][1]
    rot_camber[43][2][0] = rot_camber[43][2][1]
    rot_camber[44][1][0] = rot_camber[44][1][1]
    rot_camber[44][2][0] = rot_camber[44][2][1]

    # calculate blockage gradients
    new_blockage_rotor, new_blockage_stator, grad_blockage_rotor, grad_blockage_stator, coords_rotor, coords_stator \
        = calc_blockage_gradients(np.concatenate(([i[0] for i in rot_camber])),
                                  np.concatenate(([i[2] for i in rot_camber])),
                                  np.concatenate(([i[0] for i in stat_camber])),
                                  np.concatenate(([i[2] for i in stat_camber])),
                                  np.concatenate((rot_blockage)), np.concatenate((stat_blockage)))

    with open(f"{path}/blockage_data", 'w') as nx:
        nx.write('//(x r blockage) \n')
        nx.write('( \n')
        for i in range(0, np.size(coords_rotor[0][:, 0])):
            for j in range(0, np.size(coords_rotor[0][0, :])):
                if not np.isnan(new_blockage_rotor[i][j]):
                    nx.write("(" + str(coords_rotor[0][i][j]) + " " + str(coords_rotor[1][i][j]) + " " + str(
                        new_blockage_rotor[i][j]) + ")\n")
        for i in range(0, np.size(coords_stator[0][:, 0])):
            for j in range(0, np.size(coords_stator[0][0, :])):
                if not np.isnan(new_blockage_stator[i][j]):
                    nx.write("(" + str(coords_stator[0][i][j]) + " " + str(coords_stator[1][i][j]) + " " + str(
                        new_blockage_stator[i][j]) + ")\n")
        nx.write(')')

    for i in range(0, np.size(coords_rotor[0][:, 0])):
        for j in range(0, np.size(coords_rotor[0][0, :])):
            try:
                if np.isnan(grad_blockage_rotor[0][i][j]) and not np.isnan(grad_blockage_rotor[0][i][j + 1]):
                    grad_blockage_rotor[0][i][j] = grad_blockage_rotor[0][i][j + 1]
                if np.isnan(grad_blockage_rotor[0][i][j]) and not np.isnan(grad_blockage_rotor[0][i][j - 1]):
                    grad_blockage_rotor[0][i][j] = grad_blockage_rotor[0][i][j - 1]
                    pass
            except:
                try:
                    if np.isnan(grad_blockage_rotor[0][i][j]) and not np.isnan(grad_blockage_rotor[0][i][j - 1]):
                        grad_blockage_rotor[0][i][j] = grad_blockage_rotor[0][i][j - 1]
                except:
                    pass

    for i in range(0, np.size(coords_rotor[0][:, 0])):
        for j in range(0, np.size(coords_rotor[0][0, :])):
            try:
                if np.isnan(grad_blockage_rotor[1][i][j]) and not np.isnan(grad_blockage_rotor[1][i][j + 1]):
                    grad_blockage_rotor[1][i][j] = grad_blockage_rotor[1][i][j + 1]
                if np.isnan(grad_blockage_rotor[1][i][j]) and not np.isnan(grad_blockage_rotor[1][i][j - 1]):
                    grad_blockage_rotor[1][i][j] = grad_blockage_rotor[1][i][j - 1]
                    pass
            except:
                try:
                    if np.isnan(grad_blockage_rotor[1][i][j]) and not np.isnan(grad_blockage_rotor[1][i][j - 1]):
                        grad_blockage_rotor[1][i][j] = grad_blockage_rotor[1][i][j - 1]
                except:
                    pass

    for i in range(0, np.size(coords_stator[0][:, 0])):
        for j in range(0, np.size(coords_stator[0][0, :])):
            try:
                if np.isnan(grad_blockage_stator[0][i][j]) and not np.isnan(grad_blockage_stator[0][i][j + 1]):
                    grad_blockage_stator[0][i][j] = grad_blockage_stator[0][i][j + 1]
                if np.isnan(grad_blockage_stator[0][i][j]) and not np.isnan(grad_blockage_stator[0][i][j - 1]):
                    grad_blockage_stator[0][i][j] = grad_blockage_stator[0][i][j - 1]
                    pass
            except:
                try:
                    if np.isnan(grad_blockage_stator[0][i][j]) and not np.isnan(grad_blockage_stator[0][i][j - 1]):
                        grad_blockage_stator[0][i][j] = grad_blockage_stator[0][i][j - 1]
                except:
                    pass

    for i in range(0, np.size(coords_stator[0][:, 0])):
        for j in range(0, np.size(coords_stator[0][0, :])):
            try:
                if np.isnan(grad_blockage_stator[1][i][j]) and not np.isnan(grad_blockage_stator[1][i][j + 1]):
                    grad_blockage_stator[1][i][j] = grad_blockage_stator[1][i][j + 1]
                if np.isnan(grad_blockage_stator[1][i][j]) and not np.isnan(grad_blockage_stator[1][i][j - 1]):
                    grad_blockage_stator[1][i][j] = grad_blockage_stator[1][i][j - 1]
                    pass
            except:
                try:
                    if np.isnan(grad_blockage_stator[1][i][j]) and not np.isnan(grad_blockage_stator[1][i][j - 1]):
                        grad_blockage_stator[1][i][j] = grad_blockage_stator[1][i][j - 1]
                except:
                    pass

    with open(f"{path}/blockage_gradient_x", 'w') as nx:
        nx.write('//(x r blockgradx) \n')
        nx.write('( \n')
        for i in range(0, np.size(coords_rotor[0][:, 0])):
            for j in range(0, np.size(coords_rotor[0][0, :])):
                if not np.isnan(grad_blockage_rotor[0][i][j]):
                    nx.write("(" + str(coords_rotor[0][i][j]) + " " + str(coords_rotor[1][i][j]) + " " + str(
                        grad_blockage_rotor[0][i][j]) + ")\n")
        for i in range(0, np.size(coords_stator[0][:, 0])):
            for j in range(0, np.size(coords_stator[0][0, :])):
                if not np.isnan(grad_blockage_stator[0][i][j]):
                    nx.write("(" + str(coords_stator[0][i][j]) + " " + str(coords_stator[1][i][j]) + " " + str(
                        grad_blockage_stator[0][i][j]) + ")\n")
        nx.write(')')

    with open(f"{path}/blockage_gradient_z", 'w') as nx:
        nx.write('//(x r blockgradz) \n')
        nx.write('( \n')
        for i in range(0, np.size(coords_rotor[0][:, 0])):
            for j in range(0, np.size(coords_rotor[0][0, :])):
                if not np.isnan(grad_blockage_rotor[1][i][j]):
                    nx.write("(" + str(coords_rotor[0][i][j]) + " " + str(coords_rotor[1][i][j]) + " " + str(
                        grad_blockage_rotor[1][i][j]) + ")\n")
        for i in range(0, np.size(coords_stator[0][:, 0])):
            for j in range(0, np.size(coords_stator[0][0, :])):
                if not np.isnan(grad_blockage_stator[1][i][j]):
                    nx.write("(" + str(coords_stator[0][i][j]) + " " + str(coords_stator[1][i][j]) + " " + str(
                        grad_blockage_stator[1][i][j]) + ")\n")
        nx.write(')')

    with open(f"{path}/nx_data", 'w') as nx:
        nx.write('//(x r nx) \n')
        nx.write('( \n')
        for i in range(0, len(rot_camber)):
            for j in range(0, len(rot_camber[0][0])):
                nx.write("(" + str(rot_camber[i][0][j]) + " " + str(rot_camber[i][2][j]) + " " + str(
                    rot_normals[i][j][0]) + ")\n")
        for i in range(0, len(stat_camber)):
            for j in range(0, len(stat_camber[0][0])):
                nx.write("(" + str(stat_camber[i][0][j]) + " " + str(stat_camber[i][2][j]) + " " + str(
                    stat_normals[i][j][0]) + ")\n")
        nx.write(')')

    with open(f"{path}/nth_data", 'w') as nx:
        nx.write('//(x r nth) \n')
        nx.write('( \n')
        for i in range(0, len(rot_camber)):
            for j in range(0, len(rot_camber[0][0])):
                nx.write("(" + str(rot_camber[i][0][j]) + " " + str(rot_camber[i][2][j]) + " " + str(
                    rot_normals[i][j][2]) + ")\n")
        for i in range(0, len(stat_camber)):
            for j in range(0, len(stat_camber[0][0])):
                nx.write("(" + str(stat_camber[i][0][j]) + " " + str(stat_camber[i][2][j]) + " " + str(
                    stat_normals[i][j][2]) + ")\n")
        nx.write(')')

    with open(f"{path}/nr_data", 'w') as nx:
        nx.write('//(x r nr) \n')
        nx.write('( \n')
        for i in range(0, len(rot_camber)):
            for j in range(0, len(rot_camber[0][0])):
                nx.write("(" + str(rot_camber[i][0][j]) + " " + str(rot_camber[i][2][j]) + " " + str(
                    rot_normals[i][j][1]) + ")\n")
        for i in range(0, len(stat_camber)):
            for j in range(0, len(stat_camber[0][0])):
                nx.write("(" + str(stat_camber[i][0][j]) + " " + str(stat_camber[i][2][j]) + " " + str(
                    stat_normals[i][j][1]) + ")\n")
        nx.write(')')
        nx.write(')')

    if plot:
        fig3, ax3 = plt.subplots()
        im3 = ax3.pcolormesh([i[0] for i in rot_camber], [i[2] for i in rot_camber],
                             [[i[0] for i in j] for j in rot_normals])
        ax3.pcolormesh([i[0] for i in stat_camber], [i[2] for i in stat_camber],
                       [[i[0] for i in j] for j in stat_normals])
        fig3.colorbar(im3)
        plt.show()

        fig4, ax4 = plt.subplots()
        im4 = ax4.pcolormesh([i[0] for i in rot_camber], [i[2] for i in rot_camber],
                             [[i[2] for i in j] for j in rot_normals])
        ax4.pcolormesh([i[0] for i in stat_camber], [i[2] for i in stat_camber],
                       [[i[2] for i in j] for j in stat_normals])
        fig4.colorbar(im4)
        plt.show()

        fig2, ax2 = plt.subplots()
        im2 = ax2.pcolormesh(coords_rotor[0], coords_rotor[1], new_blockage_rotor)
        ax2.pcolormesh(coords_stator[0], coords_stator[1], new_blockage_stator)
        fig2.colorbar(im2)
        plt.show()

        fig1, ax1 = plt.subplots()
        im1 = ax1.pcolormesh(coords_rotor[0], coords_rotor[1], grad_blockage_rotor[1])
        ax1.pcolormesh(coords_stator[0], coords_stator[1], grad_blockage_stator[1])
        fig1.colorbar(im1)
        plt.show()

        fig0, ax0 = plt.subplots()
        im0 = ax0.pcolormesh(coords_rotor[0], coords_rotor[1], grad_blockage_rotor[0])
        ax0.pcolormesh(coords_stator[0], coords_stator[1], grad_blockage_stator[0])
        fig0.colorbar(im0)
