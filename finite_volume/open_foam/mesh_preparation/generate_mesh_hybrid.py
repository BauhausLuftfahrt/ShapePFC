"""
Prepare .geo file for parameterized grid generation and create files required for body force model in finite volume
simulation.

Author:  A. Habermann
"""


import numpy as np
import os
from misc_functions.helpers.find_nearest import find_nearest_index
from finite_volume.open_foam.prepare_body_force_model.prepare_centreline_fan_stage.read_orig_scale_write import write_scaled_blade_data
from misc_functions.helpers.dimensionless_wall_distance import first_cell_height_y_plus
from finite_volume.open_foam.mesh_preparation.gmsh_prepare_funs import rel_value
from finite_volume.open_foam.mesh_preparation.offset_curves import find_intersections, offset_curve, extrapolate_curve_rear, \
    interpolate_between_curves, insert_coordinate, insert_and_cut_array, interpolate_and_cut_section, \
    interpolate_between_curves_vertical, intersect_and_cut_ff_stage, insert_and_cut_array_reverse
from scipy.interpolate import interp1d
from finite_volume.open_foam.mesh_preparation.progression_calc import calc_len_from_no_and_first_cell, \
    calc_ratio_and_cellno_from_first_and_last_cell, calc_ratio_from_cellno_and_first_cell, calc_last_cell_height, \
    calculate_arc_length
from geometry_generation.panel_geometry.parameter_sampling import translate_points


def generate_mesh_and_blade_data(wall_res, gridres, Ma, alt, geo, interface_loc, casepath, geotype):

    fuselage = geo['fuselage']
    nacelle_top = geo['nacelle_top']
    nacelle_bottom = geo['nacelle_bottom']
    rotor_inlet = geo['rotor_inlet']
    rotor_outlet = geo['rotor_outlet']
    stator_inlet = geo['stator_inlet']
    stator_outlet = geo['stator_outlet']

    wedge_angle = np.deg2rad(4)
    # transform y positions for half wedge angle. else, small error in y-coordinates is introduced by wedge generation later on
    fuselage = [[i[0], i[1]/np.cos(wedge_angle/2)] for i in fuselage]
    nacelle_top = [[i[0], i[1]/np.cos(wedge_angle/2)] for i in nacelle_top]
    nacelle_bottom = [[i[0], i[1]/np.cos(wedge_angle/2)] for i in nacelle_bottom]
    rotor_inlet = [[i[0], i[1]/np.cos(wedge_angle/2)] for i in rotor_inlet]
    rotor_outlet = [[i[0], i[1]/np.cos(wedge_angle/2)] for i in rotor_outlet]
    stator_inlet = [[i[0], i[1]/np.cos(wedge_angle/2)] for i in stator_inlet]
    stator_outlet = [[i[0], i[1]/np.cos(wedge_angle/2)] for i in stator_outlet]

    r_fuse = geo['r_fus']/np.cos(wedge_angle/2)

    # if wall functions employed, 30 < y+ < 150; else, y+ < 1
    # number of cells inside boundary layers
    if wall_res == 'wf':
        y_plus_fuse = 100.0
        y_plus_nac = 50.0
        no_bl_cells_fuse = 14
        no_bl_cells_nac = 18
        if gridres == 'coarse':
            grid_factor = 0.25
            ff_grid_factor = 0.35
        elif gridres == 'medium':
            grid_factor = 0.5
            ff_grid_factor = 0.5
        elif gridres == 'fine':
            grid_factor = 0.85
            ff_grid_factor = 0.85
        elif gridres == 'finer':
            grid_factor = 1.4
            ff_grid_factor = 1.4
        else:
            grid_factor = 1.0
            ff_grid_factor = 1.0
    elif wall_res == 'fr':
        y_plus_fuse = y_plus_nac = 0.95
        no_bl_cells_fuse = 40
        no_bl_cells_nac = 40
        if gridres == 'coarse':
            grid_factor = 0.18
            ff_grid_factor = 0.18
        elif gridres == 'medium':
            grid_factor = 0.45
            ff_grid_factor = 0.45
        elif gridres == 'fine':
            grid_factor = 0.82
            ff_grid_factor = 0.82
        elif gridres == 'finer':
            grid_factor = 1.4
            ff_grid_factor = 1.4
        else:
            grid_factor = 1.0
            ff_grid_factor = 1.0
    else:
        y_plus_fuse = y_plus_nac = 1.0
        no_bl_cells_fuse = 20
        no_bl_cells_nac = 20

    n_h_nac_1_orig = 45
    n_h_nac_3_orig = 140
    n_h_nose_1_orig = 200
    n_h_rear_1_orig = 400
    n_h_rear_2_orig = 450
    n_h_cent_1_orig = 750

    n_h_inlet_orig = 160
    n_h_rot_orig = 60
    n_h_gap_orig = 31
    n_h_stat_orig = 42
    n_h_nozzle_orig = 200

    n_v_front_1 = 120
    n_v_front_2 = 150

    n_nac_orig = 280
    n_fuse_front_orig = 450
    n_rear_domain_orig = 850
    n_v_tot = 400

    n_ff_orig = n_h_inlet_orig+n_h_rot_orig+n_h_gap_orig+n_h_stat_orig+n_h_nozzle_orig

    n_fuse_front_tot = round(grid_factor*n_fuse_front_orig)
    n_rear_domain_tot = round(grid_factor*n_rear_domain_orig)

    n_v_tot *= grid_factor
    n_h_rot_orig *= ff_grid_factor
    n_h_inlet_orig *= ff_grid_factor
    n_ff_orig *= ff_grid_factor
    n_h_nozzle_orig *= ff_grid_factor

    # don't change these values! based on reference geometry
    l_nac_orig = 2.5301828571428544
    l_fuse_front_orig = 61.41939928438484
    l_ff_stage_orig = 2.5301828571428544
    l_rear_domain_orig = 67.20514974207987
    h_domain_orig = 33.495

    # these widths could be kept constant for all meshes
    # rotor_first_cell_width = 0.001
    nacelle_te_cell_width = 0.002
    nacelle_le_cell_width = 0.001
    fuse_te_width = 0.01/grid_factor
    fuse_nose_le_cell_width = 0.01
    fuse_nose_te_cell_width = 0.03
    mid_inlet_width = 0.01/grid_factor
    nacelle_max_cell_width = 0.002/grid_factor
    nacelle_highlight_width = 0.005/grid_factor
    interface_cell_width = 8*fuse_nose_te_cell_width

    # assumption: no. of rotor/stator blades does not change between designs
    n_rot = 20
    n_stat = 43

    X = [[fuselage[i][0] for i in range(0,len(fuselage))],
         [nacelle_top[i][0] for i in range(0,len(nacelle_top))],
         [nacelle_bottom[i][0] for i in reversed(range(0,len(nacelle_bottom)))],
         [i[0] for i in rotor_inlet],
         [i[0] for i in rotor_outlet],
         [i[0] for i in stator_inlet],
         [i[0] for i in stator_outlet]]
    Y = [[fuselage[i][1] for i in range(0,len(fuselage))],
         [nacelle_top[i][1] for i in range(0,len(nacelle_top))],
         [nacelle_bottom[i][1] for i in reversed(range(0,len(nacelle_bottom)))],
         [i[1] for i in rotor_inlet],
         [i[1] for i in rotor_outlet],
         [i[1] for i in stator_inlet],
         [i[1] for i in stator_outlet]]

    interp_fuse = interp1d([fuselage[i][0] for i in range(0,len(fuselage))],[fuselage[i][1] for i in range(0,len(fuselage))])

    x_0 = nacelle_top[0][0]

    x_2_f = rotor_inlet[0][0]
    x_2_n = rotor_inlet[-1][0]
    x_22_f = rotor_outlet[0][0]
    x_22_n = rotor_outlet[-1][0]

    x_23_f = stator_inlet[0][0]
    x_23_n = stator_inlet[-1][0]
    x_3_f = stator_outlet[0][0]
    x_3_n = stator_outlet[-1][0]

    x_8 = nacelle_top[-1][0]
    x_8f = X[0][find_nearest_index(X[0], nacelle_top[-1][0])]

    h_duct_in = geo['h_duct_in']/np.cos(wedge_angle/2)
    h_duct_out = geo['h_duct_out']/np.cos(wedge_angle/2)
    l_nac = X[1][-1]-X[1][0]

    l_fuse = geo['l_fuse']
    ldomain = 2*l_fuse
    hdomain = 17*r_fuse #10

    # calculate boundary layers
    # ratio of cells inside boundary layers
    if gridres == 'wf':
        bl_ratio = 1.1
    else:
        bl_ratio = 1.2

    # height of first cell depending on BL resolution
    bl_l_first_nac = float(first_cell_height_y_plus(y_plus_nac, Ma, alt, l_nac))
    bl_l_first_fuse = float(first_cell_height_y_plus(y_plus_fuse, Ma, alt, l_fuse))

    bl_height_nac = calc_len_from_no_and_first_cell(no_bl_cells_nac, bl_l_first_nac, bl_ratio)
    bl_height_fuse = calc_len_from_no_and_first_cell(no_bl_cells_fuse, bl_l_first_fuse, bl_ratio)

    if bl_height_nac >= 0.8*h_duct_in/2:
        no_bl_cells_nac = int(no_bl_cells_nac/2)
        bl_height_nac = calc_len_from_no_and_first_cell(no_bl_cells_nac, bl_l_first_nac, bl_ratio)

    if bl_height_fuse >= 0.8*h_duct_in/2:
        no_bl_cells_fuse = int(no_bl_cells_fuse/2)
        bl_height_fuse = calc_len_from_no_and_first_cell(no_bl_cells_fuse, bl_height_fuse, bl_ratio)

    # offset whole nacelle
    nacelle_tot_x = [nacelle_bottom[i][0] for i in range(0,len(nacelle_bottom))]+\
                    [nacelle_top[i][0] for i in range(0,len(nacelle_top))]
    nacelle_tot_y = [nacelle_bottom[i][1] for i in range(0,len(nacelle_bottom))]+\
                    [nacelle_top[i][1] for i in range(0,len(nacelle_top))]
    bl_nac_cowling_tot_x, bl_nac_cowling_tot_y = offset_curve(nacelle_tot_x,nacelle_tot_y, bl_height_nac)

    h_duct_in_int = bl_nac_cowling_tot_y[np.where(bl_nac_cowling_tot_x == min(bl_nac_cowling_tot_x))[0][-1]]-\
                interp_fuse(min(bl_nac_cowling_tot_x))

    h_duct_in = h_duct_in_int*np.cos(np.deg2rad(geo['teta_int_in']))

    # fuselage nose domain
    idx_nose_half = int(np.where(Y[0] == np.array(r_fuse))[0][0] * 0.3)
    idx_nose = int(np.where(Y[0] == np.array(r_fuse))[0][0])
    x_c0_1 = X[0][0:idx_nose_half]
    x_c0_2 = X[0][idx_nose_half:idx_nose]

    x_nose_total = X[0][0:idx_nose]
    y_nose_total = Y[0][0:idx_nose]

    x_c3_0 = translate_points(x_nose_total,x_nose_total[0],x_nose_total[-1],-3*h_duct_in,x_nose_total[-1])
    y_c3_0 = translate_points(y_nose_total,y_nose_total[0],y_nose_total[-1],y_nose_total[0],y_nose_total[-1]+3*h_duct_in)
    x_c4_0 = translate_points(x_nose_total,x_nose_total[0],x_nose_total[-1],-ldomain/2,x_nose_total[-1])  #-hdomain
    y_c4_0 = translate_points(y_nose_total,y_nose_total[0],y_nose_total[-1],y_nose_total[0],y_nose_total[-1]+hdomain)

    # x_c1_1, y_c1_1 = x_c1_0[0:idx_nose_half], y_c1_0[0:idx_nose_half]
    x_c3_1, y_c3_1 = x_c3_0[0:idx_nose_half], y_c3_0[0:idx_nose_half]
    x_c4_1, y_c4_1 = x_c4_0[0:idx_nose_half], y_c4_0[0:idx_nose_half]

    # x_c1_2, y_c1_2 = x_c1_0[idx_nose_half:], y_c1_0[idx_nose_half:]
    x_c3_2, y_c3_2 = x_c3_0[idx_nose_half:], y_c3_0[idx_nose_half:]
    x_c4_2, y_c4_2 = x_c4_0[idx_nose_half:], y_c4_0[idx_nose_half:]

    x_int = interface_loc*geo['l_cent_f']
    if not np.where(X[0] == np.array(x_int))[0]:
        X[0], Y[0] = insert_coordinate(X[0], Y[0], [x_int, r_fuse])

    idx_int = int(np.where(X[0] == np.array(x_int))[0])

    # fuselage center section
    idx_cent = int(np.where(Y[0] == np.array(r_fuse))[0][-1])
    x_c0_3, y_c0_3 = insert_and_cut_array(X[0][idx_nose:idx_cent],Y[0][idx_nose:idx_cent],
                                      [(X[0][idx_nose]+0.5*(X[0][idx_cent]-X[0][idx_nose])),Y[0][idx_cent]])
    x_c0_7, y_c0_7 = insert_and_cut_array_reverse(X[0][idx_nose:idx_cent],Y[0][idx_nose:idx_cent],
                                      [(X[0][idx_nose]+0.5*(X[0][idx_cent]-X[0][idx_nose])),Y[0][idx_cent]])
    X[0], Y[0] = insert_coordinate(X[0], Y[0], [(X[0][idx_nose]+0.5*(X[0][idx_cent]-X[0][idx_nose])),Y[0][idx_cent]])

    x_c3_3, y_c3_3 = offset_curve(x_c0_3, y_c0_3, 3*h_duct_in)
    x_c4_3, y_c4_3 = offset_curve(x_c0_3, y_c0_3, hdomain)

    x_c3_7, y_c3_7 = offset_curve(x_c0_7, y_c0_7, 3*h_duct_in)
    x_c4_7, y_c4_7 = offset_curve(x_c0_7, y_c0_7, hdomain)

    # fuselage inlet to FF stage section
    idx_fuse_0 = int(np.where(X[0] == np.array(x_0))[0][0])
    idx_rear_half = int(idx_cent+0.5*(idx_fuse_0-idx_cent))
    x_c0_4 = X[0][idx_cent:idx_rear_half]
    y_c0_4 = Y[0][idx_cent:idx_rear_half]
    x_c3_4, y_c3_4 = offset_curve(x_c0_4, y_c0_4, 3*h_duct_in)
    x_c4_4= x_c3_4#np.linspace(X[0][idx_cent-1],X[0][idx_rear_half],100)
    y_c4_4 = np.full_like(x_c4_4,y_c4_3[-1])

    x_c0_5 = X[0][idx_rear_half:idx_fuse_0+1]
    y_c0_5 = Y[0][idx_rear_half:idx_fuse_0+1]
    x_c3_5, y_c3_5 = [min(bl_nac_cowling_tot_x),nacelle_top[0][0]], [bl_nac_cowling_tot_y[np.where(min(bl_nac_cowling_tot_x)==bl_nac_cowling_tot_x)[0][0]], nacelle_top[0][1]]
    x_c4_5 = x_c3_5#np.linspace(X[0][idx_rear_half-1],X[0][idx_fuse_0],100)
    y_c4_5 = np.full_like(x_c4_5,y_c4_3[-1])

    # find location where nacelle BL contour and internal line parallel to fuselage intersect
    int_nacelle_bl_fuselage_rear_mid = find_intersections(bl_nac_cowling_tot_x, bl_nac_cowling_tot_y, x_c3_5, y_c3_5)[0]

    # insert point in fuselage contour
    # insert intersection with nacelle point in internal line
    x_c3_5, y_c3_5 = insert_and_cut_array(x_c3_5,y_c3_5,int_nacelle_bl_fuselage_rear_mid)
    # cut other lines to x
    x_c0_5, y_c0_5 = extrapolate_curve_rear(x_c0_5, y_c0_5, int_nacelle_bl_fuselage_rear_mid[0])

    # fuselage tail
    idx_fuse_8 = int(find_nearest_index(np.array(X[0]), X[1][-1]))
    x_c0_16 = X[0][idx_fuse_8:]
    y_c0_16 = Y[0][idx_fuse_8:]
    x_c2_16, y_c2_16 = offset_curve(x_c0_16, y_c0_16, (h_duct_out-bl_height_nac)/2)
    x_c3_16, y_c3_16 = offset_curve(x_c0_16, y_c0_16, h_duct_out)
    x_c4_16 = np.linspace(X[0][idx_fuse_8],X[0][-1],100)
    y_c4_16 = np.full_like(x_c4_16,y_c4_3[-1])
    x_c5_16, y_c5_16 = offset_curve(x_c0_16, y_c0_16, h_duct_out-bl_height_nac)
    x_c6_16, y_c6_16 = offset_curve(x_c0_16, y_c0_16, h_duct_out+bl_height_nac)
    x_c2_16, y_c2_16 = extrapolate_curve_rear(x_c2_16, y_c2_16, x_c0_16[-1])
    x_c3_16, y_c3_16 = extrapolate_curve_rear(x_c3_16, y_c3_16, x_c0_16[-1])
    x_c5_16, y_c5_16 = extrapolate_curve_rear(x_c5_16, y_c5_16, x_c0_16[-1])
    x_c6_16, y_c6_16 = extrapolate_curve_rear(x_c6_16, y_c6_16, x_c0_16[-1])

    y_c3_16 = [y_c3_16[0]]*len(y_c3_16)
    y_4_17 = y_c3_16[-1]

    # domain behind fuselage
    x_rear_half = l_fuse+ldomain/10
    x_rear_back = ldomain+l_fuse
    y_0_17 = 0.0
    y_2_17 = (h_duct_out-bl_height_nac)/2
    y_3_17 = h_duct_out-bl_height_nac
    y_6_17 = y_c4_3[-1]

    # upper nacelle
    nacelle_top_x, nacelle_top_y = [nacelle_top[i][0] for i in range(0,len(nacelle_top))],\
                                   [nacelle_top[i][1] for i in range(0,len(nacelle_top))]
    x_nac_max = geo['x_nac_max']#[int(np.where(nacelle_top_y==max(nacelle_top_y))[0])]
    x_back_sep = x_nac_max+0.5*(x_8-x_nac_max)

    idx_split_bl_nac = np.where(bl_nac_cowling_tot_x == min(bl_nac_cowling_tot_x))[0][-1]
    bl_nac_bottom_x = bl_nac_cowling_tot_x[0:idx_split_bl_nac+1]
    bl_nac_top_x = bl_nac_cowling_tot_x[idx_split_bl_nac:]
    bl_nac_bottom_y = bl_nac_cowling_tot_y[0:idx_split_bl_nac+1]
    bl_nac_top_y = bl_nac_cowling_tot_y[idx_split_bl_nac:]

    x_max, y_max = [x_nac_max, x_nac_max], [0, hdomain]
    x_max_half, y_max_half = [x_0+0.5*(x_nac_max-x_0), x_0+0.5*(x_nac_max-x_0)], [0, hdomain]
    x_back, y_back = [x_back_sep, x_back_sep], [0, hdomain]

    # cut all upper lines in pieces
    list_vert_lines_top = [[x_max_half, y_max_half],
                       [x_max, y_max],
                       [x_back, y_back]
    ]

    lines_0, nacelle_top_x, nacelle_top_y, bl_nac_top_x, bl_nac_top_y,indices_nac_top,indices_bl_nac_top = \
        intersect_and_cut_ff_stage([nacelle_top_x, nacelle_top_y], [bl_nac_top_x, bl_nac_top_y],
                                   list_vert_lines_top)

    # FF stage middle line
    x_c_fuse_ff, y_c_fuse_ff = interpolate_and_cut_section(X[0],Y[0],int_nacelle_bl_fuselage_rear_mid[0],x_8,200)
    x_mid_ff_stage, y_mid_ff_stage = interpolate_between_curves(x_c_fuse_ff, y_c_fuse_ff,bl_nac_bottom_x,
                                                bl_nac_bottom_y)

    # FF BL contour
    x_bl_ff_stage, y_bl_ff_stage = offset_curve(x_c_fuse_ff, y_c_fuse_ff, bl_height_fuse)

    # from shapely.geometry import LineString
    # # check if offest curve intersects itself
    # line_check = LineString([(x_bl_ff_stage[i],y_bl_ff_stage[i]) for i in range(0,len(x_bl_ff_stage))])
    #
    # if not line_check.is_simple:
    #     bl_height_fuse *= 0.5
    #     x_bl_ff_stage, y_bl_ff_stage = offset_curve(x_c_fuse_ff, y_c_fuse_ff, bl_height_fuse)
    #     line_check = LineString([(x_bl_ff_stage[i], y_bl_ff_stage[i]) for i in range(0, len(x_bl_ff_stage))])
    #     if not line_check.is_simple:
    #         bl_height_fuse *= 0.5
    #         x_bl_ff_stage, y_bl_ff_stage = offset_curve(x_c_fuse_ff, y_c_fuse_ff, bl_height_fuse)

    y_1_17 = bl_height_fuse

    x_c1_3, y_c1_3 = offset_curve(x_c0_3, y_c0_3, bl_height_fuse)
    x_c1_7, y_c1_7 = offset_curve(x_c0_7, y_c0_7, bl_height_fuse)
    x_c1_4, y_c1_4 = offset_curve(x_c0_4, y_c0_4, bl_height_fuse)

    x_c1_5, y_c1_5 = offset_curve(x_c0_5, y_c0_5, bl_height_fuse)
    x_c1_5, y_c1_5 = extrapolate_curve_rear(x_c1_5, y_c1_5, int_nacelle_bl_fuselage_rear_mid[0])

    x_c1_16, y_c1_16 = offset_curve(x_c0_16, y_c0_16, bl_height_fuse)
    x_c1_16, y_c1_16 = extrapolate_curve_rear(x_c1_16, y_c1_16, x_c0_16[-1])

    x_c1_0 = translate_points(x_nose_total,x_nose_total[0],x_nose_total[-1],-bl_height_fuse,x_nose_total[-1])
    y_c1_0 = translate_points(y_nose_total,y_nose_total[0],y_nose_total[-1],y_nose_total[0],y_nose_total[-1]+bl_height_fuse)

    y_ff_inlet = np.linspace(y_c_fuse_ff[0],bl_nac_top_y[0],50)
    x_ff_inlet = np.full_like(y_ff_inlet, int_nacelle_bl_fuselage_rear_mid[0])

    y_ff_nozzle = np.linspace(y_c_fuse_ff[-1],nacelle_bottom[0][1],50)
    x_ff_nozzle = np.full_like(y_ff_nozzle, x_8)

    # vertical lines in FF stage
    x_mid_inlet, y_mid_inlet = interpolate_between_curves_vertical(x_ff_inlet, y_ff_inlet, [i[0] for i in rotor_inlet],
                                                      [i[1] for i in rotor_inlet],min(min([i[1] for i in rotor_inlet]),
                                                                                      min(y_ff_inlet)),
                                                      max(max([i[1] for i in rotor_inlet]),max(y_ff_inlet)))
    #
    # x_mid_rotor, y_mid_rotor = interpolate_between_curves_vertical([i[0] for i in rotor_inlet], [i[1] for i in rotor_inlet],
    #                                                                [i[0] for i in rotor_outlet], [i[1] for i in rotor_outlet],
    #                                                                min(min([i[1] for i in rotor_inlet]),
    #                                                                    min([i[1] for i in rotor_outlet])),
    #                                                                 max(max([i[1] for i in rotor_inlet]),
    #                                                                     max([i[1] for i in rotor_outlet])))
    #
    # x_mid_gap, y_mid_gap = interpolate_between_curves_vertical([i[0] for i in rotor_outlet], [i[1] for i in rotor_outlet],
    #                                                                [i[0] for i in stator_inlet], [i[1] for i in stator_inlet],
    #                                                                min(min([i[1] for i in rotor_outlet]),
    #                                                                    min([i[1] for i in stator_inlet])),
    #                                                                 max(max([i[1] for i in rotor_outlet]),
    #                                                                     max([i[1] for i in stator_inlet])))
    #
    # x_mid_stator, y_mid_stator = interpolate_between_curves_vertical([i[0] for i in stator_inlet], [i[1] for i in stator_inlet],
    #                                                                [i[0] for i in stator_outlet], [i[1] for i in stator_outlet],
    #                                                                min(min([i[1] for i in stator_inlet]),
    #                                                                    min([i[1] for i in stator_outlet])),
    #                                                                 max(max([i[1] for i in stator_inlet]),
    #                                                                     max([i[1] for i in stator_outlet])))

    x_mid_nozzle, y_mid_nozzle = interpolate_between_curves_vertical(x_ff_nozzle, y_ff_nozzle,
                                                                   [i[0] for i in stator_outlet], [i[1] for i in stator_outlet],
                                                                   min(min([i[1] for i in stator_inlet]),
                                                                       min(y_ff_nozzle)),
                                                                    max(max([i[1] for i in stator_inlet]),
                                                                        max(y_ff_nozzle)))

    x_c_fuse_ff, y_c_fuse_ff = insert_coordinate(x_c_fuse_ff, y_c_fuse_ff,rotor_inlet[0])
    x_c_fuse_ff, y_c_fuse_ff = insert_coordinate(x_c_fuse_ff, y_c_fuse_ff,rotor_outlet[0])
    x_c_fuse_ff, y_c_fuse_ff = insert_coordinate(x_c_fuse_ff, y_c_fuse_ff,stator_inlet[0])
    x_c_fuse_ff, y_c_fuse_ff = insert_coordinate(x_c_fuse_ff, y_c_fuse_ff,stator_outlet[0])

    nacelle_bottom_x, nacelle_bottom_y = insert_coordinate([nacelle_bottom[i][0] for i in range(0,len(nacelle_bottom))],
                                                  [nacelle_bottom[i][1] for i in range(0,len(nacelle_bottom))],rotor_inlet[-1])
    nacelle_bottom_x, nacelle_bottom_y = insert_coordinate(nacelle_bottom_x, nacelle_bottom_y,rotor_outlet[-1])
    nacelle_bottom_x, nacelle_bottom_y = insert_coordinate(nacelle_bottom_x, nacelle_bottom_y,stator_inlet[-1])
    nacelle_bottom_x, nacelle_bottom_y = insert_coordinate(nacelle_bottom_x, nacelle_bottom_y,stator_outlet[-1])

    # half curve behind nacelle trailing edge
    teta_te = (geo['beta_te_up']-geo['beta_te_low'])/2+geo['beta_te_low']
    nacelle_te_x = [nacelle_bottom_x[-1]-np.tan(np.deg2rad(teta_te))*0.5,nacelle_bottom_x[-1],
                    nacelle_bottom_x[-1]+np.tan(np.deg2rad(teta_te))*0.5]
    nacelle_te_y = [nacelle_bottom_y[-1]-0.5,nacelle_bottom_y[-1],
                    nacelle_bottom_y[-1]+0.5]

    # cut all lines in pieces
    list_vert_lines_ff_stage = [[x_mid_inlet, y_mid_inlet],
                       [[i[0] for i in rotor_inlet], [i[1] for i in rotor_inlet]],
                       [[i[0] for i in rotor_outlet], [i[1] for i in rotor_outlet]],
                       [[i[0] for i in stator_inlet], [i[1] for i in stator_inlet]],
                       [[i[0] for i in stator_outlet], [i[1] for i in stator_outlet]],
                       [x_mid_nozzle, y_mid_nozzle]
    ]

    try:
        lines_4, nacelle_bottom_x, nacelle_bottom_y, _, _,indices_nac_bottom,_ = \
            intersect_and_cut_ff_stage([nacelle_bottom_x, nacelle_bottom_y],[bl_nac_bottom_x, bl_nac_bottom_y],
                                       list_vert_lines_ff_stage)

        lines_3, bl_nac_bottom_x, bl_nac_bottom_y,_, _,indices_bl_nac_bottom,_ = intersect_and_cut_ff_stage([bl_nac_bottom_x, bl_nac_bottom_y],
                                             [x_mid_ff_stage, y_mid_ff_stage],
                                             list_vert_lines_ff_stage)

        lines_2, _, _, x_mid_ff_stage, y_mid_ff_stage, _, indices_mid_ff_stage = intersect_and_cut_ff_stage(
            [x_bl_ff_stage, y_bl_ff_stage],
            [x_mid_ff_stage, y_mid_ff_stage],
            list_vert_lines_ff_stage)

        lines_1, x_bl_ff_stage, y_bl_ff_stage, x_c_fuse_ff, y_c_fuse_ff, indices_bl_ff_stage, indices_fuse_ff_stage = \
            intersect_and_cut_ff_stage([x_bl_ff_stage, y_bl_ff_stage],
                                       [x_c_fuse_ff, y_c_fuse_ff],
                                       list_vert_lines_ff_stage)
    except:
        try:
            y_mid_inlet = translate_points(y_mid_inlet,y_mid_inlet[0],y_mid_inlet[-1],y_mid_inlet[0]-0.05,y_mid_inlet[-1]+0.05)
            y_mid_nozzle = translate_points(y_mid_nozzle,y_mid_nozzle[0],y_mid_nozzle[-1],y_mid_nozzle[0]-0.05,y_mid_nozzle[-1]+0.05)    # cut all lines in pieces
            list_vert_lines_ff_stage = [[x_mid_inlet, y_mid_inlet],
                               [[i[0] for i in rotor_inlet], [i[1] for i in rotor_inlet]],
                               [[i[0] for i in rotor_outlet], [i[1] for i in rotor_outlet]],
                               [[i[0] for i in stator_inlet], [i[1] for i in stator_inlet]],
                               [[i[0] for i in stator_outlet], [i[1] for i in stator_outlet]],
                               [x_mid_nozzle, y_mid_nozzle]]

            lines_4, nacelle_bottom_x, nacelle_bottom_y, _, _, indices_nac_bottom, _ = \
                intersect_and_cut_ff_stage([nacelle_bottom_x, nacelle_bottom_y], [bl_nac_bottom_x, bl_nac_bottom_y],
                                           list_vert_lines_ff_stage)

            lines_3, bl_nac_bottom_x, bl_nac_bottom_y, _, _, indices_bl_nac_bottom, _ = intersect_and_cut_ff_stage(
                [bl_nac_bottom_x, bl_nac_bottom_y],
                [x_mid_ff_stage, y_mid_ff_stage],
                list_vert_lines_ff_stage)

            lines_2, _, _, x_mid_ff_stage, y_mid_ff_stage, _, indices_mid_ff_stage = intersect_and_cut_ff_stage(
                [x_bl_ff_stage, y_bl_ff_stage],
                [x_mid_ff_stage, y_mid_ff_stage],
                list_vert_lines_ff_stage)

            lines_1, x_bl_ff_stage, y_bl_ff_stage, x_c_fuse_ff, y_c_fuse_ff, indices_bl_ff_stage, indices_fuse_ff_stage = \
                intersect_and_cut_ff_stage([x_bl_ff_stage, y_bl_ff_stage],
                                           [x_c_fuse_ff, y_c_fuse_ff],
                                           list_vert_lines_ff_stage)
        except:
            try:
                y_mid_inlet = translate_points(y_mid_inlet,y_mid_inlet[0],y_mid_inlet[-1],y_mid_inlet[0]-0.1,y_mid_inlet[-1]+0.1)
                y_mid_nozzle = translate_points(y_mid_nozzle,y_mid_nozzle[0],y_mid_nozzle[-1],y_mid_nozzle[0]-0.1,y_mid_nozzle[-1]+0.1)
                list_vert_lines_ff_stage = [[x_mid_inlet, y_mid_inlet],
                                   [[i[0] for i in rotor_inlet], [i[1] for i in rotor_inlet]],
                                   [[i[0] for i in rotor_outlet], [i[1] for i in rotor_outlet]],
                                   [[i[0] for i in stator_inlet], [i[1] for i in stator_inlet]],
                                   [[i[0] for i in stator_outlet], [i[1] for i in stator_outlet]],
                                   [x_mid_nozzle, y_mid_nozzle]]

                lines_4, nacelle_bottom_x, nacelle_bottom_y, _, _, indices_nac_bottom, _ = \
                    intersect_and_cut_ff_stage([nacelle_bottom_x, nacelle_bottom_y], [bl_nac_bottom_x, bl_nac_bottom_y],
                                               list_vert_lines_ff_stage)

                lines_3, bl_nac_bottom_x, bl_nac_bottom_y, _, _, indices_bl_nac_bottom, _ = intersect_and_cut_ff_stage(
                    [bl_nac_bottom_x, bl_nac_bottom_y],
                    [x_mid_ff_stage, y_mid_ff_stage],
                    list_vert_lines_ff_stage)

                lines_2, _, _, x_mid_ff_stage, y_mid_ff_stage, _, indices_mid_ff_stage = intersect_and_cut_ff_stage(
                    [x_bl_ff_stage, y_bl_ff_stage],
                    [x_mid_ff_stage, y_mid_ff_stage],
                    list_vert_lines_ff_stage)

                lines_1, x_bl_ff_stage, y_bl_ff_stage, x_c_fuse_ff, y_c_fuse_ff, indices_bl_ff_stage, indices_fuse_ff_stage = \
                    intersect_and_cut_ff_stage([x_bl_ff_stage, y_bl_ff_stage],
                                               [x_c_fuse_ff, y_c_fuse_ff],
                                               list_vert_lines_ff_stage)
            except:
                y_mid_inlet = translate_points(y_mid_inlet, y_mid_inlet[0], y_mid_inlet[-1], y_mid_inlet[0] - 0.3,
                                               y_mid_inlet[-1] + 0.3)
                y_mid_nozzle = translate_points(y_mid_nozzle, y_mid_nozzle[0], y_mid_nozzle[-1], y_mid_nozzle[0] - 0.3,
                                                y_mid_nozzle[-1] + 0.3)
                list_vert_lines_ff_stage = [[x_mid_inlet, y_mid_inlet],
                                            [[i[0] for i in rotor_inlet], [i[1] for i in rotor_inlet]],
                                            [[i[0] for i in rotor_outlet], [i[1] for i in rotor_outlet]],
                                            [[i[0] for i in stator_inlet], [i[1] for i in stator_inlet]],
                                            [[i[0] for i in stator_outlet], [i[1] for i in stator_outlet]],
                                            [x_mid_nozzle, y_mid_nozzle]]

                lines_4, nacelle_bottom_x, nacelle_bottom_y, _, _, indices_nac_bottom, _ = \
                    intersect_and_cut_ff_stage([nacelle_bottom_x, nacelle_bottom_y], [bl_nac_bottom_x, bl_nac_bottom_y],
                                               list_vert_lines_ff_stage)

                lines_3, bl_nac_bottom_x, bl_nac_bottom_y, _, _, indices_bl_nac_bottom, _ = intersect_and_cut_ff_stage(
                    [bl_nac_bottom_x, bl_nac_bottom_y],
                    [x_mid_ff_stage, y_mid_ff_stage],
                    list_vert_lines_ff_stage)

                lines_2, _, _, x_mid_ff_stage, y_mid_ff_stage, _, indices_mid_ff_stage = intersect_and_cut_ff_stage(
                    [x_bl_ff_stage, y_bl_ff_stage],
                    [x_mid_ff_stage, y_mid_ff_stage],
                    list_vert_lines_ff_stage)

                lines_1, x_bl_ff_stage, y_bl_ff_stage, x_c_fuse_ff, y_c_fuse_ff, indices_bl_ff_stage, indices_fuse_ff_stage = \
                    intersect_and_cut_ff_stage([x_bl_ff_stage, y_bl_ff_stage],
                                               [x_c_fuse_ff, y_c_fuse_ff],
                                               list_vert_lines_ff_stage)

    h4_67_x, h4_67_y, h4_78_x, h4_78_y, h4_910_x, h4_910_y, h4_1112_x, \
    h4_1112_y, h4_1314_x, h4_1314_y, h4_1415_x, h4_1415_y = lines_4[0][0], lines_4[0][1], \
                                                                                  lines_4[1][0], lines_4[1][1], \
                                                                                  lines_4[2][0], lines_4[2][1], \
                                                                                  lines_4[3][0], lines_4[3][1], \
                                                                                  lines_4[4][0], lines_4[4][1], \
                                                                                  lines_4[5][0], lines_4[5][1]


    h3_67_x, h3_67_y, h3_78_x, h3_78_y, h3_910_x, h3_910_y, h3_1112_x, \
    h3_1112_y, h3_1314_x, h3_1314_y, h3_1415_x, h3_1415_y = lines_3[0][0], lines_3[0][1], \
                                                                                  lines_3[1][0], lines_3[1][1], \
                                                                                  lines_3[2][0], lines_3[2][1], \
                                                                                  lines_3[3][0], lines_3[3][1], \
                                                                                  lines_3[4][0], lines_3[4][1], \
                                                                                  lines_3[5][0], lines_3[5][1]


    h2_67_x, h2_67_y, h2_78_x, h2_78_y, h2_910_x, h2_910_y, h2_1112_x, \
    h2_1112_y, h2_1314_x, h2_1314_y, h2_1415_x, h2_1415_y = lines_2[0][0], lines_2[0][1], \
                                                                                  lines_2[1][0], lines_2[1][1], \
                                                                                  lines_2[2][0], lines_2[2][1], \
                                                                                  lines_2[3][0], lines_2[3][1], \
                                                                                  lines_2[4][0], lines_2[4][1], \
                                                                                  lines_2[5][0], lines_2[5][1]


    h1_67_x, h1_67_y, h1_78_x, h1_78_y, h1_910_x, h1_910_y, h1_1112_x, \
    h1_1112_y, h1_1314_x, h1_1314_y, h1_1415_x, h1_1415_y = lines_1[0][0], lines_1[0][1], \
                                                                                  lines_1[1][0], lines_1[1][1], \
                                                                                  lines_1[2][0], lines_1[2][1], \
                                                                                  lines_1[3][0], lines_1[3][1], \
                                                                                  lines_1[4][0], lines_1[4][1], \
                                                                                  lines_1[5][0], lines_1[5][1]

    # intersection of bl_nac_top and rear line
    nac_bl_top_inters = find_intersections(bl_nac_top_x, bl_nac_top_y, nacelle_te_x, nacelle_te_y)[0]
    bl_nac_top_x, bl_nac_top_y = insert_and_cut_array(bl_nac_top_x, bl_nac_top_y, nac_bl_top_inters)
    x_c6_16, y_c6_16 = insert_and_cut_array_reverse(x_c6_16, y_c6_16, nac_bl_top_inters)

    y_5_17 = y_c3_4[0]

    # intersection of bl_nac_bot and rear line
    nac_bl_bot_inters = find_intersections(bl_nac_bottom_x, bl_nac_bottom_y, nacelle_te_x, nacelle_te_y)[0]
    bl_nac_bottom_x, bl_nac_bottom_y = insert_and_cut_array(bl_nac_bottom_x, bl_nac_bottom_y, nac_bl_bot_inters)

    x_nac_te2 = x_8+np.cos(np.deg2rad(teta_te))*bl_height_nac
    y_nac_te2 = nacelle_bottom_y[-1]-np.sin(np.deg2rad(teta_te))*bl_height_nac

    h_up = bl_height_nac/np.cos(np.deg2rad(teta_te)+np.deg2rad(geo['beta_te_up']))
    x_nac_te1 = bl_nac_top_x[-1]+np.cos(np.deg2rad(geo['beta_te_up']))*h_up
    y_nac_te1 = bl_nac_top_y[-1]-np.sin(np.deg2rad(geo['beta_te_up']))*h_up

    h_low = bl_height_nac/np.cos(np.deg2rad(teta_te)+np.deg2rad(geo['beta_te_low']))
    x_nac_te3 = bl_nac_bottom_x[-1]+np.cos(np.deg2rad(geo['beta_te_low']))*h_low
    y_nac_te3 = bl_nac_bottom_y[-1]+np.sin(np.deg2rad(geo['beta_te_low']))*h_low

    # fuselage TE
    y_fuse_te = 0.
    x_fuse_te = l_fuse+2*bl_height_fuse

    # use indices on X[0]; points: 0 -> X
    idx_f0 = 0
    idx_f1 = idx_nose_half
    idx_f2 = idx_nose
    idx_f3 = idx_cent
    idx_f4 = idx_rear_half
    idx_f4_1 = -1
    for i in range(0,len(X[0])):
        if X[0][i] < x_c_fuse_ff[0]:
            idx_f4_1 += 1
        else:
            break
    idx_f15 = np.where(X[0] == x_8f)[0][0]
    idx_f16 = len(X[0])-1

    # use indices on x_c_fuse_ff: 10000 -> X
    point_no_f = 5000
    idx_f5 = 0
    idx_f6 = indices_fuse_ff_stage[0]
    idx_f7 = indices_fuse_ff_stage[1]
    # idx_f8 = indices_fuse_ff_stage[2]
    idx_f9 = indices_fuse_ff_stage[2]
    # idx_f10 = indices_fuse_ff_stage[4]
    idx_f11 = indices_fuse_ff_stage[3]
    # idx_f12 = indices_fuse_ff_stage[6]
    idx_f13 = indices_fuse_ff_stage[4]
    idx_f14 = indices_fuse_ff_stage[5]
    idx_f15_1 = len(x_c_fuse_ff)-1

    # use indices on x_mid_ff_stage: 12000 -> X
    idx_m7 = indices_mid_ff_stage[1]
    # idx_m8 = indices_mid_ff_stage[2]
    idx_m9 = indices_mid_ff_stage[2]
    # idx_m10 = indices_mid_ff_stage[4]
    idx_m11 = indices_mid_ff_stage[3]
    # idx_m12 = indices_mid_ff_stage[6]
    idx_m13 = indices_mid_ff_stage[4]
    idx_m16 = len(x_c2_16)

    # use indices on fuse BL in ff stage: 8000 -> X
    idx_bf5 = 0
    idx_bf6 = indices_bl_ff_stage[0]
    idx_bf7 = indices_bl_ff_stage[1]
    # idx_bf8 = indices_bl_ff_stage[2]
    idx_bf9 = indices_bl_ff_stage[2]
    # idx_bf10 = indices_bl_ff_stage[4]
    idx_bf11 = indices_bl_ff_stage[3]
    # idx_bf12 = indices_bl_ff_stage[6]
    idx_bf13 = indices_bl_ff_stage[4]
    idx_bf14 = indices_bl_ff_stage[5]
    idx_bf15 = len(x_bl_ff_stage)-1

    # use indices on nacelle bottom BL in ff stage: 9000 -> X
    idx_bnb5 = 0
    idx_bnb6 = indices_bl_nac_bottom[0]
    idx_bnb7 = indices_bl_nac_bottom[1]
    # idx_bnb8 = indices_bl_nac_bottom[2]
    idx_bnb9 = indices_bl_nac_bottom[2]
    # idx_bnb10 = indices_bl_nac_bottom[4]
    idx_bnb11 = indices_bl_nac_bottom[3]
    # idx_bnb12 = indices_bl_nac_bottom[6]
    idx_bnb13 = indices_bl_nac_bottom[4]
    idx_bnb14 = indices_bl_nac_bottom[5]
    idx_bnb15 = len(bl_nac_bottom_x)-1

    # use indices on nacelle bottom in ff stage: 6000 -> X
    idx_nb5 = 0
    idx_nb6 = indices_nac_bottom[0]
    idx_nb7 = indices_nac_bottom[1]
    # idx_nb8 = indices_nac_bottom[2]
    idx_nb9 = indices_nac_bottom[2]
    # idx_nb10 = indices_nac_bottom[4]
    idx_nb11 = indices_nac_bottom[3]
    # idx_nb12 = indices_nac_bottom[6]
    idx_nb13 = indices_nac_bottom[4]
    idx_nb14 = indices_nac_bottom[5]
    idx_nb15 = len(nacelle_bottom_x)-1

    # use indices on nacelle top BL: 11000 -> X
    idx_bnt50 = 0
    idx_bnt51 = indices_bl_nac_top[0]
    idx_bnt52 = indices_bl_nac_top[1]
    idx_bnt53 = indices_bl_nac_top[2]
    idx_bnt54 = len(bl_nac_top_x)-1

    # use indices on nacelle top: 5000 -> X
    idx_nt50 = 0
    idx_nt51 = indices_nac_top[0]
    idx_nt52 = indices_nac_top[1]
    idx_nt53 = indices_nac_top[2]
    idx_nt54 = len(nacelle_top_x)-1

    # use indices for fuselage BL
    idx_fbl1 = len(x_c0_1)
    idx_fbl2 = len(x_c0_2)
    idx_fbl3 = len(x_c0_3)
    idx_fbl4 = len(x_c0_4)
    idx_fbl5 = len(x_c0_5)
    idx_fbl16 = len(x_c1_16)
    idx_fbl7 = len(x_c0_7)
    idx_fbl_int = int(np.where(x_c0_7 == np.array(x_int))[0])

    # indices for vertical lines in FF stage
    idx_67_1 = len(h1_67_x)
    # idx_67_2 = len(h2_67_x)
    # idx_67_3 = len(h3_67_x)
    idx_67_4 = len(h4_67_x)

    idx_78_1 = len(h1_78_x)
    idx_78_2 = len(h2_78_x)
    idx_78_3 = len(h3_78_x)
    idx_78_4 = len(h4_78_x)

    # idx_89_1 = len(h1_89_x)
    # idx_89_2 = len(h2_89_x)
    # idx_89_3 = len(h3_89_x)
    # idx_89_4 = len(h4_89_x)

    idx_910_1 = len(h1_910_x)
    idx_910_2 = len(h2_910_x)
    idx_910_3 = len(h3_910_x)
    idx_910_4 = len(h4_910_x)

    # idx_1011_1 = len(h1_1011_x)
    # idx_1011_2 = len(h2_1011_x)
    # idx_1011_3 = len(h3_1011_x)
    # idx_1011_4 = len(h4_1011_x)

    idx_1112_1 = len(h1_1112_x)
    idx_1112_2 = len(h2_1112_x)
    idx_1112_3 = len(h3_1112_x)
    idx_1112_4 = len(h4_1112_x)

    # idx_1213_1 = len(h1_1213_x)
    # idx_1213_2 = len(h2_1213_x)
    # idx_1213_3 = len(h3_1213_x)
    # idx_1213_4 = len(h4_1213_x)

    idx_1314_1 = len(h1_1314_x)
    idx_1314_2 = len(h2_1314_x)
    idx_1314_3 = len(h3_1314_x)
    idx_1314_4 = len(h4_1314_x)

    idx_1415_1 = len(h1_1415_x)
    # idx_1415_2 = len(h2_1415_x)
    # idx_1415_3 = len(h3_1415_x)
    idx_1415_4 = len(h4_1415_x)

    idx_bnt16 = len(x_c6_16)
    idx_bnb16 = len(x_c5_16)
    # idx_d16 = len(x_c4_16)

    idx_f17 = idx_fbl7 + idx_f2
    # idx_i5 = len(x_c3_5)
    # idx_i16 = len(x_c3_16)

    n_h_nac_1 = rel_value(n_h_nac_1_orig, n_nac_orig, 0.14753596689888937, l_nac_orig, n_nac_orig, (x_nac_max-x_0)/2, l_nac)
    n_h_nac_3 = rel_value(n_h_nac_3_orig, n_nac_orig, 0.5587777308362689, l_nac_orig, n_nac_orig, (x_back_sep-x_nac_max)/2, l_nac)
    n_h_nose_1 = rel_value(n_h_nose_1_orig, n_fuse_front_orig, 2.9987182182182184, l_fuse_front_orig, n_fuse_front_tot, x_c0_1[-1], x_c0_5[-1])

    n_h_inlet_fus = rel_value(n_h_inlet_orig, n_ff_orig, 2*0.48010219714285896, l_ff_stage_orig, n_ff_orig, (x_2_f-x_0), x_8-x_0)
    n_h_inlet_nac = rel_value(n_h_inlet_orig, n_ff_orig, 1.0474007175483635, l_ff_stage_orig, n_ff_orig, (x_2_n-x_0), x_8-x_0)
    n_h_rot_fus = rel_value(n_h_rot_orig, n_ff_orig, 2*0.1526417906451627, l_ff_stage_orig, n_ff_orig, (x_22_f-x_2_f), x_8-x_0)
    n_h_nozzle_fus = rel_value(n_h_nozzle_orig, n_ff_orig, 2*0.44600119940686866, l_ff_stage_orig, n_ff_orig, (x_8f-x_3_f), x_8-x_0)
    n_h_nozzle_nac = rel_value(n_h_nozzle_orig, n_ff_orig, 0.7646724886990555, l_ff_stage_orig, n_ff_orig, (x_8f-x_3_n), x_8-x_0)
    n_h_rear_1 = rel_value(n_h_rear_1_orig, n_rear_domain_orig, 2*6.72442, 2*l_rear_domain_orig,
                           n_rear_domain_tot, x_rear_half-l_fuse, x_rear_back-l_fuse)
    n_h_rear_2 = rel_value(n_h_rear_2_orig, n_rear_domain_orig, 2*60.51983, 2*l_rear_domain_orig,
                           n_rear_domain_tot, x_rear_back-x_rear_half, x_rear_back-l_fuse)
    n_h_cent_1 = rel_value(n_h_cent_1_orig, n_fuse_front_orig, 20.428656406406404, l_fuse_front_orig,
                           n_fuse_front_tot, (geo['l_cent_f']-x_c0_2[-1])/2, x_c0_5[-1])

    # horizontal progressions
    # starting from front part of rotor stage
    # fuselage side
    p_h_rot = 1
    p_h_gap = 1
    p_h_stat = 1
    # p_v_duct = 1

    l_h_rot_fus = (x_22_f-x_2_f)  # approximate length of rotor stage
    l_h_gap_fus = (x_23_f-x_22_f)  # approximate length of gap
    l_h_stat_fus = (x_3_f-x_23_f)  # approximate length of stator
    l_h_rot_nac = (x_22_n-x_2_n)  # approximate length of rotor stage
    l_h_gap_nac = (x_23_n-x_22_n)  # approximate length of gap
    l_h_stat_nac = (x_3_n-x_23_n)  # approximate length of stator

    l_cell_rot_fus = l_h_rot_fus/n_h_rot_fus

    n_h_rot_nac = round(l_h_rot_nac/l_cell_rot_fus)
    n_h_gap_nac = round(l_h_gap_nac/l_cell_rot_fus)
    n_h_stat_nac = round(l_h_stat_nac/l_cell_rot_fus)
    n_h_gap_fus = round(l_h_gap_fus/l_cell_rot_fus)
    n_h_stat_fus = round(l_h_stat_fus/l_cell_rot_fus)

    l_h_nozzle_half_aprox_fus = (x_8f - x_3_f) / 2  # approximate length of nozzle
    p_h_ff_9_fus = calc_ratio_from_cellno_and_first_cell(l_cell_rot_fus, round(n_h_nozzle_fus/2), l_h_nozzle_half_aprox_fus, 1.01)
    h_ff_9_last_cell_height_fus = calc_last_cell_height(l_cell_rot_fus, p_h_ff_9_fus, round(n_h_nozzle_fus/2) - 1)
    p_h_ff_10_fus, n_h_ff_10_fus = calc_ratio_and_cellno_from_first_and_last_cell(h_ff_9_last_cell_height_fus,
                                                                        nacelle_te_cell_width, l_h_nozzle_half_aprox_fus)

    l_h_nozzle_half_aprox_nac = (x_8f - x_3_n) / 2  # approximate length of nozzle
    p_h_ff_9_nac = calc_ratio_from_cellno_and_first_cell(l_cell_rot_fus, round(n_h_nozzle_nac/2), l_h_nozzle_half_aprox_nac, 1.01)
    h_ff_9_last_cell_height_nac = calc_last_cell_height(l_cell_rot_fus, p_h_ff_9_nac, round(n_h_nozzle_nac/2) - 1)
    p_h_ff_10_nac, n_h_ff_10_nac = calc_ratio_and_cellno_from_first_and_last_cell(h_ff_9_last_cell_height_nac,
                                                                        nacelle_te_cell_width, l_h_nozzle_half_aprox_nac)

    l_h_inlet_half_aprox_fus = (x_2_f-x_0)/2   # approximate length of nacelle inlet stage
    n_h_ff_2_fus = round(n_h_inlet_fus/2)
    p_h_ff_2_fus = calc_ratio_from_cellno_and_first_cell(l_cell_rot_fus, n_h_ff_2_fus, l_h_inlet_half_aprox_fus, 1.01)
    h_ff_2_last_cell_height_fus = calc_last_cell_height(l_cell_rot_fus, p_h_rot, n_h_rot_fus-1)
    p_h_ff_1_fus, n_h_ff_1_fus = calc_ratio_and_cellno_from_first_and_last_cell(h_ff_2_last_cell_height_fus,
                                                                        nacelle_highlight_width, l_h_inlet_half_aprox_fus)

    l_h_inlet_half_aprox_nac = (x_2_n-x_0)/2   # approximate length of nacelle inlet stage
    n_h_ff_2_nac = round(n_h_inlet_nac /2)
    p_h_ff_2_nac = calc_ratio_from_cellno_and_first_cell(l_cell_rot_fus, n_h_ff_2_nac, l_h_inlet_half_aprox_nac, 1.01)
    h_ff_2_last_cell_height_nac = calc_last_cell_height(l_cell_rot_fus, p_h_ff_2_nac, n_h_ff_2_nac-1)
    p_h_ff_1_nac, n_h_ff_1_nac = calc_ratio_and_cellno_from_first_and_last_cell(h_ff_2_last_cell_height_nac ,
                                                                        nacelle_le_cell_width, l_h_inlet_half_aprox_nac )
    bl_nac_last_cell_height = calc_last_cell_height(bl_l_first_nac, bl_ratio, no_bl_cells_nac-1)
    bl_fus_last_cell_height = calc_last_cell_height(bl_l_first_fuse, bl_ratio, no_bl_cells_fuse-1)
    p_v_duct_rot_in, n_v_duct_rot_in = calc_ratio_and_cellno_from_first_and_last_cell(bl_nac_last_cell_height,
                                                                        bl_fus_last_cell_height, rotor_inlet[-1][1]-rotor_inlet[0][1])
    p_v_duct_rot_out, n_v_duct_rot_out = calc_ratio_and_cellno_from_first_and_last_cell(bl_nac_last_cell_height,
                                                                        bl_fus_last_cell_height, rotor_outlet[-1][1]-rotor_outlet[0][1])
    p_v_duct_stat_in, n_v_duct_stat_in = calc_ratio_and_cellno_from_first_and_last_cell(bl_nac_last_cell_height,
                                                                        bl_fus_last_cell_height, stator_inlet[-1][1]-stator_inlet[0][1])
    p_v_duct_stat_out, n_v_duct_stat_out = calc_ratio_and_cellno_from_first_and_last_cell(bl_nac_last_cell_height,
                                                                        bl_fus_last_cell_height, stator_outlet[-1][1]-stator_outlet[0][1])

    l_h_tail = l_fuse-x_8f
    p_h_tail, n_h_tail = calc_ratio_and_cellno_from_first_and_last_cell(nacelle_te_cell_width, fuse_te_width, l_h_tail)

    p_h_rear_1 = calc_ratio_from_cellno_and_first_cell(fuse_te_width, n_h_rear_1, x_rear_half-l_fuse, 1.01)
    h_rear_1_last_cell_width = calc_last_cell_height(fuse_te_width, p_h_rear_1, n_h_rear_1-1)

    n_v_low = round((y_5_17 - y_0_17) / h_rear_1_last_cell_width)
    p_v_low = 1

    n_v_up = n_v_tot-n_v_low

    n_h_fuse_inner_back = round((x_rear_half-x_c3_4[1])/ h_rear_1_last_cell_width)
    p_h_fuse_inner_back = 1
    n_h_fuse_inner_cent = round((1-interface_loc)*geo['l_cent_f']/h_rear_1_last_cell_width)
    p_h_fuse_inner_cent = 1

    # calculate progressions
    # vertical progression, based on cell heights of first BL cells
    p_v_up = calc_ratio_from_cellno_and_first_cell(h_rear_1_last_cell_width, n_v_up, hdomain-h_duct_in-r_fuse, 1.01)
    p_h_rear_2 = calc_ratio_from_cellno_and_first_cell(h_rear_1_last_cell_width, n_h_rear_2, ldomain+l_fuse-x_rear_half, 1.01)

    l_nose_front_fuse = calculate_arc_length(X[0][0:idx_nose_half],Y[0][0:idx_nose_half])
    l_nose_front_inter = calculate_arc_length(x_c3_1,y_c3_1)
    l_nose_rear_fuse = calculate_arc_length(X[0][idx_nose_half:idx_nose],Y[0][idx_nose_half:idx_nose])
    l_nose_rear_inter = calculate_arc_length(x_c3_2,y_c3_2)

    p_h_nose_1 = calc_ratio_from_cellno_and_first_cell(fuse_nose_le_cell_width, n_h_nose_1, l_nose_front_fuse, 1.01)
    mid_nose_cell_width_fuse = calc_last_cell_height(fuse_nose_le_cell_width, p_h_nose_1, n_h_nose_1-1)
    p_h_nose_2, n_h_nose_2 = calc_ratio_and_cellno_from_first_and_last_cell(mid_nose_cell_width_fuse, fuse_nose_te_cell_width,
                                                                            l_nose_rear_fuse)

    p_h_nose_12 = calc_ratio_from_cellno_and_first_cell(fuse_nose_le_cell_width, n_h_nose_1, l_nose_front_inter, 1.01)
    mid_nose_cell_width_inter = calc_last_cell_height(fuse_nose_le_cell_width, p_h_nose_12, n_h_nose_1-1)
    p_h_nose_22 = calc_ratio_from_cellno_and_first_cell(mid_nose_cell_width_inter,n_h_nose_2,l_nose_rear_inter,1.01)

    l_inl_2 = (x_0-geo['l_cent_f'])/2
    p_h_inl_2, n_h_inl_2 = calc_ratio_and_cellno_from_first_and_last_cell(mid_inlet_width, nacelle_highlight_width,
                                                                            l_inl_2)
    p_h_inl_1, n_h_inl_1 = calc_ratio_and_cellno_from_first_and_last_cell(fuse_nose_te_cell_width, mid_inlet_width,
                                                                            l_inl_2)

    l_cent = (1-interface_loc)*geo['l_cent_f']
    p_h_cent_2, n_h_cent_2 = calc_ratio_and_cellno_from_first_and_last_cell(interface_cell_width, fuse_nose_te_cell_width,
                                                                            l_cent)

    l_nac1 = (x_nac_max-x_0)/2
    p_h_nac_1 = calc_ratio_from_cellno_and_first_cell(nacelle_le_cell_width, n_h_nac_1, l_nac1, 1.01)
    mid_nac_max_cell_width = calc_last_cell_height(nacelle_le_cell_width, p_h_nac_1, n_h_nac_1-1)
    p_h_nac_2, n_h_nac_2 = calc_ratio_and_cellno_from_first_and_last_cell(mid_nac_max_cell_width, nacelle_max_cell_width,
                                                                            l_nac1)

    l_nac3 = (x_8-x_nac_max)/2
    p_h_nac_3 = calc_ratio_from_cellno_and_first_cell(nacelle_max_cell_width, n_h_nac_3, l_nac3, 1.01)
    mid_nac_rear_cell_width = calc_last_cell_height(nacelle_max_cell_width, p_h_nac_3, n_h_nac_3-1)
    p_h_nac_4, n_h_nac_4 = calc_ratio_and_cellno_from_first_and_last_cell(mid_nac_rear_cell_width, nacelle_te_cell_width,
                                                                            l_nac3)

    n_h_inner = int(round(0.6*(n_h_inl_1 + n_h_inl_2 + n_h_nac_1 + n_h_nac_2 + n_h_nac_3 + n_h_nac_4 + n_h_tail + n_h_rear_1)))
    p_h_inner = calc_ratio_from_cellno_and_first_cell(fuse_nose_te_cell_width, n_h_inner, x_rear_half-geo['l_cent_f'], 1.01)

    m1_first_cell_height = calc_last_cell_height(bl_l_first_fuse, bl_ratio, no_bl_cells_fuse-2)
    p_v_front_1 = calc_ratio_from_cellno_and_first_cell(m1_first_cell_height, n_v_front_1, -x_c3_1[0], 1.01)
    fuselage_le_cell_width = calc_last_cell_height(m1_first_cell_height, p_v_front_1, n_v_front_1-1)
    p_v_front_2 = calc_ratio_from_cellno_and_first_cell(fuselage_le_cell_width, n_v_front_2, ldomain/2+x_c3_1[0], 1.01)

    p_h_nose_1_inner, n_h_nose_1_inner = calc_ratio_and_cellno_from_first_and_last_cell(fuselage_le_cell_width, 2*mid_nose_cell_width_fuse,
                                                                            l_nose_front_inter)
    p_h_nose_2_inner, n_h_nose_2_inner = calc_ratio_and_cellno_from_first_and_last_cell(h_rear_1_last_cell_width, 2*mid_nose_cell_width_fuse,
                                                                            l_nose_rear_inter)

    n_v_up_mid = round(0.6*n_v_up)
    p_v_up_mid = calc_ratio_from_cellno_and_first_cell(h_rear_1_last_cell_width, n_v_up_mid, y_6_17-y_5_17, 1.01)

    last_cell_mid = calc_last_cell_height(h_rear_1_last_cell_width, p_v_up_mid, n_v_up_mid-1)

    ####
    p_v_int_mid, n_v_int_mid = calc_ratio_and_cellno_from_first_and_last_cell(interface_cell_width, h_rear_1_last_cell_width,
                                                                            y_c3_3[0]-y_c1_7[0])
    p_v_int_up, n_v_int_up = calc_ratio_and_cellno_from_first_and_last_cell(h_rear_1_last_cell_width, last_cell_mid,
                                                                            hdomain-y_c3_3[0])

    n_rear_domain = (x_rear_back-x_rear_half)/last_cell_mid
    p_rear_domain = 1

    last_cell_rear = calc_last_cell_height(h_rear_1_last_cell_width, p_h_rear_2, n_h_rear_2-1)

    n_rear_low = round(y_5_17/last_cell_rear)
    p_rear_low = 1

    p_rear_up, n_rear_up = calc_ratio_and_cellno_from_first_and_last_cell(last_cell_rear, last_cell_mid, y_6_17-y_5_17)

    p_cent_up = 1
    n_cent_up = round((x_rear_half-interface_loc*geo['l_cent_f'])/last_cell_mid)

    last_cell_front = calc_last_cell_height(fuselage_le_cell_width, p_v_front_2, n_v_front_2-1)
    l_nose_outer = calculate_arc_length(x_c4_1,y_c4_1)+calculate_arc_length(x_c4_2,y_c4_2)
    p_h_nose_up, n_h_nose_up = calc_ratio_and_cellno_from_first_and_last_cell(last_cell_front, last_cell_mid, l_nose_outer)

    if geotype == 'pfc':
        defaultfilepath = 'default_files/gmsh_default_pfc_hybrid.geo'
    rel_path = os.path.abspath(__file__)

    with open(f'{casepath}//gmsh_coords.geo', 'w') as f, open(os.path.join(os.path.dirname(rel_path), defaultfilepath), 'r') as geo:
        f.write('//fuselage\n')
        for i in range(0,len(X[0])):
            f.write('Point('+str(i)+') = {'+str(X[0][i])+','+str(Y[0][i])+',0};\n')
        f.write('//fuselage FF stage\n')
        for i in range(0,len(x_c_fuse_ff)):
            f.write('Point('+str(i+10000)+') = {'+str(x_c_fuse_ff[i])+','+str(y_c_fuse_ff[i])+',0};\n')
        f.write('//nacelle top\n')
        for i in range(0,len(nacelle_top_x)):
            f.write('Point('+str(i+5000)+') = {'+str(nacelle_top_x[i])+','+str(nacelle_top_y[i])+',0};\n')
        f.write('//nacelle bottom\n')
        for i in range(0,len(nacelle_bottom_x)):
            f.write('Point('+str(i+6000)+') = {'+str(nacelle_bottom_x[i])+','+str(nacelle_bottom_y[i])+',0};\n')
        f.write('//FF stage middle line\n')
        for i in range(0,len(x_mid_ff_stage)):
            f.write('Point('+str(i+12000)+') = {'+str(x_mid_ff_stage[i])+','+str(y_mid_ff_stage[i])+',0};\n')
        f.write('//FF stage fuse BL\n')
        for i in range(0,len(x_bl_ff_stage)):
            f.write('Point('+str(i+8000)+') = {'+str(x_bl_ff_stage[i])+','+str(y_bl_ff_stage[i])+',0};\n')
        f.write('//FF stage nacelle BL\n')
        for i in range(0,len(bl_nac_bottom_x)):
            f.write('Point('+str(i+9000)+') = {'+str(bl_nac_bottom_x[i])+','+str(bl_nac_bottom_y[i])+',0};\n')
        f.write('//upper nacelle BL\n')
        for i in range(0,len(bl_nac_top_x)):
            f.write('Point('+str(i+11000)+') = {'+str(bl_nac_top_x[i])+','+str(bl_nac_top_y[i])+',0};\n')
        f.write('//FF stage inlet mid line\n')
        for i in range(0,len(h1_67_x)):
            f.write('Point('+str(i+20000)+') = {'+str(h1_67_x[i])+','+str(h1_67_y[i])+',0};\n')
        for i in range(0,len(h2_67_x)):
            f.write('Point('+str(i+20100)+') = {'+str(h2_67_x[i])+','+str(h2_67_y[i])+',0};\n')
        for i in range(0,len(h3_67_x)):
            f.write('Point('+str(i+20200)+') = {'+str(h3_67_x[i])+','+str(h3_67_y[i])+',0};\n')
        for i in range(0,len(h4_67_x)):
            f.write('Point('+str(i+20300)+') = {'+str(h4_67_x[i])+','+str(h4_67_y[i])+',0};\n')
        f.write('//FF stage rotor inlet\n')
        for i in range(0,len(h1_78_x)):
            f.write('Point('+str(i+21000)+') = {'+str(h1_78_x[i])+','+str(h1_78_y[i])+',0};\n')
        for i in range(0,len(h2_78_x)):
            f.write('Point('+str(i+21100)+') = {'+str(h2_78_x[i])+','+str(h2_78_y[i])+',0};\n')
        for i in range(0,len(h3_78_x)):
            f.write('Point('+str(i+21200)+') = {'+str(h3_78_x[i])+','+str(h3_78_y[i])+',0};\n')
        for i in range(0,len(h4_78_x)):
            f.write('Point('+str(i+21300)+') = {'+str(h4_78_x[i])+','+str(h4_78_y[i])+',0};\n')
        # f.write('//FF stage rotor mid line\n')
        # for i in range(0,len(h1_89_x)):
        #     f.write('Point('+str(i+22000)+') = {'+str(h1_89_x[i])+','+str(h1_89_y[i])+',0};\n')
        # for i in range(0,len(h2_89_x)):
        #     f.write('Point('+str(i+22100)+') = {'+str(h2_89_x[i])+','+str(h2_89_y[i])+',0};\n')
        # for i in range(0,len(h3_89_x)):
        #     f.write('Point('+str(i+22200)+') = {'+str(h3_89_x[i])+','+str(h3_89_y[i])+',0};\n')
        # for i in range(0,len(h4_89_x)):
        #     f.write('Point('+str(i+22300)+') = {'+str(h4_89_x[i])+','+str(h4_89_y[i])+',0};\n')
        f.write('//FF stage rotor outlet\n')
        for i in range(0,len(h1_910_x)):
            f.write('Point('+str(i+23000)+') = {'+str(h1_910_x[i])+','+str(h1_910_y[i])+',0};\n')
        for i in range(0,len(h2_910_x)):
            f.write('Point('+str(i+23100)+') = {'+str(h2_910_x[i])+','+str(h2_910_y[i])+',0};\n')
        for i in range(0,len(h3_910_x)):
            f.write('Point('+str(i+23200)+') = {'+str(h3_910_x[i])+','+str(h3_910_y[i])+',0};\n')
        for i in range(0,len(h4_910_x)):
            f.write('Point('+str(i+23300)+') = {'+str(h4_910_x[i])+','+str(h4_910_y[i])+',0};\n')
        # f.write('//FF stage gap mid line\n')
        # for i in range(0,len(h1_1011_x)):
        #     f.write('Point('+str(i+24000)+') = {'+str(h1_1011_x[i])+','+str(h1_1011_y[i])+',0};\n')
        # for i in range(0,len(h2_1011_x)):
        #     f.write('Point('+str(i+24100)+') = {'+str(h2_1011_x[i])+','+str(h2_1011_y[i])+',0};\n')
        # for i in range(0,len(h3_1011_x)):
        #     f.write('Point('+str(i+24200)+') = {'+str(h3_1011_x[i])+','+str(h3_1011_y[i])+',0};\n')
        # for i in range(0,len(h4_1011_x)):
        #     f.write('Point('+str(i+24300)+') = {'+str(h4_1011_x[i])+','+str(h4_1011_y[i])+',0};\n')
        f.write('//FF stage stator inlet\n')
        for i in range(0,len(h1_1112_x)):
            f.write('Point('+str(i+25000)+') = {'+str(h1_1112_x[i])+','+str(h1_1112_y[i])+',0};\n')
        for i in range(0,len(h2_1112_x)):
            f.write('Point('+str(i+25100)+') = {'+str(h2_1112_x[i])+','+str(h2_1112_y[i])+',0};\n')
        for i in range(0,len(h3_1112_x)):
            f.write('Point('+str(i+25200)+') = {'+str(h3_1112_x[i])+','+str(h3_1112_y[i])+',0};\n')
        for i in range(0,len(h4_1112_x)):
            f.write('Point('+str(i+25300)+') = {'+str(h4_1112_x[i])+','+str(h4_1112_y[i])+',0};\n')
        # f.write('//FF stage stator mid line\n')
        # for i in range(0,len(h1_1213_x)):
        #     f.write('Point('+str(i+26000)+') = {'+str(h1_1213_x[i])+','+str(h1_1213_y[i])+',0};\n')
        # for i in range(0,len(h2_1213_x)):
        #     f.write('Point('+str(i+26100)+') = {'+str(h2_1213_x[i])+','+str(h2_1213_y[i])+',0};\n')
        # for i in range(0,len(h3_1213_x)):
        #     f.write('Point('+str(i+26200)+') = {'+str(h3_1213_x[i])+','+str(h3_1213_y[i])+',0};\n')
        # for i in range(0,len(h4_1213_x)):
        #     f.write('Point('+str(i+26300)+') = {'+str(h4_1213_x[i])+','+str(h4_1213_y[i])+',0};\n')
        f.write('//FF stage stator outlet\n')
        for i in range(0,len(h1_1314_x)):
            f.write('Point('+str(i+27000)+') = {'+str(h1_1314_x[i])+','+str(h1_1314_y[i])+',0};\n')
        for i in range(0,len(h2_1314_x)):
            f.write('Point('+str(i+27100)+') = {'+str(h2_1314_x[i])+','+str(h2_1314_y[i])+',0};\n')
        for i in range(0,len(h3_1314_x)):
            f.write('Point('+str(i+27200)+') = {'+str(h3_1314_x[i])+','+str(h3_1314_y[i])+',0};\n')
        for i in range(0,len(h4_1314_x)):
            f.write('Point('+str(i+27300)+') = {'+str(h4_1314_x[i])+','+str(h4_1314_y[i])+',0};\n')
        f.write('//FF stage nozzle mid line\n')
        for i in range(0,len(h1_1415_x)):
            f.write('Point('+str(i+28000)+') = {'+str(h1_1415_x[i])+','+str(h1_1415_y[i])+',0};\n')
        for i in range(0,len(h2_1415_x)):
            f.write('Point('+str(i+28100)+') = {'+str(h2_1415_x[i])+','+str(h2_1415_y[i])+',0};\n')
        for i in range(0,len(h3_1415_x)):
            f.write('Point('+str(i+28200)+') = {'+str(h3_1415_x[i])+','+str(h3_1415_y[i])+',0};\n')
        for i in range(0,len(h4_1415_x)):
            f.write('Point('+str(i+28300)+') = {'+str(h4_1415_x[i])+','+str(h4_1415_y[i])+',0};\n')

        f.write('//Fuselage BL\n')
        for i in range(0,len(x_c1_3)):
            f.write('Point('+str(i+32000)+') = {'+str(x_c1_3[i])+','+str(y_c1_3[i])+',0};\n')
        for i in range(0,len(x_c1_7)):
            f.write('Point('+str(i+32500)+') = {'+str(x_c1_7[i])+','+str(y_c1_7[i])+',0};\n')
        for i in range(0,len(x_c1_4)):
            f.write('Point('+str(i+33000)+') = {'+str(x_c1_4[i])+','+str(y_c1_4[i])+',0};\n')
        for i in range(0,len(x_c1_5)):
            f.write('Point('+str(i+34000)+') = {'+str(x_c1_5[i])+','+str(y_c1_5[i])+',0};\n')
        for i in range(0,len(x_c1_16)):
            f.write('Point('+str(i+35000)+') = {'+str(x_c1_16[i])+','+str(y_c1_16[i])+',0};\n')

        f.write('//Fuselage intermediate line\n')
        for i in range(0,len(x_c3_3)):
            f.write('Point('+str(i+52000)+') = {'+str(x_c3_3[i])+','+str(y_c3_3[i])+',0};\n')
        for i in range(0,len(x_c3_7)):
            f.write('Point('+str(i+52500)+') = {'+str(x_c3_7[i])+','+str(y_c3_7[i])+',0};\n')
        for i in range(0,2):
            f.write('Point('+str(i+53000)+') = {'+str(x_c3_4[i])+','+str(y_c3_4[i])+',0};\n')
        for i in range(len(x_c3_5)-1,len(x_c3_5)):
            f.write('Point('+str(i+54000)+') = {'+str(x_c3_5[i])+','+str(y_c3_5[i])+',0};\n')
        for i in range(0,2):
            f.write('Point('+str(i+55000)+') = {'+str(x_c3_16[i])+','+str(y_c3_16[i])+',0};\n')

        f.write('//Nacelle TE BL point\n')
        f.write('Point(' + str(101000) + ') = {' + str(x_nac_te1) + ',' + str(y_nac_te1) + ',0};\n')
        f.write('Point(' + str(101001) + ') = {' + str(x_nac_te2) + ',' + str(y_nac_te2) + ',0};\n')
        f.write('Point(' + str(101002) + ') = {' + str(x_nac_te3) + ',' + str(y_nac_te3) + ',0};\n')

        f.write('//Fuselage TE BL point\n')
        f.write('Point(' + str(101003) + ') = {' + str(x_fuse_te) + ',' + str(y_fuse_te) + ',0};\n')

        f.write('//Domain\n')
        f.write('Point('+str(62000)+') = {'+str(X[0][idx_int])+','+str(y_c4_3[0])+',0};\n')
        for i in range(0,len(x_c4_7)):
            f.write('Point('+str(i+62500)+') = {'+str(x_c4_7[i])+','+str(y_c4_7[i])+',0};\n')
        for i in range(0,len(x_c4_4)):
            f.write('Point('+str(i+63000)+') = {'+str(x_c4_4[i])+','+str(y_c4_4[i])+',0};\n')
        for i in range(0,len(x_c4_5)):
            f.write('Point('+str(i+64000)+') = {'+str(x_c4_5[i])+','+str(y_c4_5[i])+',0};\n')
        for i in range(0,len(x_c4_16)):
            f.write('Point('+str(i+65000)+') = {'+str(x_c4_16[i])+','+str(y_c4_16[i])+',0};\n')

        f.write('Point(' + str(66000) + ') = {' + str(x_rear_half) + ',' + str(y_0_17) + ',0};\n')
        f.write('Point(' + str(66001) + ') = {' + str(x_rear_back) + ',' + str(y_0_17) + ',0};\n')
        f.write('Point(' + str(66002) + ') = {' + str(x_rear_half) + ',' + str(y_0_17) + ',0};\n')
        f.write('Point(' + str(66003) + ') = {' + str(x_rear_back) + ',' + str(y_1_17) + ',0};\n')
        f.write('Point(' + str(66004) + ') = {' + str(x_rear_half) + ',' + str(y_2_17) + ',0};\n')
        f.write('Point(' + str(66005) + ') = {' + str(x_rear_back) + ',' + str(y_2_17) + ',0};\n')
        f.write('Point(' + str(66006) + ') = {' + str(x_rear_half) + ',' + str(y_3_17) + ',0};\n')
        f.write('Point(' + str(66007) + ') = {' + str(x_rear_back) + ',' + str(y_3_17) + ',0};\n')
        f.write('Point(' + str(66008) + ') = {' + str(x_rear_half) + ',' + str(y_4_17) + ',0};\n')
        f.write('Point(' + str(66009) + ') = {' + str(x_rear_back) + ',' + str(y_4_17) + ',0};\n')
        f.write('Point(' + str(66010) + ') = {' + str(x_rear_half) + ',' + str(y_5_17) + ',0};\n')
        f.write('Point(' + str(66011) + ') = {' + str(x_rear_back) + ',' + str(y_5_17) + ',0};\n')
        f.write('Point(' + str(66012) + ') = {' + str(x_rear_half) + ',' + str(y_6_17) + ',0};\n')
        f.write('Point(' + str(66013) + ') = {' + str(x_rear_back) + ',' + str(y_6_17) + ',0};\n')
        f.write('Point(' + str(66014) + ') = {' + str(x_max_half[-1]) + ',' + str(y_6_17) + ',0};\n')
        f.write('Point(' + str(66015) + ') = {' + str(x_max[-1]) + ',' + str(y_6_17) + ',0};\n')
        f.write('Point(' + str(66016) + ') = {' + str(x_back[-1]) + ',' + str(y_6_17) + ',0};\n')
        f.write('Point(' + str(66017) + ') = {' + str(bl_nac_top_x[-1]) + ',' + str(y_6_17) + ',0};\n')

        # indices for fuselage
        f.write('idx_f0 = ' + str(idx_f0) + ';\n')
        f.write('idx_f1 = ' + str(idx_f1) + ';\n')
        f.write('idx_f2 = ' + str(idx_f2) + ';\n')
        f.write('idx_f3 = ' + str(idx_f3) + ';\n')
        f.write('idx_f4 = ' + str(idx_f4) + ';\n')
        f.write('idx_f4_1 = ' + str(idx_f4_1) + ';\n')
        f.write('idx_f15 = ' + str(idx_f15) + ';\n')
        f.write('idx_f16 = ' + str(idx_f16) + ';\n')
        f.write('idx_f17 = ' + str(idx_f17) + ';\n')
        f.write('idx_f_int = ' + str(idx_int) + ';\n')

        # indices for fuselage near fan stage
        f.write('idx_f5 = ' + str(idx_f5+10000) + ';\n')
        f.write('idx_f6 = ' + str(idx_f6+10000) + ';\n')
        f.write('idx_f7 = ' + str(idx_f7+10000) + ';\n')
        # f.write('idx_f8 = ' + str(idx_f8+10000) + ';\n')
        f.write('idx_f9 = ' + str(idx_f9+10000) + ';\n')
        # f.write('idx_f10 = ' + str(idx_f10+10000) + ';\n')
        f.write('idx_f11 = ' + str(idx_f11+10000) + ';\n')
        # f.write('idx_f12 = ' + str(idx_f12+10000) + ';\n')
        f.write('idx_f13 = ' + str(idx_f13+10000) + ';\n')
        f.write('idx_f14 = ' + str(idx_f14+10000) + ';\n')
        f.write('idx_f15_1 = ' + str(idx_f15_1+10000) + ';\n')

        # indices x_mid_ff_stage
        f.write('idx_m7 = ' + str(idx_m7+12000) + ';\n')
        # f.write('idx_m8 = ' + str(idx_m8+12000) + ';\n')
        f.write('idx_m9 = ' + str(idx_m9+12000) + ';\n')
        # f.write('idx_m10 = ' + str(idx_m10+12000) + ';\n')
        f.write('idx_m11 = ' + str(idx_m11+12000) + ';\n')
        # f.write('idx_m12 = ' + str(idx_m12+12000) + ';\n')
        f.write('idx_m13 = ' + str(idx_m13+12000) + ';\n')

        # indices fuse BL in ff stage
        f.write('idx_bf5 = ' + str(idx_bf5+8000) + ';\n')
        f.write('idx_bf6 = ' + str(idx_bf6+8000) + ';\n')
        f.write('idx_bf7 = ' + str(idx_bf7+8000) + ';\n')
        # f.write('idx_bf8 = ' + str(idx_bf8+8000) + ';\n')
        f.write('idx_bf9 = ' + str(idx_bf9+8000) + ';\n')
        # f.write('idx_bf10 = ' + str(idx_bf10+8000) + ';\n')
        f.write('idx_bf11 = ' + str(idx_bf11+8000) + ';\n')
        # f.write('idx_bf12 = ' + str(idx_bf12+8000) + ';\n')
        f.write('idx_bf13 = ' + str(idx_bf13+8000) + ';\n')
        f.write('idx_bf14 = ' + str(idx_bf14+8000) + ';\n')
        f.write('idx_bf15 = ' + str(idx_bf15+8000) + ';\n')

        # indices nacelle bottom BL in ff stage
        f.write('idx_bnb5 = ' + str(idx_bnb5+9000) + ';\n')
        f.write('idx_bnb6 = ' + str(idx_bnb6+9000) + ';\n')
        f.write('idx_bnb7 = ' + str(idx_bnb7+9000) + ';\n')
        # f.write('idx_bnb8 = ' + str(idx_bnb8+9000) + ';\n')
        f.write('idx_bnb9 = ' + str(idx_bnb9+9000) + ';\n')
        # f.write('idx_bnb10 = ' + str(idx_bnb10+9000) + ';\n')
        f.write('idx_bnb11 = ' + str(idx_bnb11+9000) + ';\n')
        # f.write('idx_bnb12 = ' + str(idx_bnb12+9000) + ';\n')
        f.write('idx_bnb13 = ' + str(idx_bnb13+9000) + ';\n')
        f.write('idx_bnb14 = ' + str(idx_bnb14+9000) + ';\n')
        f.write('idx_bnb15 = ' + str(idx_bnb15+9000) + ';\n')
        f.write('idx_bnb16 = ' + str(idx_bnb16+80000) + ';\n')

        # indices nacelle bottom in ff stage
        f.write('idx_nb5 = ' + str(idx_nb5+6000) + ';\n')
        f.write('idx_nb6 = ' + str(idx_nb6+6000) + ';\n')
        f.write('idx_nb7 = ' + str(idx_nb7+6000) + ';\n')
        # f.write('idx_nb8 = ' + str(idx_nb8+6000) + ';\n')
        f.write('idx_nb9 = ' + str(idx_nb9+6000) + ';\n')
        # f.write('idx_nb10 = ' + str(idx_nb10+6000) + ';\n')
        f.write('idx_nb11 = ' + str(idx_nb11+6000) + ';\n')
        # f.write('idx_nb12 = ' + str(idx_nb12+6000) + ';\n')
        f.write('idx_nb13 = ' + str(idx_nb13+6000) + ';\n')
        f.write('idx_nb14 = ' + str(idx_nb14+6000) + ';\n')
        f.write('idx_nb15 = ' + str(idx_nb15+6000) + ';\n')

        # indices nacelle top BL in ff stage
        f.write('idx_bnt50 = ' + str(idx_bnt50+11000) + ';\n')
        f.write('idx_bnt51 = ' + str(idx_bnt51+11000) + ';\n')
        f.write('idx_bnt52 = ' + str(idx_bnt52+11000) + ';\n')
        f.write('idx_bnt53 = ' + str(idx_bnt53+11000) + ';\n')
        f.write('idx_bnt54 = ' + str(idx_bnt54+11000) + ';\n')
        f.write('idx_bnt55 = ' + str(idx_bnt16+90000) + ';\n')

        # indices nacelle top ff stage
        f.write('idx_nt50 = ' + str(idx_nt50+5000) + ';\n')
        f.write('idx_nt51 = ' + str(idx_nt51+5000) + ';\n')
        f.write('idx_nt52 = ' + str(idx_nt52+5000) + ';\n')
        f.write('idx_nt53 = ' + str(idx_nt53+5000) + ';\n')
        f.write('idx_nt54 = ' + str(idx_nt54+5000) + ';\n')

        # indices fuselage BL
        # f.write('idx_fbl1 = ' + str(idx_fbl1+30000) + ';\n')
        # f.write('idx_fbl2 = ' + str(idx_fbl2+31000) + ';\n')
        # f.write('idx_fbl3 = ' + str(idx_fbl3+32000) + ';\n')
        f.write('idx_fbl4 = ' + str(idx_fbl4+33000) + ';\n')
        f.write('idx_fbl5 = ' + str(idx_fbl5+34000) + ';\n')
        f.write('idx_fbl16 = ' + str(idx_fbl16+35000) + ';\n')
        f.write('idx_fbl17 = ' + str(idx_fbl7+32500) + ';\n')
        f.write('idx_fbl_int = ' + str(idx_fbl_int+32500) + ';\n')

        # # indices fuselage mid line
        # f.write('idx_m1 = ' + str(idx_fbl1+40000) + ';\n')
        # f.write('idx_m2 = ' + str(idx_fbl2+41000) + ';\n')
        # f.write('idx_m3 = ' + str(idx_fbl3+42000) + ';\n')
        # f.write('idx_m4 = ' + str(idx_fbl4+43000) + ';\n')
        # f.write('idx_m5_1 = ' + str(idx_fbl5+44000) + ';\n')
        # f.write('idx_m16 = ' + str(idx_m16+45000) + ';\n')
        # f.write('idx_m17 = ' + str(idx_fbl7+42500) + ';\n')

        # indices fuselage intermediate line
        # f.write('idx_i1 = ' + str(idx_fbl1+50000) + ';\n')
        # f.write('idx_i2 = ' + str(idx_fbl2+51000) + ';\n')
        # f.write('idx_i3 = ' + str(idx_fbl3+52000) + ';\n')
        # f.write('idx_i4 = ' + str(idx_fbl4+53000) + ';\n')
        # f.write('idx_i5 = ' + str(idx_i5+54000) + ';\n')
        # f.write('idx_i16 = ' + str(idx_i16+55000) + ';\n')
        f.write('idx_i17 = ' + str(idx_fbl7+52500) + ';\n')
        f.write('idx_i_int = ' + str(idx_fbl_int+52500) + ';\n')

        # indices domain
        # f.write('idx_d1 = ' + str(idx_fbl1+60000) + ';\n')
        # f.write('idx_d2 = ' + str(idx_fbl2+61000) + ';\n')
        # f.write('idx_d3 = ' + str(idx_fbl3+62000) + ';\n')
        # f.write('idx_d4 = ' + str(idx_fbl4+63000) + ';\n')
        # f.write('idx_d5 = ' + str(idx_fbl5+64000) + ';\n')
        # f.write('idx_d16 = ' + str(idx_d16+65000) + ';\n')
        # f.write('idx_d17 = ' + str(idx_fbl7+62500) + ';\n')

        # indices ff stage vertical lines
        f.write('idx_67_1 = ' + str(idx_67_1+20000) + ';\n')
        # f.write('idx_67_2 = ' + str(idx_67_2+20100) + ';\n')
        # f.write('idx_67_3 = ' + str(idx_67_3+20200) + ';\n')
        f.write('idx_67_4 = ' + str(idx_67_4+20300) + ';\n')

        f.write('idx_78_1 = ' + str(idx_78_1+21000) + ';\n')
        f.write('idx_78_2 = ' + str(idx_78_2+21100) + ';\n')
        f.write('idx_78_3 = ' + str(idx_78_3+21200) + ';\n')
        f.write('idx_78_4 = ' + str(idx_78_4+21300) + ';\n')

        # f.write('idx_89_1 = ' + str(idx_89_1+22000) + ';\n')
        # f.write('idx_89_2 = ' + str(idx_89_2+22100) + ';\n')
        # f.write('idx_89_3 = ' + str(idx_89_3+22200) + ';\n')
        # f.write('idx_89_4 = ' + str(idx_89_4+22300) + ';\n')

        f.write('idx_910_1 = ' + str(idx_910_1+23000) + ';\n')
        f.write('idx_910_2 = ' + str(idx_910_2+23100) + ';\n')
        f.write('idx_910_3 = ' + str(idx_910_3+23200) + ';\n')
        f.write('idx_910_4 = ' + str(idx_910_4+23300) + ';\n')

        # f.write('idx_1011_1 = ' + str(idx_1011_1+24000) + ';\n')
        # f.write('idx_1011_2 = ' + str(idx_1011_2+24100) + ';\n')
        # f.write('idx_1011_3 = ' + str(idx_1011_3+24200) + ';\n')
        # f.write('idx_1011_4 = ' + str(idx_1011_4+24300) + ';\n')

        f.write('idx_1112_1 = ' + str(idx_1112_1+25000) + ';\n')
        f.write('idx_1112_2 = ' + str(idx_1112_2+25100) + ';\n')
        f.write('idx_1112_3 = ' + str(idx_1112_3+25200) + ';\n')
        f.write('idx_1112_4 = ' + str(idx_1112_4+25300) + ';\n')

        # f.write('idx_1213_1 = ' + str(idx_1213_1+26000) + ';\n')
        # f.write('idx_1213_2 = ' + str(idx_1213_2+26100) + ';\n')
        # f.write('idx_1213_3 = ' + str(idx_1213_3+26200) + ';\n')
        # f.write('idx_1213_4 = ' + str(idx_1213_4+26300) + ';\n')

        f.write('idx_1314_1 = ' + str(idx_1314_1+27000) + ';\n')
        f.write('idx_1314_2 = ' + str(idx_1314_2+27100) + ';\n')
        f.write('idx_1314_3 = ' + str(idx_1314_3+27200) + ';\n')
        f.write('idx_1314_4 = ' + str(idx_1314_4+27300) + ';\n')

        f.write('idx_1415_1 = ' + str(idx_1415_1+28000) + ';\n')
        # f.write('idx_1415_2 = ' + str(idx_1415_2+28100) + ';\n')
        # f.write('idx_1415_3 = ' + str(idx_1415_3+28200) + ';\n')
        f.write('idx_1415_4 = ' + str(idx_1415_4+28300) + ';\n')

        f.write('//No. of points vertical\n')
        f.write('n_v_fbl = ' + str(no_bl_cells_fuse) + ';\n')
        f.write('n_v_duct_rot_in = ' + str(n_v_duct_rot_in) + ';\n')
        f.write('n_v_duct_rot_out = ' + str(n_v_duct_rot_out) + ';\n')
        f.write('n_v_duct_stat_in = ' + str(n_v_duct_stat_in) + ';\n')
        f.write('n_v_duct_stat_out = ' + str(n_v_duct_stat_out) + ';\n')
        f.write('n_v_up = ' + str(n_v_up) + ';\n')
        f.write('n_v_low = ' + str(n_v_low) + ';\n')
        f.write('n_v_front_1 = ' + str(n_v_front_1) + ';\n')
        f.write('n_v_front_2 = ' + str(n_v_front_2) + ';\n')
        f.write('n_v_nbl = ' + str(no_bl_cells_nac) + ';\n')
        f.write('n_v_up_mid = ' + str(n_v_up_mid) + ';\n')
        f.write('n_v_int_mid = ' + str(n_v_int_mid) + ';\n')
        f.write('n_v_int_up = ' + str(n_v_int_up) + ';\n')
        f.write('n_v_up_mid = ' + str(n_v_up_mid) + ';\n')

        f.write('//No. of points horizontal\n')
        f.write('n_h_nose_1 = ' + str(n_h_nose_1) + ';\n')
        f.write('n_h_nose_2 = ' + str(n_h_nose_2) + ';\n')
        f.write('n_h_cent_2 = ' + str(n_h_cent_2) + ';\n')
        f.write('n_h_inlet_1 = ' + str(n_h_inl_1) + ';\n')
        f.write('n_h_inlet_2 = ' + str(n_h_inl_2) + ';\n')
        f.write('n_h_tail = ' + str(n_h_tail) + ';\n')
        f.write('n_h_rear_1 = ' + str(n_h_rear_1) + ';\n')
        f.write('n_h_rear_2 = ' + str(n_h_rear_2) + ';\n')
        f.write('n_h_ff_1_nac = ' + str(n_h_ff_1_nac) + ';\n')
        f.write('n_h_ff_2_nac = ' + str(n_h_ff_2_nac) + ';\n')
        f.write('n_h_ff_1_fus = ' + str(n_h_ff_1_fus) + ';\n')
        f.write('n_h_ff_2_fus = ' + str(n_h_ff_2_fus) + ';\n')
        f.write('n_h_rot_nac = ' + str(n_h_rot_nac) + ';\n')
        f.write('n_h_gap_nac = ' + str(n_h_gap_nac) + ';\n')
        f.write('n_h_stat_nac = ' + str(n_h_stat_nac) + ';\n')
        f.write('n_h_rot_fus = ' + str(n_h_rot_fus) + ';\n')
        f.write('n_h_gap_fus = ' + str(n_h_gap_fus) + ';\n')
        f.write('n_h_stat_fus = ' + str(n_h_stat_fus) + ';\n')
        f.write('n_h_ff_9_nac = ' + str(round(n_h_nozzle_nac/2)) + ';\n')
        f.write('n_h_ff_10_nac = ' + str(n_h_ff_10_nac) + ';\n')
        f.write('n_h_ff_9_fus = ' + str(round(n_h_nozzle_fus/2)) + ';\n')
        f.write('n_h_ff_10_fus = ' + str(n_h_ff_10_fus) + ';\n')
        f.write('n_h_nac_1 = ' + str(n_h_nac_1) + ';\n')
        f.write('n_h_nac_2 = ' + str(n_h_nac_2) + ';\n')
        f.write('n_h_nac_3 = ' + str(n_h_nac_3) + ';\n')
        f.write('n_h_nac_4 = ' + str(n_h_nac_4) + ';\n')
        f.write('n_h_inner = ' + str(n_h_inner) + ';\n')
        f.write('n_h_nose_out = ' + str(round(0.2*(n_h_nose_2+n_h_nose_1))) + ';\n')
        f.write('n_h_fuse_inner_back = ' + str(n_h_fuse_inner_back) + ';\n')
        f.write('n_h_fuse_inner_cent = ' + str(n_h_fuse_inner_cent) + ';\n')
        f.write('n_h_nose_1_inner = ' + str(n_h_nose_1_inner) + ';\n')
        f.write('n_h_nose_2_inner = ' + str(n_h_nose_2_inner) + ';\n')
        f.write('n_rear_domain = ' + str(n_rear_domain) + ';\n')
        f.write('n_rear_low = ' + str(n_rear_low) + ';\n')
        f.write('n_rear_up = ' + str(n_rear_up) + ';\n')
        f.write('n_cent_up = ' + str(n_cent_up) + ';\n')
        f.write('n_h_nose_up = ' + str(n_h_nose_up) + ';\n')

        f.write('//progressions vertical\n')
        f.write('p_v_fbl = ' + str(bl_ratio) + ';\n')
        f.write('p_v_low = ' + str(p_v_low) + ';\n')
        f.write('p_v_up = ' + str(p_v_up) + ';\n')
        f.write('p_v_duct_rot_in = ' + str(p_v_duct_rot_in) + ';\n')
        f.write('p_v_duct_rot_out = ' + str(p_v_duct_rot_out) + ';\n')
        f.write('p_v_duct_stat_in = ' + str(p_v_duct_stat_in) + ';\n')
        f.write('p_v_duct_stat_out = ' + str(p_v_duct_stat_out) + ';\n')
        f.write('p_v_nbl = ' + str(bl_ratio) + ';\n')
        f.write('p_v_front_1 = ' + str(p_v_front_1) + ';\n')
        f.write('p_v_front_2 = ' + str(p_v_front_2) + ';\n')
        f.write('p_v_up_mid = ' + str(p_v_up_mid) + ';\n')
        f.write('p_v_int_mid = ' + str(p_v_int_mid) + ';\n')
        f.write('p_v_int_up = ' + str(p_v_int_up) + ';\n')
        f.write('p_v_up_mid = ' + str(p_v_up_mid) + ';\n')

        f.write('//progressions horizontal\n')
        f.write('p_h_nose_1 = ' + str(p_h_nose_1) + ';\n')
        f.write('p_h_nose_2 = ' + str(p_h_nose_2) + ';\n')
        f.write('p_h_cent_2 = ' + str(p_h_cent_2) + ';\n')
        f.write('p_h_inlet_1 = ' + str(p_h_inl_1) + ';\n')
        f.write('p_h_inlet_2 = ' + str(p_h_inl_2) + ';\n')
        f.write('p_h_tail = ' + str(p_h_tail) + ';\n')
        f.write('p_h_rear_1 = ' + str(p_h_rear_1) + ';\n')
        f.write('p_h_rear_2 = ' + str(p_h_rear_2) + ';\n')
        f.write('p_h_ff_1_nac = ' + str(p_h_ff_1_nac) + ';\n')
        f.write('p_h_ff_2_nac = ' + str(p_h_ff_2_nac) + ';\n')
        f.write('p_h_ff_1_fus = ' + str(p_h_ff_1_fus) + ';\n')
        f.write('p_h_ff_2_fus = ' + str(p_h_ff_2_fus) + ';\n')
        f.write('p_h_rot = ' + str(p_h_rot) + ';\n')
        f.write('p_h_gap = ' + str(p_h_gap) + ';\n')
        f.write('p_h_stat = ' + str(p_h_stat) + ';\n')
        f.write('p_h_ff_9_nac = ' + str(p_h_ff_9_nac) + ';\n')
        f.write('p_h_ff_10_nac = ' + str(p_h_ff_10_nac) + ';\n')
        f.write('p_h_ff_9_fus = ' + str(p_h_ff_9_fus) + ';\n')
        f.write('p_h_ff_10_fus = ' + str(p_h_ff_10_fus) + ';\n')
        f.write('p_h_nose_out = ' + str(1.0) + ';\n')
        f.write('p_h_fuse_inner_back = ' + str(p_h_fuse_inner_back) + ';\n')
        f.write('p_h_fuse_inner_cent = ' + str(p_h_fuse_inner_cent) + ';\n')
        f.write('p_h_nose_1_inner = ' + str(p_h_nose_1_inner) + ';\n')
        f.write('p_h_nose_2_inner = ' + str(p_h_nose_2_inner) + ';\n')
        f.write('p_rear_domain = ' + str(p_rear_domain) + ';\n')
        f.write('p_rear_low = ' + str(p_rear_low) + ';\n')
        f.write('p_rear_up = ' + str(p_rear_up) + ';\n')
        f.write('p_cent_up = ' + str(p_cent_up) + ';\n')
        f.write('p_h_nose_up = ' + str(p_h_nose_up) + ';\n')
        f.write('p_h_nac_1 = ' + str(p_h_nac_1) + ';\n')
        f.write('p_h_nac_2 =' + str(p_h_nac_2) + ';\n')
        f.write('p_h_nac_3 = ' + str(p_h_nac_3) + ';\n')
        f.write('p_h_nac_4 = ' + str(p_h_nac_4) + ';\n')
        f.write('p_h_inner = ' + str(p_h_inner) + ';\n')

        for line in geo:
            f.write(line)

    # write files required for BFM in OpenFOAM
    h_duct = rotor_inlet[-1][1]-rotor_inlet[0][1]
    r_hub = rotor_inlet[0][1]
    write_scaled_blade_data(f'{casepath}',h_duct, r_hub, x_2_f, n_rot, n_stat, plot=False)

    return y_plus_fuse, hdomain
