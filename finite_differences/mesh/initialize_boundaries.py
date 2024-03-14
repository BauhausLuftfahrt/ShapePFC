"""Generates boundaries for grid generation.

Author:  A. Habermann

 Args:
    Xn              [m]     1-D array X-coordinate of geometric profile
    Yn              [m]     1-D array Y-coordinate of geometric profile
    n_x             [-]     Number of xi-ccordinates (transformed grid)
    n_y             [-]     Number of eta-ccordinates (transformed grid)
    ext_front       [-]     Extension of grid in front of geometry (in percent of max. body length)
    ext_rear       [-]     Extension of grid behind geometry (in percent of max. body length)
    ext_rad       [-]       Radial extension of grid (in percent of max. body height). Must be bigger than 1.
    type: str               Type of mesh. Valid variables: rect, slit, slab

Returns:
    boundaries: [2, x] array     x- any y-coordinates of grid boundary
    n_space: [1, 2] array        number of points in x- and y-direction

Sources:
    [1] Thompson, Joe F.; Thames, Frank C.; Mastin, C.Wayne: Automatic numerical generation of body-fitted curvilinear 
        coordinate system for field containing any number of arbitrary two-dimensional bodies.
        Journal of Computational Physics 15:3 (1974), 299 - 319.
    [2] Uchikawa, S.: Generation of boundary-fitted curvilinear coordinate systems for a two-dimensional axisymmetric 
        flow problem. Journal of Computational Physics 50:2 (1983), 316 - 321.
    [3] Thompson, Joe F.; Soni, B. K.; Weatherill, N. P.: Handbook of grid generation, CRC Press, Boca Raton (1999).
    [4] Thompson, Joe F.; Warsi, Z. U.; Mastin, C. W.: Numerical grid generation - Foundations and applications, 
        North-Holland, New York (1985).

"""

import numpy as np
from scipy import interpolate
from misc_functions.geometry.spacing_control import number_spacing, number_spacing_bump
from misc_functions.geometry.bezier_curve import bezier_curve
from geometry_generation.panel_geometry.parameter_sampling import paramSampling
import geomdl.BSpline as bspline
from geomdl import utilities
import warnings
from misc_functions.helpers.find_nearest import find_nearest_replace
from misc_functions.helpers.dimensionless_wall_distance import first_cell_height_y_plus
from finite_differences.mesh.kutta_condition import kutta_y
from geometry_generation.finite_difference_geometry.generate_jet import jetSnel_AreaRule, jetSeibold_AreaRule, \
    jetLiem_AreaRule


class InitBoundaries:

    def __init__(self, Xn: list, Yn: list, n_x: int, n_y: int, ext_front: float, ext_rear: float, ext_rad: float,
                 grid_type: str, grid_shape: str, calc_first_cell=False, **kwargs):
        self.grid_shape = grid_shape
        self.calc_first_cell = calc_first_cell
        self.Xn = Xn
        self.Yn = Yn
        self.n_x = n_x
        self.n_y = n_y
        self.ext_front = ext_front
        self.ext_rear = ext_rear
        self.ext_rad = ext_rad
        self.grid_type = grid_type
        # for PFC it is required to identify important FF stations by their x-coordinate in the following order:
        # [x_thr, x_rot,in , x_rot,out , x_stat_in , x_stat,out]
        if "fan_stations" in kwargs:
            self.fan_stations = kwargs["fan_stations"]
        # to calculate the height of the first cell with y+, the following input is required:
        # [y+, freestream Mach, freestream altitude]
        if "first_cell_values" in kwargs and calc_first_cell == True:
            self.first_cell_values = kwargs["first_cell_values"]
        elif "first_cell_values" not in kwargs and calc_first_cell == True:
            raise Warning("First cell height should be calculated, but no variables are given. Please check input.")
        # use cell width ratios for node spacing
        if "bl_ratio" in kwargs:
            self.bl_ratio = kwargs["bl_ratio"]

    def run(self):

        from geometry_generation.panel_geometry.parameter_sampling import translate_points
        # determine interpolated spline functions for body coordinates
        Fs = [0] * len(self.Xn)
        for i in range(0, len(self.Xn)):
            Fs[i] = interpolate.UnivariateSpline(self.Xn[i], self.Yn[i], s=0)

        # initialize boundaries
        # identify edges of boundaries
        x_geom_min = min(list(map(min, self.Xn)))
        x_geom_max = max(list(map(max, self.Xn)))
        x_grid_min = x_geom_min - self.ext_front * (x_geom_max - x_geom_min)
        x_grid_max = x_geom_max + self.ext_rear * (x_geom_max - x_geom_min)
        r_geom_min = min(list(map(min, self.Yn)))
        r_geom_max = max(list(map(max, self.Yn)))
        r_grid_min = min(list(map(min, self.Yn)))
        r_grid_max = r_geom_max * self.ext_rad + r_geom_max
        x_fuse_cent = x_geom_min + (x_geom_max - x_geom_min) * 2 / 3
        r_fuse_cent = Fs[0](x_fuse_cent)

        if self.grid_type == 'rect':
            # calculate number of points in front of and behind whole geometry
            n_x_front = max(int(round(self.n_x * self.ext_front / (self.ext_front + self.ext_rear + 1), 0)), int(40))
            n_x_rear = max(int(round(self.n_x * self.ext_rear / (self.ext_front + self.ext_rear + 1), 0)), int(20))
            n_x_geom = self.n_x - n_x_rear
            if len(self.Xn) == 3 and n_x_geom < 190:
                n_x_geom = 190
            elif len(self.Xn) == 1 and n_x_geom < 100:
                n_x_geom = 100  # 25
            warnings.warn(f"Specified number of nodes too small. Increased to n_x={n_x_geom + n_x_rear + n_x_front}")
            self.n_x = n_x_geom + n_x_rear + n_x_front

            # differentiate between geometry with one body (one subgrid) vs fuselage and nacelle bodies (two subgrids)
            if len(self.Xn) == 1 and self.grid_shape == 'c-grid':
                n_x_front = 0
                self.n_x = n_x_geom + n_x_rear + n_x_front
                front_length_ratio = 1
                x_grid_min = x_geom_min - front_length_ratio * (r_grid_max - r_fuse_cent)
                # define points
                x1 = min(self.Xn[0])
                x5 = max(self.Xn[0])
                x_geom_inlet = x2 = self.Xn[0][self.Yn[0].index(max(self.Yn[0]))]
                r_geom_inlet = np.max(self.Yn[0][0])

                # calculate height of first cell based on y+
                if self.calc_first_cell == True:
                    fuse_firstcell = first_cell_height_y_plus(self.first_cell_values[0], self.first_cell_values[1],
                                                              self.first_cell_values[2],
                                                              np.abs(self.Xn[0][-1] - self.Xn[0][0]))
                else:
                    fuse_firstcell = 5e-3

                spacing_right_down, _ = number_spacing(r_grid_min, 0.5 * r_grid_max, fuse_firstcell, int(self.n_y / 2),
                                                       r_grid_min, 0)
                spacing_right_up = np.arange(0.5 * r_grid_max, r_grid_max,
                                             spacing_right_down[-1] - spacing_right_down[-2])

                spacing_right = np.concatenate((spacing_right_down, spacing_right_up))

                spacing_low = np.array(
                    [-(spacing_right[i] - spacing_right[0]) + x_geom_min for i in range(0, len(spacing_right))])
                spacing_low = translate_points(spacing_low, spacing_low[0], spacing_low[-1], spacing_low[0], x_grid_min)

                r_grid_max = spacing_right[-1]
                self.n_y = len(spacing_right) - 1

                n_x_1_2 = max(int(60), int(round((x2 - x1) / (x5 - x1) * n_x_geom, 0)))
                n_x_2_3 = n_x_geom - n_x_1_2

                idx_inlet = np.where((np.array(self.Yn[0]) == max(np.array(self.Yn[0]))))[0][0]
                Fs_inlet = interpolate.UnivariateSpline(self.Yn[0][0:idx_inlet + 1], self.Xn[0][0:idx_inlet + 1], s=0)
                spacing_inlet_y, _, _ = number_spacing_bump(r_geom_min, r_geom_max, 0.01 * (r_geom_max - r_geom_min),
                                                            0.005 * (r_geom_max - r_geom_min), n_x_1_2 + 1)
                spacing_inlet = np.sort(Fs_inlet(spacing_inlet_y))
                # spacing_inlet, _ = number_spacing(x1, x2, 5e-5*(x_geom_max-x_geom_min), n_x_1_2 + 1, x1, 0)
                # spacing_inlet_y = Fs[0](spacing_inlet)

                # spacing_inlet, _ = number_spacing(x1, x2, 5e-5*(x_geom_max-x_geom_min), n_x_1_2 + 1, x1, 0)
                spacing_inlet_y_up = translate_points(spacing_inlet_y, spacing_inlet_y[0], spacing_inlet_y[-1],
                                                      r_grid_min, r_grid_max)
                spacing_geometry, _ = number_spacing(spacing_inlet[-1], x5, spacing_inlet[-1] - spacing_inlet[-2],
                                                     n_x_2_3, spacing_inlet[-1], 0)
                spacing_back, _ = number_spacing(x5, x_grid_max, spacing_geometry[-1] - spacing_geometry[-2],
                                                 n_x_rear + 1, x5, 0)
                spacing_inlet_up = translate_points(spacing_inlet, spacing_inlet[0], spacing_inlet[-1], x_grid_min,
                                                    spacing_inlet[-1])

                x_a0 = x_a1 = x_grid_min
                x_a2 = x2
                y_a0 = 0
                y_a1 = y_a2 = r_grid_max
                x_upper, y_upper = bezier_curve([[x_a0, y_a0], [x_a1, y_a1], [x_a2, y_a2]], nTimes=500)
                Fs_upper = interpolate.UnivariateSpline(np.flip(x_upper), np.flip(y_upper), s=0)
                Fs_upper_x = interpolate.UnivariateSpline(np.flip(y_upper), np.flip(x_upper), s=0)

                c1 = np.array([np.zeros((2 * self.n_x - 1)), np.zeros((2 * self.n_x - 1))])
                c2 = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
                c3 = np.array([np.zeros((2 * self.n_x - 1)), np.zeros((2 * self.n_x - 1))])
                c4 = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])

                c1_dup = np.array([np.concatenate((spacing_back[::-1], spacing_geometry[::-1], spacing_inlet[::-1],
                                                   spacing_inlet, spacing_geometry, spacing_back)),
                                   np.concatenate((np.concatenate(np.full((1, len(spacing_back)), r_grid_min)),
                                                   -Fs[0](spacing_geometry)[::-1],
                                                   -Fs[0](spacing_inlet)[::-1],
                                                   Fs[0](spacing_inlet),
                                                   Fs[0](spacing_geometry),
                                                   np.concatenate(np.full((1, len(spacing_back)),
                                                                          r_grid_min))))])  # lower bottom (left to right)

                c4_dup = np.array([np.concatenate(np.full((1, len(spacing_right)), spacing_back[-1])),
                                   spacing_right])

                c2_dup = np.array([np.concatenate(np.full((1, len(spacing_right)), spacing_back[-1])),
                                   -spacing_right])

                c3_dup = np.array([np.concatenate((spacing_back[::-1],
                                                   spacing_geometry[::-1],
                                                   Fs_upper_x(spacing_inlet_y_up[:-1])[::-1],
                                                   Fs_upper_x(spacing_inlet_y_up[:-1]),
                                                   spacing_geometry,
                                                   spacing_back)),
                                   np.concatenate((np.concatenate(np.full((1, len(spacing_back)), -r_grid_max)),
                                                   np.concatenate(np.full((1, len(spacing_geometry)), -r_grid_max)),
                                                   -spacing_inlet_y_up[:-1][::-1],
                                                   spacing_inlet_y_up[:-1],
                                                   np.concatenate(np.full((1, len(spacing_geometry)), r_grid_max)),
                                                   np.concatenate(np.full((1, len(spacing_back)),
                                                                          r_grid_max))))])  # lower top (left to right)

                # identify duplicates and remove from arrays
                c1_un = np.array([c1_dup[0][i] == c1_dup[0][i + 1] for i in range(0, len(c1_dup[0]) - 1)])
                c1_un = np.append(c1_un, bool(False))
                c1[0] = c1_dup[0][np.where(c1_un == False)]
                c1[1] = c1_dup[1][np.where(c1_un == False)]
                c3_un = np.array([c3_dup[0][i] == c3_dup[0][i + 1] for i in range(0, len(c3_dup[0]) - 1)])
                c3_un = np.append(c3_un, bool(False))
                c3[0] = c3_dup[0][np.where(c3_un == False)]
                c3[1] = c3_dup[1][np.where(c3_un == False)]

                # c1_un = np.unique(c1_dup[0], return_index=True)
                # c1[0] = c1_un[0]
                # c1[1] = c1_dup[1][c1_un[1]]
                c2_un = np.unique(c2_dup[1], return_index=True)
                c2[1] = np.flip(c2_un[0])
                c2[0] = np.flip(c2_dup[0][c2_un[1]])
                # c3_un = np.unique(c3_dup[0], return_index=True)
                # c3[0] = c3_un[0]
                # c3[1] = c3_dup[1][c3_un[1]]
                c4_un = np.unique(c4_dup[1], return_index=True)
                c4[1] = c4_un[0]
                c4[0] = c4_dup[0][c4_un[1]]

                boundaries = [c1, c2, c3, c4]

                # flag matrix contains information about the node types
                # 0: inner point; 1: surface; 2: inlet; 3: outlet; 4: farfield; 5: center axis
                flags_1 = [0] * (n_x_rear) + [1] * (2 * n_x_geom - 1) + [0] * (n_x_rear)
                flags_2 = [3] * len(c2[0])
                flags_3 = [4] * (self.n_x - len(spacing_inlet)) + [2] * (2 * len(spacing_inlet) - 1) + [4] * (
                            self.n_x - len(spacing_inlet))
                flags_4 = [3] * len(c4[0])

                boundaries = [c1, c2, c3, c4]
                boundary_flags = [flags_1, flags_2, flags_3, flags_4]
                n_space = [self.n_x, self.n_y]

            elif len(self.Xn) == 1 and self.grid_shape == 'rect-grid':
                x2 = self.Xn[0][self.Yn[0].index(max(self.Yn[0]))]
                n_x_1_2 = max(int(8), int(round((x2 - x_geom_min) / (x_geom_max - x_geom_min) * n_x_geom, 0)))
                spacing_nose, _ = paramSampling(np.linspace(x_geom_min, x2, 1000), Fs[0], n_x_1_2 + 1, 0.3, 0)
                samples_rest, _ = paramSampling(np.linspace(x2, x_geom_max, 1000), Fs[0], n_x_geom - n_x_1_2 + 2, 0.3,
                                                0)
                spacing_front0 = number_spacing(x_grid_min, float(x_geom_min), spacing_nose[1] - spacing_nose[0],
                                                n_x_front, x_geom_min, 0)
                spacing_back = number_spacing(x_geom_max, x_grid_max, samples_rest[-1] - samples_rest[-2], n_x_rear,
                                              x_geom_max, 0)

                c1 = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c1_dup = np.array([np.concatenate((spacing_front0[0],
                                                   spacing_nose[1:],
                                                   samples_rest[1:],
                                                   spacing_back[0])),
                                   np.concatenate((np.ones(n_x_front) * r_grid_min,  # np.zeros(n_x_front),
                                                   Fs[0](spacing_nose[1:]),
                                                   Fs[0](samples_rest[1:]),
                                                   np.ones(
                                                       n_x_rear) * r_grid_min))])  # np.zeros(n_x_rear)))])  # bottom (left to right)
                # calculate height of first cell based on y+
                if self.calc_first_cell == True:
                    fuse_firstcell = first_cell_height_y_plus(self.first_cell_values[0], self.first_cell_values[1],
                                                              self.first_cell_values[2],
                                                              np.abs(self.Xn[0][-1] - self.Xn[0][0]))
                else:
                    fuse_firstcell = 1e-2  # 7e-2

                spacing_left_down, ratio0 = number_spacing(r_grid_min, 0.5 * r_grid_max, fuse_firstcell,
                                                           int(self.n_y / 2), r_grid_min, 0)
                spacing_left_up = np.arange(0.5 * r_grid_max, r_grid_max, spacing_left_down[-1] - spacing_left_down[-2])
                r_grid_max = spacing_left_up[-1]
                self.n_y = len(spacing_left_up) + len(spacing_left_down) - 1
                c2 = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
                c2_dup = np.array([np.concatenate((np.full((1, self.n_y + 1), x_grid_min))),
                                   np.concatenate((spacing_left_down,
                                                   spacing_left_up))])
                c3 = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c3_dup = np.array([np.concatenate((spacing_front0[0],
                                                   spacing_nose[1:],
                                                   samples_rest[1:],
                                                   spacing_back[0])),
                                   np.concatenate((np.full((1, self.n_x + 3), r_grid_max)))])  # top (left to right)
                c4 = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
                c4_dup = np.array([np.concatenate((np.full((1, self.n_y + 1), x_grid_max))),
                                   np.concatenate((spacing_left_down,
                                                   spacing_left_up))])

                # identify duplicates and remove from arrays
                c1_un = np.unique(c1_dup[0], return_index=True)
                c1[0] = c1_un[0]
                c1[1] = c1_dup[1][c1_un[1]]
                c2_un = np.unique(c2_dup[1], return_index=True)
                c2[1] = c2_un[0]
                c2[0] = c2_dup[0][c2_un[1]]
                c3_un = np.unique(c3_dup[0], return_index=True)
                c3[0] = c3_un[0]
                c3[1] = c3_dup[1][c3_un[1]]
                c4_un = np.unique(c4_dup[1], return_index=True)
                c4[1] = c4_un[0]
                c4[0] = c4_dup[0][c4_un[1]]
                boundaries = [c1, c2, c3, c4]

                # flag matrix contains information about the node types
                # 0: inner point; 1: surface; 2: inlet; 3: outlet; 4: farfield; 5: center axis
                # flags_1 = [5]*(n_x_front-1)+[1]*(len(c1[0])-n_x_front-n_x_rear+2)+[5]*(n_x_rear-1)
                flags_1 = [5] * (n_x_front - 1) + [1] * (len(c1[0]) - n_x_front - n_x_rear + 2) + [5] * (n_x_rear - 1)
                flags_2 = [2] * len(c2[0])  # left / inlet
                flags_3 = [4] * len(c3[0])  # top
                flags_4 = [3] * len(c4[0])  # right / outlet
                boundary_flags = [flags_1, flags_2, flags_3, flags_4]
                n_space = [self.n_x, self.n_y]
                X_discret = [
                    c1[0][int(np.where(c1[0] == min(self.Xn[0]))[0]):int(np.where(c1[0] == max(self.Xn[0]))[0]) + 1]]
                Y_discret = [
                    c1[1][int(np.where(c1[0] == min(self.Xn[0]))[0]):int(np.where(c1[0] == max(self.Xn[0]))[0]) + 1]]

            elif len(self.Xn) == 3 and self.grid_shape == 'c-grid':
                # lower subgrid index "low", upper subgrid index "up"
                # define points of upper and lower grid. index "bar" for inner points
                self.n_x = n_x_geom + n_x_rear
                front_length_ratio = 1
                x_grid_min = x_geom_min - front_length_ratio * (r_grid_max - r_fuse_cent)
                x1 = min(self.Xn[0])
                x2 = self.Xn[0][self.Yn[0].index(max(self.Yn[0]))]
                x0 = x1 - (x2 - x1)
                x5 = max(self.Xn[0])
                x6 = x_grid_max
                x7 = x8 = x6
                x9 = x5_bar = x5
                x15 = x_grid_min
                x16 = x17 = x15
                x0_bar = x14 = x0
                x1_bar = x13 = x1
                x2_bar = x12 = x2
                x3_bar = min(self.Xn[1])
                x4_bar = max(self.Xn[1])
                x3 = x3_bar
                x4 = x4_bar
                x10 = x4_bar
                x11 = x3_bar
                x0_bar = x0 - 0.15 * (x1 - x0)  # +0.5*(x1-x0)

                delta = self.Yn[1][self.Xn[1].index(x3_bar)] - Fs[0](x3_bar)

                y1 = r_grid_min
                y2 = Fs[0](x2)
                y3 = Fs[0](x3)
                y4 = Fs[0](x4)
                y0 = y5 = y6 = y17 = y1
                y8 = r_grid_max
                y9 = y10 = y11 = y12 = y13 = y14 = y15 = y8
                y2_bar = y2 + delta
                y3_bar = self.Yn[1][self.Xn[1].index(x3_bar)]
                y4_bar = self.Yn[1][self.Xn[1].index(x4_bar)]
                y5_bar = y7 = y4_bar
                y16 = 0.8 * y7
                y0_bar = y16

                # generate bezier curve for front part of separated grids
                xb1 = x0_bar + 0.5 * (x2_bar - x0_bar)
                xb2 = x2_bar - 0.4 * (x2_bar - x0_bar)
                yb1 = y0_bar
                yb2 = y2_bar
                x_c = x0_bar + 0.5 * (x2_bar - x0_bar)
                y_c = 0.7 * (y2_bar - y0_bar) + y0_bar

                x_bez, y_bez = bezier_curve([[x0_bar, y0_bar], [xb1, yb1], [x_c, y_c], [xb2, yb2], [x2_bar, y2_bar]],
                                            nTimes=1000)
                Fs_bez = interpolate.UnivariateSpline(np.flip(x_bez), np.flip(y_bez), s=0)

                # generate curve for mid part of separated grids
                Y_in = [self.Yn[0][i] + delta for i in range(len(self.Yn[0]))]
                Fs_in = interpolate.UnivariateSpline(self.Xn[0], Y_in, s=0)

                n_x_1_2 = max(int(20), int(round((x2 - x1) / (x5 - x1) * n_x_geom, 0)))
                n_x_3_4 = max(int(20), int(round((x4 - x3) / (x5 - x1) * n_x_geom, 0)))
                n_x_4_5 = max(int(14), int(round((x5 - x4) / (x5 - x1) * n_x_geom, 0)))
                n_x_2_3 = n_x_geom - n_x_1_2 - n_x_3_4 - n_x_4_5

                n_x_0_1 = max(int(14), int(round((x1 - x0) / (x1 - x17) * n_x_front, 0)))
                n_x_17_0 = n_x_front - n_x_0_1
                n_x_5_6 = n_x_rear

                if n_x_1_2 < 0 or n_x_3_4 < 0 or n_x_4_5 < 0 or n_x_2_3 < 0 or n_x_0_1 < 0 or n_x_5_6 < 0:
                    raise Warning("No. of x-nodes too small. Solution impossible.")

                n_r_geom_low = int(round(1 / 2 * self.n_y))
                n_r_geom_up = self.n_y - n_r_geom_low

                x_nacelle = np.linspace(x3, x4, 1000)
                spacing_nacelle_bottom, _ = paramSampling(x_nacelle, Fs[2], n_x_3_4 + 2, 1, 1)
                # ensure that FF stations are part of sample
                spacing_nacelle_bottom = find_nearest_replace(spacing_nacelle_bottom, self.fan_stations)
                spacing_nacelle_top = np.flip(spacing_nacelle_bottom)
                self.n_x += (len(spacing_nacelle_bottom) - 2 - n_x_3_4)
                n_x_3_4 = len(spacing_nacelle_bottom) - 2
                idx_inlet = np.where((np.array(self.Yn[0]) == max(np.array(self.Yn[0]))))[0][0]
                Fs_inlet = interpolate.UnivariateSpline(self.Yn[0][0:idx_inlet + 1], self.Xn[0][0:idx_inlet + 1], s=0)
                spacing_nose_y, _, _ = number_spacing_bump(r_geom_min, r_geom_max, 0.01 * (r_geom_max - r_geom_min),
                                                           0.01 * (r_geom_max - r_geom_min), n_x_1_2 + 1)
                spacing_nose = Fs_inlet(spacing_nose_y)

                # spacing_nose, _ = number_spacing(x1, x2, 0.5e-3, n_x_1_2 + 2, x1, 0)#paramSampling(np.linspace(x1,x2,1000),Fs[0](np.linspace(x1,x2,1000)), n_x_1_2 + 2, 0.1, 0)#number_spacing(x1, x2, 0.08, n_x_1_2 + 2, x1, 0)
                spacing_front, _ = number_spacing(x0_bar, x1, spacing_nose[1] - spacing_nose[0], n_x_0_1 + 1, x1, -1)

                spacing_tail, _ = number_spacing(x4, x5, spacing_nacelle_bottom[-2] - spacing_nacelle_bottom[-1],
                                                 n_x_4_5 + 1,
                                                 x4)  # spacing_tail, _ = paramSampling(np.linspace(x4,x5,1000),Fs[0](np.linspace(x4,x5,1000)), n_x_4_5+1, 0.5, 0)
                spacing_back, _ = number_spacing(x5, x6, spacing_tail[-1] - spacing_tail[-2], n_x_5_6, x5, 0)

                # jet
                x_c0 = max(self.Xn[1])
                y_c0 = Fs[1](x_c0)
                x_c1 = x_geom_max
                x_c2 = spacing_back[3]
                # nacelle incidence angle
                alpha_0 = np.arctan(
                    (Fs[1](min(self.Xn[1])) - Fs[1](max(self.Xn[1]))) / (max(self.Xn[1]) - min(self.Xn[1])))
                # fuselage boattail angle
                alpha_1 = np.arctan((Fs[0](max(self.Xn[1])) - Fs[0](x_geom_max)) / (x_geom_max - max(self.Xn[1])))
                alpha_2 = (alpha_0 + alpha_1) / 2
                y_c1 = y_c2 = y_c0 - (x_c1 - x_c0) * np.tan(alpha_2)

                x_jet, y_jet = bezier_curve([[x_c0, y_c0], [x_c1, y_c1], [x_c2, y_c2]], nTimes=500)
                Fs_jet = interpolate.UnivariateSpline(np.flip(x_jet), np.flip(y_jet), s=0)

                spacing_centre, _, _ = number_spacing_bump(x2, x3, spacing_nose[-1] - spacing_nose[-2],
                                                           spacing_nacelle_top[1] - spacing_nacelle_top[0], n_x_2_3)
                spacing_centre_back, _ = number_spacing(x_fuse_cent, x3,
                                                        spacing_nacelle_top[-1] - spacing_nacelle_top[-2],
                                                        int(n_x_2_3 * 3 / 5 - 1), x3, -1)
                spacing_centre_front, _ = number_spacing(x2, x_fuse_cent,
                                                         spacing_centre_back[1] - spacing_centre_back[0],
                                                         int(n_x_2_3 * 2 / 5), x_fuse_cent,
                                                         -1)  # spacing_centre[0:int(len(spacing_centre)/2)]

                n_x1 = len(spacing_nose) + len(spacing_centre_front) - 1
                n_x2 = self.n_x - n_x1 - 2
                n_y1 = n_r_geom_low + 1

                # calculate height of first cell based on y+
                if self.calc_first_cell == True:
                    fuse_firstcell = first_cell_height_y_plus(self.first_cell_values[0], self.first_cell_values[1],
                                                              self.first_cell_values[2],
                                                              np.abs(self.Xn[0][-1] - self.Xn[0][0]))
                    nac_firstcell = fuse_firstcell  # conservative assumption, because nacelle inside slow velocity boundary layer
                else:
                    fuse_firstcell = 1e-6
                    nac_firstcell = 1e-6

                spacing_left_low, ratio0, ratio1 = number_spacing_bump(Fs[0](spacing_centre_front[-1]),
                                                                       Fs_in(spacing_centre_front[-1]), fuse_firstcell,
                                                                       nac_firstcell, n_r_geom_low + 1)

                spacing_left_up1, ratio2 = number_spacing(Fs_in(spacing_centre_front[-1]), 0.5 * r_grid_max,
                                                          spacing_left_low[-1] - spacing_left_low[-2], n_r_geom_up,
                                                          Fs_in(spacing_centre_front[-1]), 0)
                spacing_left_up2 = np.arange(0.5 * r_grid_max, r_grid_max, spacing_left_up1[-1] - spacing_left_up1[-2])

                spacing_left_up = np.concatenate((spacing_left_up1, spacing_left_up2))

                r_grid_max = spacing_left_up[-1]
                self.n_y = n_y1 + len(spacing_left_up) - 2
                n_y2 = self.n_y - n_y1 + 1

                spacing_right_low, ratio_right_1, ratio_right_2 = number_spacing_bump(y6, y_c2, fuse_firstcell,
                                                                                      nac_firstcell, n_r_geom_low + 1)
                spacing_right_up = translate_points(spacing_left_up, spacing_left_up[0], spacing_left_up[-1], y_c2,
                                                    r_grid_max - r_geom_max)

                x_grid_min = x_geom_min - front_length_ratio * (r_grid_max - r_fuse_cent)
                x_a0 = x_a1 = x_grid_min
                y_a0 = 0
                x_a2 = x2

                x_b0 = x_fuse_cent
                x_b1 = x_b2 = x_geom_max
                x_b3 = x_geom_max + r_geom_max / 2
                x_b4 = x_grid_max
                y_b0 = y_b1 = r_grid_max
                y_b2 = y_b3 = y_b4 = r_grid_max - r_geom_max

                x_upper_back, y_upper_back = bezier_curve(
                    [[x_b0, y_b0], [x_b1, y_b1], [x_b2, y_b2], [x_b3, y_b3], [x_b4, y_b4]], nTimes=500)
                Fs_upper_back = interpolate.UnivariateSpline(np.flip(x_upper_back), np.flip(y_upper_back), s=0)
                y_a1 = y_a2 = r_grid_max

                x_upper, y_upper = bezier_curve([[x_a0, y_a0], [x_a1, y_a1], [x_a2, y_a2]], nTimes=500)
                # Fs_upper = interpolate.UnivariateSpline(np.flip(x_upper), np.flip(y_upper), s=0)
                Fs_upper_x = interpolate.UnivariateSpline(np.flip(y_upper), np.flip(x_upper), s=0)

                spacing_low_1 = np.array([-(spacing_left_low[i] - spacing_left_low[0]) + x_geom_min for i in
                                          range(0, len(spacing_left_low))])
                spacing_low_2 = np.array(
                    [-(spacing_left_up[i] - spacing_left_up[0]) + x_geom_min - (spacing_low_1[0] - spacing_low_1[-1])
                     for i in range(0, len(spacing_left_up))])

                spacing_low_1 = translate_points(spacing_low_1, spacing_low_1[0], spacing_low_1[-1], spacing_low_1[0],
                                                 spacing_low_1[0] - front_length_ratio * (
                                                             spacing_low_1[0] - spacing_low_1[-1]))
                spacing_low_2 = translate_points(spacing_low_2, spacing_low_2[0], spacing_low_2[-1], spacing_low_1[-1],
                                                 spacing_low_1[0] - front_length_ratio * (
                                                             spacing_low_1[0] - spacing_low_2[-1]))
                n_x2 += 1
                c1_front = np.array([np.zeros((n_x1)), np.zeros((n_x1))])
                c2_front = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
                c3_front = np.array([np.zeros((n_x1)), np.zeros((n_x1))])
                c4_front = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])

                c1_back_low = np.array([np.zeros((n_x2)), np.zeros((n_x2))])
                c2_back_low = np.array([np.zeros((n_y1)), np.zeros((n_y1))])
                c3_back_low = np.array([np.zeros((n_x2)), np.zeros((n_x2))])
                c4_back_low = np.array([np.zeros((n_y1)), np.zeros((n_y1))])

                c1_back_up = np.array([np.zeros((n_x2)), np.zeros((n_x2))])
                c2_back_up = np.array([np.zeros((n_y2)), np.zeros((n_y2))])
                c3_back_up = np.array([np.zeros((n_x2)), np.zeros((n_x2))])
                c4_back_up = np.array([np.zeros((n_y2)), np.zeros((n_y2))])

                spacing_nose_y, _, _ = number_spacing_bump(r_geom_min, r_geom_max, 0.01 * (r_geom_max - r_geom_min),
                                                           0.002 * (r_geom_max - r_geom_min), n_x_1_2 + 1)
                spacing_nose = Fs_inlet(spacing_nose_y)
                spacing_nose_up = translate_points(spacing_nose_y, spacing_nose_y[0], spacing_nose_y[-1], r_grid_min,
                                                   r_grid_max)

                # 1 front part
                c1_front_dup = np.array([np.concatenate((spacing_nose,
                                                         spacing_centre_front)),
                                         np.concatenate((spacing_nose_y,
                                                         Fs[0](spacing_centre_front)))])  # lower bottom (left to right)

                c2_front = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
                c2_front_dup = np.array([np.concatenate((spacing_low_1, spacing_low_2)),
                                         np.concatenate((np.concatenate(np.full((1, len(spacing_low_1)), 0)),
                                                         np.concatenate(np.full((1, len(spacing_low_2)), 0))))])

                c3_front_dup = np.array([np.concatenate((Fs_upper_x(spacing_nose_up),
                                                         spacing_centre_front[1:])),
                                         np.concatenate((spacing_nose_up,
                                                         np.concatenate(np.full((1, len(spacing_centre_front) - 1),
                                                                                r_grid_max))))])  # lower top (left to right)

                c4_front = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
                c4_front_dup = np.array(
                    [np.concatenate((np.concatenate(np.full((1, n_r_geom_low + 1), spacing_centre_front[-1])),
                                     np.concatenate(np.full((1, len(spacing_left_up)), spacing_centre_front[-1])))),
                     np.concatenate((spacing_left_low,
                                     spacing_left_up))])

                # 2 lower back part
                c1_back_low_dup = np.array([np.concatenate((spacing_centre_back,
                                                            np.flip(spacing_nacelle_bottom[1:]),
                                                            spacing_tail,
                                                            spacing_back)),
                                            np.concatenate((Fs[0](spacing_centre_back),
                                                            Fs[0](np.flip(spacing_nacelle_bottom[1:])),
                                                            Fs[0](spacing_tail),
                                                            np.concatenate(
                                                                np.full((1, n_x_5_6), y5))))])

                c2_back_low_dup = np.array([np.concatenate(np.full((1, n_r_geom_low + 1), spacing_centre_front[-1])),
                                            spacing_left_low])

                c3_back_low_dup = np.array([np.concatenate((spacing_centre_back,
                                                            np.flip(spacing_nacelle_bottom[1:]),
                                                            np.concatenate((spacing_tail, spacing_back[0:3])),
                                                            spacing_back[3:])),
                                            np.concatenate((Fs_in(spacing_centre_back),
                                                            # np.linspace(x2, x3, n_x_2_3)),
                                                            Fs[2](np.flip(spacing_nacelle_bottom[1:])),
                                                            Fs_jet(np.concatenate((spacing_tail, spacing_back[0:3]))),
                                                            np.concatenate(np.full((1, len(spacing_back[3:])),
                                                                                   y_c2))))])  # bottom (left to right)

                c4_back_low_dup = np.array([np.concatenate(np.full((1, n_r_geom_low + 1), x6)),
                                            spacing_right_low])

                # 3 upper back part
                c1_back_up_dup = np.array([np.concatenate((spacing_centre_back,
                                                           spacing_nacelle_top,
                                                           np.concatenate((spacing_tail, spacing_back[0:3])),
                                                           spacing_back[3:])),
                                           np.concatenate((Fs_in(spacing_centre_back),  # np.linspace(x2, x3, n_x_2_3)),
                                                           Fs[1](spacing_nacelle_top),
                                                           Fs_jet(np.concatenate((spacing_tail, spacing_back[0:3]))),
                                                           np.concatenate(np.full((1, len(spacing_back[3:])),
                                                                                  y_c2))))])  # bottom (left to right)

                c2_back_up_dup = np.array([np.concatenate(np.full((1, len(spacing_left_up)), spacing_centre_front[-1])),
                                           spacing_left_up])

                c3_back_up_dup = np.array([np.concatenate((spacing_centre_back,
                                                           spacing_nacelle_top[1:],
                                                           spacing_tail,
                                                           spacing_back)),
                                           np.concatenate((Fs_upper_back(spacing_centre_back),
                                                           Fs_upper_back(spacing_nacelle_top[1:]),
                                                           Fs_upper_back(spacing_tail),
                                                           Fs_upper_back(spacing_back)))])  # top (left to right)

                c4_back_up_dup = np.array([np.concatenate(np.full((1, len(spacing_left_up)), x6)),
                                           spacing_right_up])

                # identify duplicates and remove from arrays
                c1_front_un = np.unique(c1_front_dup[0], return_index=True)
                c1_front[0] = c1_front_un[0]
                c1_front[1] = c1_front_dup[1][c1_front_un[1]]
                c2_front_un = np.unique(c2_front_dup[0], return_index=True)
                c2_front[0] = np.flip(c2_front_un[0])
                c2_front[1] = np.flip(c2_front_dup[1][c2_front_un[1]])
                c3_front_un = np.unique(c3_front_dup[0], return_index=True)
                c3_front[0] = c3_front_un[0]
                c3_front[1] = c3_front_dup[1][c3_front_un[1]]
                c4_front_un = np.unique(c4_front_dup[1], return_index=True)
                c4_front[1] = c4_front_un[0]
                c4_front[0] = c4_front_dup[0][c4_front_un[1]]

                c1_back_up_un = np.unique(c1_back_up_dup[0], return_index=True)
                c1_back_up[0] = c1_back_up_un[0]
                c1_back_up[1] = c1_back_up_dup[1][c1_back_up_un[1]]
                c2_back_up_un = np.unique(c2_back_up_dup[1], return_index=True)
                c2_back_up[1] = c2_back_up_un[0]
                c2_back_up[0] = c2_back_up_dup[0][c2_back_up_un[1]]
                c3_back_up_un = np.unique(c3_back_up_dup[0], return_index=True)
                c3_back_up[0] = c3_back_up_un[0]
                c3_back_up[1] = c3_back_up_dup[1][c3_back_up_un[1]]
                c4_back_up_un = np.unique(c4_back_up_dup[1], return_index=True)
                c4_back_up[1] = c4_back_up_un[0]
                c4_back_up[0] = c4_back_up_dup[0][c4_back_up_un[1]]

                c1_back_low_un = np.unique(c1_back_low_dup[0], return_index=True)
                c1_back_low[0] = c1_back_low_un[0]
                c1_back_low[1] = c1_back_low_dup[1][c1_back_low_un[1]]
                c2_back_low_un = np.unique(c2_back_low_dup[1], return_index=True)
                c2_back_low[1] = c2_back_low_un[0]
                c2_back_low[0] = c2_back_low_dup[0][c2_back_low_un[1]]
                c3_back_low_un = np.unique(c3_back_low_dup[0], return_index=True)
                c3_back_low[0] = c3_back_low_un[0]
                c3_back_low[1] = c3_back_low_dup[1][c3_back_low_un[1]]
                c4_back_low_un = np.unique(c4_back_low_dup[1], return_index=True)
                c4_back_low[1] = c4_back_low_un[0]
                c4_back_low[0] = c4_back_low_dup[0][c4_back_low_un[1]]

                boundaries = [c1_front, c2_front, c3_front, c4_front,
                              c1_back_low, c2_back_low, c3_back_low, c4_back_low,
                              c1_back_up, c2_back_up, c3_back_up, c4_back_up]

                # flag matrix contains information about the node types
                # 0: inner point; 1: surface; 2: inlet; 3: outlet; 4: farfield; 5: center axis
                flags_1_front = [1] * len(c1_front[0])
                flags_2_front = [5] * len(c2_front[0])  # left / inlet
                flags_3_front = [2] * (len(spacing_nose_up) - 1) + [4] * len(
                    spacing_centre_front)  # top (will be inside of mesh)
                flags_4_front = [0] * len(c4_front[0])  # right / outlet

                flags_1_back_up = [0] * (len(spacing_centre_back) - 1) + [1] * len(spacing_nacelle_top) + [0] * (
                            len(spacing_tail) + len(spacing_back) - 2)
                flags_2_back_up = [0] * len(c2_back_up[0])
                flags_3_back_up = [4] * len(c3_back_up[0])
                flags_4_back_up = [3] * len(c4_back_up[0])

                flags_1_back_low = [1] * (
                            len(spacing_centre_back) + len(spacing_nacelle_bottom[1:]) + len(spacing_tail) - 1) + [
                                       5] * (len(spacing_back) - 1)
                flags_2_back_low = [0] * len(c2_back_low[0])
                flags_3_back_low = [0] * (len(spacing_centre_back) - 1) + [1] * len(spacing_nacelle_top) + [0] * (
                            len(spacing_tail) + len(spacing_back) - 2)
                flags_4_back_low = [3] * len(c4_back_low[0])

                boundary_flags = [flags_1_front, flags_2_front, flags_3_front, flags_4_front,
                                  flags_1_back_low, flags_2_back_low, flags_3_back_low, flags_4_back_low,
                                  flags_1_back_up, flags_2_back_up, flags_3_back_up, flags_4_back_up]

                n_space = [n_x1, self.n_y, n_x2, n_y1, n_x2, n_y2]

            # nacelle
            elif len(self.Xn) == 2 and self.grid_shape == 'c-grid':
                n_x_front = 0
                len_nac = np.abs(self.Xn[0][-1] - self.Xn[0][0])
                r_grid_mid_back = self.Yn[0][-1]
                r_grid_mid_front = self.Yn[0][0]
                r_grid_max = r_grid_mid_back + self.ext_rad * 0.5 * len_nac
                r_grid_min = r_grid_mid_back - self.ext_rad * 0.5 * len_nac
                self.n_x = n_x_geom + n_x_rear + n_x_front
                x_grid_min = x_geom_min - (r_grid_max - r_grid_mid_back)

                # calculate height of first cell based on y+
                if self.calc_first_cell == True:
                    nac_firstcell = first_cell_height_y_plus(self.first_cell_values[0], self.first_cell_values[1],
                                                             self.first_cell_values[2],
                                                             len_nac)
                else:
                    nac_firstcell = 1e-5

                spacing_right_up, _ = number_spacing(r_grid_max, r_grid_mid_back, -nac_firstcell, int(self.n_y / 2 + 1),
                                                     r_grid_mid_back, -1)
                spacing_left, _ = number_spacing(r_grid_mid_back, r_grid_min, -nac_firstcell, int(self.n_y / 2 + 1),
                                                 r_grid_mid_back, 0)
                spacing_nacelle_top, _, _ = number_spacing_bump(self.Xn[0][0], self.Xn[0][-1], 0.001 * len_nac,
                                                                0.001 * len_nac, n_x_geom)
                spacing_nacelle_bottom, _, _ = number_spacing_bump(self.Xn[1][0], self.Xn[1][-1], 0.001 * len_nac,
                                                                   0.001 * len_nac, n_x_geom)
                spacing_back, _ = number_spacing(self.Xn[0][-1], x_grid_max,
                                                 spacing_nacelle_top[-1] - spacing_nacelle_top[-2], n_x_rear,
                                                 self.Xn[0][-1], 0)

                x_a0 = x_a1 = x_b0 = x_b1 = x_grid_min
                x_a2 = x_b2 = x_geom_max
                y_a0 = y_b0 = r_grid_mid_front
                y_a1 = y_a2 = r_grid_max
                y_b1 = y_b2 = r_grid_min

                x_freestream_top, y_freestream_top = bezier_curve([[x_a0, y_a0], [x_a1, y_a1], [x_a2, y_a2]],
                                                                  nTimes=500)
                x_freestream_bottom, y_freestream_bottom = bezier_curve([[x_b0, y_b0], [x_b1, y_b1], [x_b2, y_b2]],
                                                                        nTimes=500)
                F_freestream_top = interpolate.UnivariateSpline(np.flip(x_freestream_top), np.flip(y_freestream_top),
                                                                s=0)
                F_freestream_bottom = interpolate.UnivariateSpline(np.flip(x_freestream_bottom),
                                                                   np.flip(y_freestream_bottom), s=0)

                spacing_freestream_top = translate_points(spacing_nacelle_top, spacing_nacelle_top[0],
                                                          spacing_nacelle_top[-1],
                                                          x_grid_min, spacing_nacelle_top[-1])
                spacing_freestream_bottom = translate_points(spacing_nacelle_bottom, spacing_nacelle_bottom[0],
                                                             spacing_nacelle_bottom[-1],
                                                             x_grid_min, spacing_nacelle_bottom[-1])
                spacing_freestream_top_2, _ = number_spacing(self.Xn[0][-1], x_grid_max,
                                                             spacing_freestream_top[-1] - spacing_freestream_top[-2],
                                                             n_x_rear + 1, self.Xn[0][-1], 0)
                spacing_freestream_bottom_2, _ = number_spacing(self.Xn[1][-1], x_grid_max,
                                                                spacing_freestream_bottom[-1] -
                                                                spacing_freestream_bottom[-2],
                                                                n_x_rear + 1, self.Xn[1][-1], 0)
                spacing_inner_low = np.flip(
                    [spacing_left[i] + (x_geom_min - r_grid_mid_back) for i in range(0, len(spacing_left))])
                spacing_inner_up = np.flip(spacing_inner_low)
                spacing_right_low, _ = number_spacing(r_grid_mid_back, r_grid_min, -nac_firstcell,
                                                      int(self.n_y / 2 + 1), r_grid_mid_back, 0)

                c1_up = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c3_up = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])

                c1_dup_up = np.array([np.concatenate((spacing_nacelle_top, spacing_freestream_top_2)),
                                      np.concatenate((Fs[0](spacing_nacelle_top),
                                                      np.concatenate(np.full((1, len(spacing_freestream_top_2)),
                                                                             r_grid_mid_back))))])

                c2_up = np.array([spacing_inner_up,
                                  np.concatenate((np.full((1, len(spacing_inner_up)), r_grid_mid_front)))])

                c3_dup_up = np.array([np.concatenate((spacing_freestream_top, spacing_freestream_top_2)),
                                      np.concatenate((F_freestream_top(spacing_freestream_top),
                                                      np.concatenate(
                                                          np.full((1, len(spacing_freestream_top_2)), r_grid_max))))])

                c4_up = np.array([np.concatenate((np.full((1, len(spacing_right_up)), x_grid_max))),
                                  np.flip(spacing_right_up)])

                c1_low = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c3_low = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])

                c1_dup_low = np.array([np.concatenate((spacing_freestream_bottom, spacing_freestream_bottom_2)),
                                       np.concatenate((F_freestream_bottom(spacing_freestream_bottom),
                                                       np.concatenate(
                                                           (np.full((1, len(spacing_freestream_bottom_2)), r_grid_min)))
                                                       ))])

                c2_low = np.array([spacing_inner_low,
                                   np.concatenate((np.full((1, len(spacing_inner_low)), r_grid_mid_front)))])

                c3_dup_low = np.array([np.concatenate((spacing_nacelle_bottom, spacing_freestream_bottom_2)),
                                       np.concatenate((Fs[1](spacing_nacelle_bottom),
                                                       np.concatenate((np.full((1, len(spacing_freestream_bottom_2)),
                                                                               r_grid_mid_back)))))])

                c4_low = np.array([np.concatenate((np.full((1, len(spacing_right_low)), x_grid_max))),
                                   np.flip(spacing_right_low)])

                # # identify duplicates and remove from arrays
                c1_un_low = np.unique(c1_dup_low[0], return_index=True)
                c1_low[0] = c1_un_low[0]
                c1_low[1] = c1_dup_low[1][c1_un_low[1]]
                c1_un_up = np.unique(c1_dup_up[0], return_index=True)
                c1_up[0] = c1_un_up[0]
                c1_up[1] = c1_dup_up[1][c1_un_up[1]]
                c3_un_low = np.unique(c3_dup_low[0], return_index=True)
                c3_low[0] = c3_un_low[0]
                c3_low[1] = c3_dup_low[1][c3_un_low[1]]
                c3_un_up = np.unique(c3_dup_up[0], return_index=True)
                c3_up[0] = c3_un_up[0]
                c3_up[1] = c3_dup_up[1][c3_un_up[1]]

                # flag matrix contains information about the node types
                # 0: inner point; 1: surface at symmetry; 11: surface bottom; 12 surface top; 2: inlet; 3: outlet; 4: farfield; 5: center axis
                # 61: inner boundary low; 62: inner boundary top; 7: nacelle trailing edge

                flags_1_up = [1] * len(spacing_nacelle_top) + [0] * (len(spacing_freestream_top_2) - 1)
                flags_2_up = [62] * len(c2_up[0])
                flags_3_up = [2] * len(spacing_nacelle_top) + [4] * (len(spacing_freestream_top_2) - 1)
                flags_4_up = [3] * len(c4_up[0])
                flags_1_low = [2] * len(spacing_nacelle_bottom) + [4] * (len(spacing_freestream_bottom_2) - 1)
                flags_2_low = [61] * len(c2_low[0])
                flags_3_low = [1] * len(spacing_nacelle_bottom) + [0] * (len(spacing_freestream_bottom_2) - 1)
                flags_4_low = [3] * len(c4_low[0])

                boundary_flags = [flags_1_low, flags_2_low, flags_3_low, flags_4_low,
                                  flags_1_up, flags_2_up, flags_3_up, flags_4_up]
                n_space = [self.n_x, self.n_y, self.n_x, self.n_y]
                boundaries = [c1_low, c2_low, c3_low, c4_low, c1_up, c2_up, c3_up, c4_up]

                import matplotlib.pyplot as plt
                from post_processing.finite_difference_post.plot_grid import plotBoundaries
                plotBoundaries(boundaries)
                plt.show()

            elif len(self.Xn) == 3 and self.grid_shape == 'rect-grid':
                # lower subgrid index "low", upper subgrid index "up"
                # define points of upper and lower grid. index "bar" for inner points
                x1 = min(self.Xn[0])
                x2 = self.Xn[0][self.Yn[0].index(max(self.Yn[0]))]
                x0 = x1 - 0.3 * (x2 - x1)
                x5 = max(self.Xn[0])
                x6 = x_grid_max
                x7 = x8 = x6
                x9 = x5_bar = x5
                x15 = x_grid_min
                x16 = x17 = x15
                x0_bar = x14 = x0
                x1_bar = x13 = x1
                x2_bar = x12 = x2
                x3_bar = min(self.Xn[1])
                x4_bar = max(self.Xn[1])
                x3 = x3_bar
                x4 = x4_bar
                x10 = x4_bar
                x11 = x3_bar
                x0_bar = x0 - 0.8 * (x1 - x0)  # +0.5*(x1-x0)

                delta = self.Yn[1][self.Xn[1].index(x3_bar)] - Fs[0](x3_bar)

                y1 = r_grid_min
                y2 = Fs[0](x2)
                y3 = Fs[0](x3)
                y4 = Fs[0](x4)
                y0 = y5 = y6 = y17 = y1
                y8 = r_grid_max
                y9 = y10 = y11 = y12 = y13 = y14 = y15 = y8
                y2_bar = y2 + delta
                y3_bar = self.Yn[1][self.Xn[1].index(x3_bar)]
                y4_bar = self.Yn[1][self.Xn[1].index(x4_bar)]
                y5_bar = y7 = y2_bar  # y4_bar+1.2
                y16 = y7  # r_grid_min + delta#y7
                y0_bar = y7  # y0 + delta#y7

                n_x_1_2 = max(int(35), int(round((x2 - x1) / (x5 - x1) * n_x_geom, 0)))
                n_x_3_4 = max(int(44), int(round((x4 - x3) / (x5 - x1) * n_x_geom, 0)))
                n_x_4_5 = max(int(30), int(round((x5 - x4) / (x5 - x1) * n_x_geom, 0)))
                n_x_2_3 = n_x_geom - n_x_1_2 - n_x_3_4 - n_x_4_5

                n_x_0_1 = max(int(16), int(round((x1 - x0) / (x1 - x17) * n_x_front, 0)))
                n_x_17_0 = n_x_front - n_x_0_1
                n_x_5_6 = n_x_rear

                if n_x_1_2 < 0 or n_x_3_4 < 0 or n_x_4_5 < 0 or n_x_2_3 < 0 or n_x_0_1 < 0 or n_x_17_0 < 0 or n_x_5_6 < 0:
                    raise Warning("No. of x-nodes too small. Solution impossible.")

                n_r_geom_low = int(round(2 / 3 * self.n_y))
                n_r_geom_up = self.n_y - n_r_geom_low

                x_nacelle = np.linspace(x3, x4, 1000)
                spacing_nacelle_bottom, _ = paramSampling(x_nacelle, Fs[2], n_x_3_4 + 2, 1, 1)
                # spacing_nacelle_bottom, _, _ = number_spacing_bump(x3, x4, 0.005*(x4-x3), 0.005*(x4-x3), n_x_3_4 + 2)#paramSampling(x_nacelle,Fs[2], n_x_3_4 + 2, 1, 1)
                # spacing_nacelle_bottom = np.flip(spacing_nacelle_bottom)
                # ensure that FF stations are part of sample
                spacing_nacelle_bottom = find_nearest_replace(spacing_nacelle_bottom, self.fan_stations)
                spacing_nacelle_top = np.flip(spacing_nacelle_bottom)
                self.n_x += (len(spacing_nacelle_bottom) - 2 - n_x_3_4)
                n_x_3_4 = len(spacing_nacelle_bottom) - 2
                idx_inlet = np.where((np.array(self.Yn[0]) == max(np.array(self.Yn[0]))))[0][0]
                Fs_inlet = interpolate.UnivariateSpline(self.Yn[0][0:idx_inlet + 1], self.Xn[0][0:idx_inlet + 1], s=0)
                Fs_inlet_y = interpolate.UnivariateSpline(self.Xn[0][0:idx_inlet + 1], self.Yn[0][0:idx_inlet + 1], s=0)
                # spacing_nose_y,_,_ = number_spacing_bump(r_geom_min, r_geom_max, 0.02*(r_geom_max-r_geom_min), 0.02*(r_geom_max-r_geom_min), n_x_1_2)
                # spacing_nose = Fs_inlet(spacing_nose_y)
                # insert_values = [spacing_nose[-2]+(spacing_nose[-1]-spacing_nose[-2])*(i+1)/3 for i in range(0,2)]
                # spacing_nose = np.insert(spacing_nose,-1,insert_values)
                # spacing_nose_y = np.insert(spacing_nose_y,-1,Fs_inlet_y(insert_values))
                spacing_nose, _ = paramSampling(np.linspace(x_geom_min, x2, 1000), Fs[0], n_x_1_2 + 2, 0.3, 0)
                spacing_nose_y = Fs[0](spacing_nose)
                spacing_front, _ = number_spacing(x0_bar, x1, (spacing_nose[1] - spacing_nose[0]), n_x_0_1 + 1, x1)
                spacing_front0 = np.flip(np.arange(x0_bar, x17, -(spacing_front[1] - spacing_front[0])))
                x_grid_min = x16 = spacing_front0[0]
                self.n_x -= (n_x_17_0 - len(spacing_front0))
                n_x_17_0 = len(spacing_front0)

                delta_inner = 0
                front_inner_start = spacing_front0[-1] - delta_inner
                nose_inner_start = spacing_nose[0] - delta_inner

                spacing_front0_inner = translate_points(spacing_front0, spacing_front0[0], spacing_front0[-1],
                                                        spacing_front0[0], front_inner_start)
                spacing_front_inner = translate_points(spacing_front, spacing_front[0], spacing_front[-1],
                                                       front_inner_start, nose_inner_start)
                spacing_nose_inner = translate_points(spacing_nose, spacing_nose[0], spacing_nose[-1], nose_inner_start,
                                                      spacing_nose[-1])
                x0_bar = front_inner_start

                # generate bezier curve for front part of separated grids
                xb1 = x0_bar + 0.4 * (x2_bar - x0_bar)
                xb2 = x2_bar - 0.3 * (x2_bar - x0_bar)
                yb1 = y0_bar
                yb2 = y2_bar
                x_c = x0_bar + 0.3 * (x2_bar - x0_bar)
                y_c = 0.7 * (y2_bar - y0_bar) + y0_bar

                x_bez, y_bez = bezier_curve([[x0_bar, y0_bar], [xb1, yb1], [x_c, y_c], [xb2, yb2], [x2_bar, y2_bar]],
                                            nTimes=1000)
                Fs_bez = interpolate.UnivariateSpline(np.flip(x_bez), np.flip(y_bez), s=0)

                # generate curve for mid part of separated grids
                Y_in = [self.Yn[0][i] + delta for i in range(len(self.Yn[0]))]
                Fs_in = interpolate.UnivariateSpline(self.Xn[0], Y_in, s=0)

                y_pts1 = Fs_bez(spacing_front_inner)
                y_pts2 = Fs_bez(spacing_nose_inner[1:])

                spacing_tail_test, _, _ = number_spacing_bump(x4, x5,
                                                              spacing_nacelle_bottom[-2] - spacing_nacelle_bottom[-1],
                                                              10 * (spacing_nacelle_bottom[-2] - spacing_nacelle_bottom[
                                                                  -1]), n_x_4_5 + 1)
                spacing_tail_1, _ = number_spacing(x4, x4 + 0.4 * (x5 - x4),
                                                   spacing_nacelle_bottom[-2] - spacing_nacelle_bottom[-1],
                                                   int(0.7 * n_x_4_5), x4)
                spacing_tail_2 = np.linspace(x4 + 0.4 * (x5 - x4), x5, n_x_4_5 - len(spacing_tail_1) + 2)
                spacing_tail = np.append(spacing_tail_1, spacing_tail_2[1:])
                # spacing_tail, _ = number_spacing(x4, x5, spacing_nacelle_bottom[-2]-spacing_nacelle_bottom[-1], n_x_4_5+1,x4)#spacing_tail, _ = paramSampling(np.linspace(x4,x5,1000),Fs[0](np.linspace(x4,x5,1000)), n_x_4_5+1, 0.5, 0)
                spacing_back, _ = number_spacing(x5, x6, spacing_tail[-1] - spacing_tail[-2], n_x_5_6, x5, 0)
                spacing_centre_1, _ = number_spacing(x2, x2 + 0.5 * (x3 - x2), spacing_nose[-1] - spacing_nose[-2],
                                                     int(24),
                                                     x2)  # np.arange(x2, x2+0.5*(x3-x2), spacing_nose[-1]-spacing_nose[-2])
                spacing_centre_2, _ = number_spacing(x2 + 0.5 * (x3 - x2), x3,
                                                     spacing_nacelle_top[1] - spacing_nacelle_top[0],
                                                     n_x_2_3 - len(spacing_centre_1) + 1, x3)
                spacing_centre = np.append(spacing_centre_1, spacing_centre_2[1:])

                # spacing_centre, _, _ = number_spacing_bump(x2, x3, spacing_nose[-1]-spacing_nose[-2], spacing_nacelle_top[1]-spacing_nacelle_top[0], n_x_2_3)

                # enforce kutta condition at nacelle trailing edge
                # todo: find out how many points are required
                kutta_no = 2
                x_kutta = spacing_tail[1:kutta_no + 1]
                y_kutta = kutta_y(spacing_nacelle_top, np.flip(spacing_nacelle_bottom),
                                  Fs[2](np.flip(spacing_nacelle_bottom)), Fs[1](spacing_nacelle_top), x_kutta)

                # jet geometry
                d_eq = 2 * np.sqrt(y_kutta[-1] ** 2 - Fs[0](x_kutta[-1]) ** 2)
                x_jet = np.append(spacing_tail[kutta_no + 1:-1], spacing_back) - spacing_tail[kutta_no + 1]
                r_fuselage = np.concatenate((Fs[0](spacing_tail[kutta_no + 1:-1]), np.zeros(len(spacing_back))))
                y_jet_Liem, v_ent_liem, x_len_core_liem, _ = jetLiem_AreaRule(x_jet, d_eq, r_fuselage)
                y_jet_Seibold, v_ent_seibold, x_len_core_seibold = jetSeibold_AreaRule(x_jet, d_eq, r_fuselage)
                y_jet_Snel, v_ent_snel, x_len_core_snel = jetSnel_AreaRule(x_jet, d_eq, r_fuselage)
                x_jet += spacing_tail[kutta_no + 1]
                # test alt 1 from here
                x6 = x_grid_max = \
                spacing_back[np.where((spacing_back > x_len_core_snel + spacing_tail[kutta_no + 1]))[0]][0]
                x_jet = x_back = np.concatenate((x_kutta, x_jet[0:int(np.where(x_jet == x_grid_max)[0] + 1)]))
                spacing_back = spacing_back[0:int(np.where(spacing_back == x_grid_max)[0] + 1)]
                y_back = np.concatenate((y_kutta, y_jet_Snel[0:len(x_jet) - 2]))
                self.n_x -= (-len(spacing_back) + n_x_rear)

                y7 = y_back[-1]
                n_x_5_6 -= n_x_5_6 + n_x_4_5 - len(y_back)

                # calculate height of first cell based on y+
                if self.calc_first_cell == True:
                    fuse_firstcell = first_cell_height_y_plus(self.first_cell_values[0], self.first_cell_values[1],
                                                              self.first_cell_values[2],
                                                              np.abs(self.Xn[0][-1] - self.Xn[0][0]))
                    nac_firstcell = first_cell_height_y_plus(self.first_cell_values[0], self.first_cell_values[1],
                                                             self.first_cell_values[2],
                                                             np.abs(self.Xn[1][-1] - self.Xn[1][0]))
                else:
                    fuse_firstcell = 2e-2
                    nac_firstcell = 2e-2

                spacing_left, ratio0, ratio1 = number_spacing_bump(y17, y16, fuse_firstcell, nac_firstcell,
                                                                   n_r_geom_low)

                spacing_left_up_down, _ = number_spacing(y16, y16 + 0.1 * (r_grid_max - y16),
                                                         spacing_left[-1] - spacing_left[-2], int(n_r_geom_up / 2), y16,
                                                         0)
                spacing_left_up_up = np.arange(y16 + 0.1 * (r_grid_max - y16), r_grid_max,
                                               spacing_left_up_down[-1] - spacing_left_up_down[-2])
                spacing_left_up = np.concatenate((spacing_left_up_down, spacing_left_up_up))

                y15 = r_grid_max = spacing_left_up[-1]
                self.n_y = len(spacing_left_up) - 1

                spacing_right_up_down, _ = number_spacing(y7, y7 + 0.1 * (r_grid_max - y7), nac_firstcell,
                                                          int(n_r_geom_up / 2), y7, 0)
                spacing_right_up_up = translate_points(spacing_left_up_up, spacing_left_up_up[0],
                                                       spacing_left_up_up[-1], y7 + 0.1 * (r_grid_max - y7), r_grid_max)
                spacing_right_up = np.concatenate((spacing_right_up_down, spacing_right_up_up))

                spacing_right, ratio0, ratio1 = number_spacing_bump(y6, y7, fuse_firstcell, nac_firstcell, n_r_geom_low)

                c1_low_dup = np.array([np.concatenate((spacing_front0,
                                                       spacing_front,
                                                       spacing_nose[1:],
                                                       spacing_centre,
                                                       np.flip(spacing_nacelle_bottom),
                                                       x_back)),
                                       np.concatenate((np.concatenate(np.full((1, n_x_17_0), y1)),
                                                       np.concatenate(np.full((1, n_x_0_1 + 1), y1)),
                                                       spacing_nose_y[1:],
                                                       Fs[0](spacing_centre),
                                                       Fs[0](np.flip(spacing_nacelle_bottom)),
                                                       Fs[0](spacing_tail[1:]),
                                                       np.concatenate(
                                                           np.full((1, len(x_back) - len(Fs[0](spacing_tail)) + 1),
                                                                   y5))))])  # lower bottom (left to right)

                # initialize arrays without duplicates
                c1_low = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c1_up = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c3_low = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c3_up = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c2_up = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
                c4_up = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])

                c2_low = np.array([np.concatenate((np.full((1, n_r_geom_low), x16))),
                                   spacing_left])
                c3_low_dup = np.array([np.concatenate((spacing_front0_inner,
                                                       spacing_front_inner[1:],
                                                       spacing_nose_inner[1:],
                                                       spacing_centre,
                                                       np.flip(spacing_nacelle_bottom),
                                                       x_back)),
                                       np.concatenate((np.concatenate(np.full((1, n_x_17_0), y16)),
                                                       np.array(y_pts1[1:]),
                                                       np.array(y_pts2),
                                                       Fs_in(spacing_centre),
                                                       Fs[2](np.flip(spacing_nacelle_bottom)),
                                                       y_back))])  # lower top (left to right)

                c4_low = np.array([np.concatenate((np.full((1, n_r_geom_low), x6))),
                                   spacing_right])

                c1_up_dup = np.array([np.concatenate((spacing_front0_inner,
                                                      spacing_front_inner[1:],
                                                      spacing_nose_inner[1:],
                                                      spacing_centre,
                                                      spacing_nacelle_top,
                                                      x_back)),
                                      np.concatenate((np.concatenate(np.full((1, n_x_17_0), y16)),
                                                      np.array(y_pts1[1:]),
                                                      np.array(y_pts2),
                                                      Fs_in(spacing_centre),
                                                      Fs[1](spacing_nacelle_top),
                                                      y_back))])  # lower top (left to right)

                c2_up_dup = np.array([np.concatenate((np.full((1, len(spacing_left_up)), x16))),
                                      spacing_left_up])
                c3_up_dup = np.array([np.concatenate((spacing_front0,
                                                      spacing_front,
                                                      spacing_nose[1:],
                                                      spacing_centre,
                                                      spacing_nacelle_top,
                                                      x_back)),
                                      np.ones(self.n_x + 4) * y15])  # upper top (left to right)

                c4_up_dup = np.array([np.concatenate((np.full((1, len(spacing_right_up)), x6))),
                                      spacing_right_up])

                # identify duplicates and remove from arrays
                c1_low_un = np.unique(c1_low_dup[0], return_index=True)
                c1_low[0] = c1_low_un[0]
                c1_low[1] = c1_low_dup[1][c1_low_un[1]]
                c3_low_un = np.unique(c3_low_dup[0], return_index=True)
                c3_low[0] = c3_low_un[0]
                c3_low[1] = c3_low_dup[1][c3_low_un[1]]
                c1_up_un = np.unique(c1_up_dup[0], return_index=True)
                c1_up[0] = c1_up_un[0]
                c1_up[1] = c1_up_dup[1][c1_up_un[1]]
                c3_up_un = np.unique(c3_up_dup[0], return_index=True)
                c3_up[0] = c3_up_un[0]
                c3_up[1] = c3_up_dup[1][c3_up_un[1]]
                c2_up_un = np.unique(c2_up_dup[1], return_index=True)
                c2_up[1] = c2_up_un[0]
                c2_up[0] = c2_up_dup[0][c2_up_un[1]]
                c4_up_un = np.unique(c4_up_dup[1], return_index=True)
                c4_up[1] = c4_up_un[0]
                c4_up[0] = c4_up_dup[0][c4_up_un[1]]

                boundaries = [c1_low, c2_low, c3_low, c4_low, c1_up, c2_up, c3_up, c4_up]

                # flag matrix contains information about the node types
                # 0: inner point; 1: surface at symmetry; 11: surface bottom; 12 surface top; 2: inlet; 3: outlet; 4: farfield; 5: center axis
                # 61: inner boundary low; 62: inner boundary top; 71: upper nacelle trailing edge, 7: lower nacelle trailing edge
                flags_1_low = [5] * (n_x_17_0 + n_x_0_1) + [1] * (len(c1_low[0]) - n_x_17_0 - n_x_0_1 - n_x_5_6 + 1) + [
                    5] * (n_x_5_6 - 1)  # bottom
                flags_2_low = [2] * len(c2_low[0])  # left / inlet
                flags_3_low = [61] * (n_x_17_0 + n_x_0_1 + n_x_1_2 + n_x_2_3 - 1) + [12] * (
                            len(c3_low[0]) - n_x_17_0 - n_x_0_1 -
                            n_x_1_2 - n_x_2_3 - n_x_4_5 - n_x_5_6 + 1) + [72] \
                              + [61] * (n_x_4_5 + n_x_5_6 - 1)  # top (will be inside of mesh)
                flags_4_low = [3] * len(c4_low[0])  # right / outlet
                flags_1_up = [62] * (n_x_17_0 + n_x_0_1 + n_x_1_2 + n_x_2_3 - 1) + [11] * (
                            len(c3_low[0]) - n_x_17_0 - n_x_0_1 -
                            n_x_1_2 - n_x_2_3 - n_x_4_5 - n_x_5_6 + 1) + [71] \
                             + [62] * (
                                         n_x_4_5 + n_x_5_6 - 1)  # top (will be inside of mesh)                # bottom (will be inside of mesh)
                flags_2_up = [2] * len(c2_up[0])  # left / inlet
                flags_3_up = [4] * len(c3_up[0])  # top
                flags_4_up = [3] * len(c4_up[0])  # right / outlet
                boundary_flags = [flags_1_low, flags_2_low, flags_3_low, flags_4_low, flags_1_up, flags_2_up,
                                  flags_3_up, flags_4_up]

                n_space = [self.n_x, n_r_geom_low, self.n_x, len(spacing_left_up)]
                X_discret = [
                    c1_low[0][
                    int(np.where(c1_low[0] == min(self.Xn[0]))[0]):int(np.where(c1_low[0] == max(self.Xn[0]))[0]) + 1],
                    np.flip(spacing_nacelle_bottom), spacing_nacelle_top]
                Y_discret = [
                    c1_low[1][
                    int(np.where(c1_low[0] == min(self.Xn[0]))[0]):int(np.where(c1_low[0] == max(self.Xn[0]))[0]) + 1],
                    Fs[2](np.flip(spacing_nacelle_bottom)), Fs[1](spacing_nacelle_top)]


        ### START: OLD, currently unused ###
        elif self.grid_type == 'slab':
            # calculate number of points in front of and behind whole geometry
            n_x_front = min(int(round(self.n_x * self.ext_front / (self.ext_front + self.ext_rear + 1), 0)), int(25))
            n_x_rear = 0  # min(int(round(self.n_x * self.ext_rear / (self.ext_front + self.ext_rear + 1), 0)), int(20))
            n_x_geom = self.n_x - n_x_rear - n_x_front
            # if n_x_geom < 80:
            #     if len(self.Xn) == 3:
            #         n_x_geom = 80
            #     elif len(self.Xn) == 1:
            #         n_x_geom = 25
            #     warnings.warn(f"Specified number of nodes too small. Increased to n_x={n_x_geom+n_x_rear+n_x_front}")
            self.n_x = n_x_geom + n_x_rear + n_x_front
            n_y_geom = int(round(0.3 * self.n_y))
            Fs_nose_rev = interpolate.UnivariateSpline(self.Yn[0][:99], self.Xn[0][:99], s=0)
            fuse_cent = Fs[0](x_geom_min + (x_geom_max - x_geom_min) / 2)
            x_fuse_cent = x_geom_min + (x_geom_max - x_geom_min) / 2
            r_fuse_cent = Fs[0](x_fuse_cent)

            spacing_low, _, _ = number_spacing_bump(0.99 * fuse_cent, 1.4 * fuse_cent, 2e-2, 1e-2, int((
                                                                                                                   self.n_y - n_y_geom) / 2))  # adapt upper boundary and next lower
            spacing_up, _ = number_spacing(1.4 * fuse_cent, r_grid_max, 1e-2, int((self.n_y - n_y_geom) / 2),
                                           1.4 * fuse_cent, 0)
            spacing_low_right, _, _ = number_spacing_bump(fuse_cent, 1.4 * fuse_cent, 2e-2, 1e-2,
                                                          int((self.n_y - n_y_geom) / 2))
            spacing_up_right, _ = number_spacing(1.4 * fuse_cent, r_grid_max + r_fuse_cent, 1e-2,
                                                 int((self.n_y - n_y_geom) / 2), 1.4 * fuse_cent, 0)
            # differentiate between geometry with one body (one subgrid) vs fuselage and nacelle bodies (two subgrids)

            x2 = self.Xn[0][self.Yn[0].index(max(self.Yn[0]))]

            x_a0 = x_grid_min
            x_a1 = x_geom_min - (x2 - self.Xn[0][0])
            x_a2 = x_a3 = x_geom_min
            x_a4 = x_geom_min + (x2 - self.Xn[0][0])
            x_a5 = x_fuse_cent
            x_a6 = x_geom_max - (x2 - self.Xn[0][0])
            x_a7 = x_a8 = x_geom_max
            x_a9 = x_geom_max + (x2 - self.Xn[0][0])
            x_a10 = x_fuse_cent

            y_a0 = y_a1 = y_a2 = y_a8 = y_a9 = y_a10 = r_grid_max
            y_a3 = y_a4 = y_a5 = y_a6 = y_a7 = r_fuse_cent + r_grid_max

            x_upper, y_upper = bezier_curve(
                [[x_a0, y_a0], [x_a1, y_a1], [x_a2, y_a2], [x_a3, y_a3], [x_a4, y_a4], [x_a5, y_a5]], nTimes=1000)
            Fs_upper = interpolate.UnivariateSpline(np.flip(x_upper), np.flip(y_upper), s=0)

            if len(self.Xn) == 1:
                c2 = np.array([np.concatenate((np.full((1, self.n_y), x_grid_min))),
                               np.concatenate((np.linspace(r_grid_min, 0.99 * fuse_cent, n_y_geom),
                                               spacing_low,
                                               spacing_up))])
                c1 = np.array([np.zeros((self.n_x + n_y_geom)), np.zeros((self.n_x + n_y_geom))])
                c1_dup = np.array([np.concatenate((np.linspace(x_grid_min, x_geom_min, n_x_front),
                                                   Fs_nose_rev(c2[1][1:n_y_geom + 1]),
                                                   np.linspace(Fs_nose_rev(c2[1][n_y_geom + 1]),
                                                               x_geom_min + (x_geom_max - x_geom_min) / 2,
                                                               n_x_geom + 1))),
                                   np.concatenate((np.zeros(n_x_front),
                                                   c2[1][1:n_y_geom + 1],
                                                   Fs[0](np.linspace(Fs_nose_rev(c2[1][n_y_geom + 1]),
                                                                     x_geom_min + (x_geom_max - x_geom_min) / 2,
                                                                     n_x_geom + 1))))])  # bottom (left to right)
                c3 = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c3_dup = np.array([np.concatenate((np.linspace(x_grid_min, x_geom_min, n_x_front),
                                                   np.linspace(Fs_nose_rev(c2[1][n_y_geom + 1]),
                                                               x_geom_min + (x_geom_max - x_geom_min) / 2, n_x_geom))),
                                   np.concatenate((Fs_upper(np.linspace(x_grid_min, x_geom_min, n_x_front)),
                                                   Fs_upper(np.linspace(Fs_nose_rev(c2[1][n_y_geom + 1]),
                                                                        x_geom_min + (x_geom_max - x_geom_min) / 2,
                                                                        n_x_geom))))])  # top (left to right)
                c4 = np.array(
                    [np.concatenate((np.full((1, self.n_y - n_y_geom), x_geom_min + (x_geom_max - x_geom_min) / 2))),
                     np.concatenate((spacing_low_right, spacing_up_right))])

                # identify duplicates and remove from arrays
                c1_un = np.unique(c1_dup[0], return_index=True)
                c1[0] = c1_un[0]
                c1[1] = c1_dup[1][c1_un[1]]
                c3_un = np.unique(c3_dup[0], return_index=True)
                c3[0] = c3_un[0]
                c3[1] = c3_dup[1][c3_un[1]]
                boundaries = [c1, c2, c3, c4]

                # flag matrix contains information about the node types
                # 0: inner point; 1: surface; 2: inlet; 3: outlet; 4: farfield; 5: center axis
                flags_1 = [5] * (n_x_front - 1) + [1] * (len(c1[0]) - n_x_front - n_x_rear + 2) + [5] * (n_x_rear - 1)
                flags_2 = [2] * len(c2[0])  # left / inlet
                flags_3 = [4] * len(c3[0])  # top
                flags_4 = [3] * len(c4[0])  # right / outlet
                boundary_flags = [flags_1, flags_2, flags_3, flags_4]
                n_space = [self.n_x, self.n_y]

            # todo: include jet (?)
            # todo: re-do inner boundary (some smoother way with normals maybe?)
            elif len(self.Xn) == 3:
                # lower subgrid index "low", upper subgrid index "up"
                # define points of upper and lower grid. index "bar" for inner points
                x1 = min(self.Xn[0])
                x2 = self.Xn[0][self.Yn[0].index(max(self.Yn[0]))]
                x0 = x1 - (x2 - x1)
                x5 = max(self.Xn[0])
                x6 = x_grid_max
                x7 = x8 = x6
                x9 = x5_bar = x5
                x15 = x_grid_min
                x16 = x17 = x15
                x0_bar = x14 = x0
                x1_bar = x13 = x1
                x2_bar = x12 = x2
                x3_bar = min(self.Xn[1])
                x4_bar = max(self.Xn[1])
                x3 = x3_bar
                x4 = x4_bar
                x10 = x4_bar
                x11 = x3_bar

                delta = self.Yn[1][self.Xn[1].index(x3_bar)] - Fs[0](x3_bar)

                y1 = r_grid_min
                y2 = Fs[0](x2)
                y3 = Fs[0](x3)
                y4 = Fs[0](x4)
                y0 = y5 = y6 = y17 = y1
                y8 = r_grid_max
                y9 = y10 = y11 = y12 = y13 = y14 = y15 = y8
                y16 = r_grid_min + delta
                y0_bar = y0 + delta
                y2_bar = y2 + delta
                y3_bar = self.Yn[1][self.Xn[1].index(x3_bar)]
                y4_bar = self.Yn[1][self.Xn[1].index(x4_bar)]
                y5_bar = y7 = y4_bar

                # generate bezier curve for front part of separated grids
                xb1 = xb2 = x0 + 0.01 * (x1 - x0)
                yb1 = y0_bar
                yb2 = y2_bar
                bez = bspline.Curve()
                bez.degree = 3
                bez.ctrlpts = [[x0_bar, y0_bar], [xb1, yb1], [xb2, yb2], [x2_bar, y2_bar]]
                bez.knotvector = utilities.generate_knot_vector(bez.degree, len(bez.ctrlpts))

                bez.evaluate(start=(x1_bar - x0_bar) / (x2_bar - x0_bar))
                # y1_bar = bez.evalpts[0][1]

                # generate curve for mid part of separated grids
                Y_in = [self.Yn[0][i] + delta for i in range(len(self.Yn[0]))]
                Fs_in = interpolate.UnivariateSpline(self.Xn[0], Y_in, s=0)

                n_x_1_2 = max(int(5), int(round((x2 - x1) / (x5 - x1) * n_x_geom, 0)))
                n_x_3_4 = max(int(20), int(round((x4 - x3) / (x5 - x1) * n_x_geom, 0)))
                n_x_4_5 = max(int(5), int(round((x5 - x4) / (x5 - x1) * n_x_geom, 0)))
                n_x_2_3 = n_x_geom - n_x_1_2 - n_x_3_4 - n_x_4_5

                n_x_0_1 = max(int(5), int(round((x1 - x0) / (x1 - x17) * n_x_front, 0)))
                n_x_17_0 = n_x_front - n_x_0_1
                n_x_5_6 = n_x_rear

                if n_x_1_2 < 0 or n_x_3_4 < 0 or n_x_4_5 < 0 or n_x_2_3 < 0 or n_x_0_1 < 0 or n_x_17_0 < 0 or n_x_5_6 < 0:
                    raise Warning("No. of x-nodes too small. Solution impossible.")

                n_r_geom_low = int(round((y16 - r_grid_min) / (r_grid_max - r_grid_min) * self.n_y, 0))
                n_r_geom_up = self.n_y - n_r_geom_low

                # initialize arrays without duplicates
                c1_low = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c1_up = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c3_low = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
                c3_up = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])

                bez.delta = 1 / (n_x_0_1 + 1)
                bez.evaluate(start=0, stop=(x1_bar - x0_bar) / (x2_bar - x0_bar))
                y_pts1 = [sublist[-1] for sublist in bez.evalpts]

                bez.delta = 1 / (n_x_1_2 + 2)
                bez.evaluate(start=(x1_bar - x0_bar) / (x2_bar - x0_bar), stop=1)
                y_pts2 = [sublist[-1] for sublist in bez.evalpts]

                c1_low_dup = np.array(
                    [np.concatenate((np.arange(x17, x0 + (x0 - x17) / (n_x_17_0 - 1) / 2, (x0 - x17) / (n_x_17_0 - 1)),
                                     np.arange(x0, x1 + (x1 - x0) / (n_x_0_1 + 1 - 1) / 2,
                                               (x1 - x0) / (n_x_0_1 + 1 - 1)),
                                     np.arange(x1, x2 + (x2 - x1) / (n_x_1_2 + 2 - 1) / 2,
                                               (x2 - x1) / (n_x_1_2 + 2 - 1)),
                                     np.arange(x2, x3 + (x3 - x2) / (n_x_2_3 - 1) / 2, (x3 - x2) / (n_x_2_3 - 1)),
                                     np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2,
                                               (x4 - x3) / (n_x_3_4 + 2 - 1)),
                                     np.arange(x4, x5 + (x5 - x4) / (n_x_4_5 + 1 - 1) / 2,
                                               (x5 - x4) / (n_x_4_5 + 1 - 1)),
                                     np.arange(x5, x6 + (x6 - x5) / (n_x_5_6 - 1) / 2, (x6 - x5) / (n_x_5_6 - 1)))),
                     np.concatenate((np.concatenate(np.full((1, n_x_17_0), y1)),
                                     np.concatenate(np.full((1, n_x_0_1 + 1), y1)),
                                     Fs[0](np.arange(x1, x2 + (x2 - x1) / (n_x_1_2 + 2 - 1) / 2,
                                                     (x2 - x1) / (n_x_1_2 + 2 - 1))),
                                     Fs[0](
                                         np.arange(x2, x3 + (x3 - x2) / (n_x_2_3 - 1) / 2, (x3 - x2) / (n_x_2_3 - 1))),
                                     Fs[0](np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2,
                                                     (x4 - x3) / (n_x_3_4 + 2 - 1))),
                                     Fs[0](np.arange(x4, x5 + (x5 - x4) / (n_x_4_5 + 1 - 1) / 2,
                                                     (x5 - x4) / (n_x_4_5 + 1 - 1))),
                                     np.concatenate(np.full((1, n_x_5_6), y5))))])  # lower bottom (left to right)
                c2_low = np.array([np.concatenate((np.full((1, n_r_geom_low), x16))),
                                   np.arange(y16, y17 + (y17 - y16) / (n_r_geom_low - 1),
                                             (y17 - y16) / (n_r_geom_low - 1))])  # lower left (top to bottom)
                c3_low_dup = np.array(
                    [np.concatenate((np.arange(x17, x0 + (x0 - x17) / (n_x_17_0 - 1) / 2, (x0 - x17) / (n_x_17_0 - 1)),
                                     np.arange(x0, x1 + (x1 - x0) / (n_x_0_1 + 1 - 1) / 2,
                                               (x1 - x0) / (n_x_0_1 + 1 - 1)),
                                     np.arange(x1, x2 + (x2 - x1) / (n_x_1_2 + 2 - 1) / 2,
                                               (x2 - x1) / (n_x_1_2 + 2 - 1)),
                                     np.arange(x2, x3 + (x3 - x2) / (n_x_2_3 - 1) / 2, (x3 - x2) / (n_x_2_3 - 1)),
                                     np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2,
                                               (x4 - x3) / (n_x_3_4 + 2 - 1)),
                                     np.arange(x4, x5 + (x5 - x4) / (n_x_4_5 + 1 - 1) / 2,
                                               (x5 - x4) / (n_x_4_5 + 1 - 1)),
                                     np.arange(x5, x6 + (x6 - x5) / (n_x_5_6 - 1) / 2, (x6 - x5) / (n_x_5_6 - 1)))),
                     np.concatenate((np.concatenate(np.full((1, n_x_17_0) / 2, y16)),
                                     np.array(y_pts1),
                                     np.array(y_pts2),
                                     Fs_in(
                                         np.arange(x2, x3 + (x3 - x2) / (n_x_2_3 - 1) / 2, (x3 - x2) / (n_x_2_3 - 1))),
                                     Fs[2](np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2,
                                                     (x4 - x3) / (n_x_3_4 + 2 - 1))),
                                     np.concatenate(np.full((1, n_x_4_5 + 1), y7)),
                                     np.concatenate(np.full((1, n_x_5_6),
                                                            y7))))])  # lower top (left to right)                                                                                             # lower top (left to right)
                c4_low = np.array([np.concatenate((np.full((1, n_r_geom_low), x6))),
                                   np.arange(y7, y6 + (y6 - y7) / (n_r_geom_low - 1),
                                             (y6 - y7) / (n_r_geom_low - 1))])  # lower right (top to bottom)

                c1_up_dup = np.array(
                    [np.concatenate((np.arange(x17, x0 + (x0 - x17) / (n_x_17_0 - 1) / 2, (x0 - x17) / (n_x_17_0 - 1)),
                                     np.arange(x0, x1 + (x1 - x0) / (n_x_0_1 + 1 - 1) / 2,
                                               (x1 - x0) / (n_x_0_1 + 1 - 1)),
                                     np.arange(x1, x2 + (x2 - x1) / (n_x_1_2 + 2 - 1) / 2,
                                               (x2 - x1) / (n_x_1_2 + 2 - 1)),
                                     np.arange(x2, x3 + (x3 - x2) / (n_x_2_3 - 1) / 2, (x3 - x2) / (n_x_2_3 - 1)),
                                     np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2,
                                               (x4 - x3) / (n_x_3_4 + 2 - 1)),
                                     np.arange(x4, x5 + (x5 - x4) / (n_x_4_5 + 1 - 1) / 2,
                                               (x5 - x4) / (n_x_4_5 + 1 - 1)),
                                     np.arange(x5, x6 + (x6 - x5) / (n_x_5_6 - 1) / 2, (x6 - x5) / (n_x_5_6 - 1)))),
                     np.concatenate((np.concatenate(np.full((1, n_x_17_0) / 2, y16)),
                                     np.array(y_pts1),
                                     np.array(y_pts2),
                                     Fs_in(
                                         np.arange(x2, x3 + (x3 - x2) / (n_x_2_3 - 1) / 2, (x3 - x2) / (n_x_2_3 - 1))),
                                     Fs[1](np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2,
                                                     (x4 - x3) / (n_x_3_4 + 2 - 1))),
                                     np.concatenate(np.full((1, n_x_4_5 + 1), y7)),
                                     np.concatenate(np.full((1, n_x_5_6), y7))))])  # lower top (left to right)
                c2_up = np.array([np.concatenate((np.full((1, n_r_geom_up), x16))),
                                  np.arange(y16, r_grid_max + (r_grid_max - y16) / (n_r_geom_up - 1),
                                            (r_grid_max - y16) / (n_r_geom_up - 1))])  # upper left (top to bottom)
                c3_up_dup = np.array(
                    [np.concatenate((np.arange(x17, x0 + (x0 - x17) / (n_x_17_0 - 1) / 2, (x0 - x17) / (n_x_17_0 - 1)),
                                     np.arange(x0, x1 + (x1 - x0) / (n_x_0_1 + 1 - 1) / 2,
                                               (x1 - x0) / (n_x_0_1 + 1 - 1)),
                                     np.arange(x1, x2 + (x2 - x1) / (n_x_1_2 + 2 - 1) / 2,
                                               (x2 - x1) / (n_x_1_2 + 2 - 1)),
                                     np.arange(x2, x3 + (x3 - x2) / (n_x_2_3 - 1) / 2, (x3 - x2) / (n_x_2_3 - 1)),
                                     np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2,
                                               (x4 - x3) / (n_x_3_4 + 2 - 1)),
                                     np.arange(x4, x5 + (x5 - x4) / (n_x_4_5 + 1 - 1) / 2,
                                               (x5 - x4) / (n_x_4_5 + 1 - 1)),
                                     np.arange(x5, x6 + (x6 - x5) / (n_x_5_6 - 1) / 2, (x6 - x5) / (n_x_5_6 - 1)))),
                     np.concatenate((np.concatenate(np.full((1, n_x_17_0), y15)),
                                     np.concatenate(np.full((1, n_x_0_1 + 1), y15)),
                                     np.concatenate(np.full((1, n_x_17_0), y15)),
                                     np.concatenate(np.full((1, n_x_1_2 + 2), y15)),
                                     np.concatenate(np.full((1, n_x_2_3), y15)),
                                     np.concatenate(np.full((1, n_x_3_4 + 2), y15)),
                                     np.concatenate(np.full((1, n_x_4_5 + 1), y15)),
                                     np.concatenate(np.full((1, n_x_5_6), y15))))])  # upper top (left to right)
                c4_up = np.array([np.concatenate((np.full((1, n_r_geom_up), x6))),
                                  np.arange(r_grid_max, y7 + (y7 - r_grid_max) / (n_r_geom_up - 1) / 2,
                                            (y7 - r_grid_max) / (n_r_geom_up - 1))])  # upper right (top to bottom)

                # identify duplicates and remove from arrays
                c1_low_un = np.unique(c1_low_dup[0], return_index=True)
                c1_low[0] = c1_low_un[0]
                c1_low[1] = c1_low_dup[1][c1_low_un[1]]
                c3_low_un = np.unique(c3_low_dup[0], return_index=True)
                c3_low[0] = c3_low_un[0]
                c3_low[1] = c3_low_dup[1][c3_low_un[1]]
                c1_up_un = np.unique(c1_up_dup[0], return_index=True)
                c1_up[0] = c1_up_un[0]
                c1_up[1] = c1_up_dup[1][c1_up_un[1]]
                c3_up_un = np.unique(c3_up_dup[0], return_index=True)
                c3_up[0] = c3_up_un[0]
                c3_up[1] = c3_up_dup[1][c3_up_un[1]]
                boundaries = [c1_low, c2_low, c3_low, c4_low, c1_up, c2_up, c3_up, c4_up]

                # flag matrix contains information about the node types
                # 0: inner point; 1: surface; 2: inlet; 3: outlet; 4: farfield; 5: center axis
                flags_1_low = [5] * (n_x_17_0 + n_x_0_1 - 1) + [1] * (
                            len(c1_low[0]) - n_x_17_0 - n_x_0_1 - n_x_5_6 + 2) + [5] * (n_x_5_6 - 1)  # bottom
                flags_2_low = [2] * len(c2_low[0])  # left / inlet
                flags_3_low = [0] * (n_x_17_0 + n_x_0_1 + n_x_1_2 + n_x_2_3 - 1) + [1] * (
                            len(c3_low[0]) - n_x_17_0 - n_x_0_1 - n_x_1_2 - n_x_2_3 - n_x_4_5 - n_x_5_6 + 2) + [0] * (
                                          n_x_4_5 + n_x_5_6 - 1)  # top (will be inside of mesh)
                flags_4_low = [3] * len(c4_low[0])  # right / outlet
                flags_1_up = flags_3_low  # bottom (will be inside of mesh)
                flags_2_up = [2] * len(c2_up[0])  # left / inlet
                flags_3_up = [4] * len(c3_up[0])  # top
                flags_4_up = [3] * len(c4_up[0])  # right / outlet
                boundary_flags = [flags_1_low, flags_2_low, flags_3_low, flags_4_low, flags_1_up, flags_2_up,
                                  flags_3_up, flags_4_up]
                n_space = [self.n_x, n_r_geom_low, self.n_x, n_r_geom_up]
                X_discret = [
                    c1_low[0][
                    int(np.where(c1_low[0] == min(self.Xn[0]))[0]):int(np.where(c1_low[0] == max(self.Xn[0]))[0]) + 1],
                    np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2, (x4 - x3) / (n_x_3_4 + 2 - 1)),
                    np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2, (x4 - x3) / (n_x_3_4 + 2 - 1))]
                Y_discret = [
                    c1_low[1][
                    int(np.where(c1_low[0] == min(self.Xn[0]))[0]):int(np.where(c1_low[0] == max(self.Xn[0]))[0]) + 1],
                    Fs[2](np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2, (x4 - x3) / (n_x_3_4 + 2 - 1))),
                    Fs[1](np.arange(x3, x4 + (x4 - x3) / (n_x_3_4 + 2 - 1) / 2, (x4 - x3) / (n_x_3_4 + 2 - 1)))]
            else:
                raise Warning("This type of grid generation is not specified for the number of geometrical boundaries.")

        elif self.grid_type == 'slit' and len(self.Xn) == 3:
            # calculate number of points in front of and behind whole geometry
            n_x_front = min(int(round(self.n_x * self.ext_front / (self.ext_front + self.ext_rear + 1), 0)), int(25))
            n_x_rear = min(int(round(self.n_x * self.ext_rear / (self.ext_front + self.ext_rear + 1), 0)), int(20))
            n_x_geom = self.n_x - n_x_rear - n_x_front
            if n_x_geom < 50:
                if len(self.Xn) == 3:
                    n_x_geom = 50
                elif len(self.Xn) == 1:
                    n_x_geom = 25
                warnings.warn(
                    f"Specified number of nodes too small. Increased to n_x={n_x_geom + n_x_rear + n_x_front}")
            self.n_x = n_x_geom + n_x_rear + n_x_front
            x2 = self.Xn[0][self.Yn[0].index(max(self.Yn[0]))]
            n_x_1_2 = max(int(8), int(round((x2 - x_geom_min) / (x_geom_max - x_geom_min) * n_x_geom, 0)))
            x_fuse_cent = x_geom_min + (x_geom_max - x_geom_min) / 2
            r_fuse_cent = Fs[0](x_fuse_cent)

            # spacing_nose, _ = paramSampling(np.linspace(x_geom_min, x2, 1000), Fs[0], n_x_1_2 + 1, 0.3, 0)#np.arange(x_geom_min, x2+(x2-x_geom_min)/1000/2, (x2-x_geom_min)/1000), Fs[0], n_x_1_2 + 1, 0.3, 0)
            samples_rest, _ = paramSampling(np.linspace(x2, x_geom_max, 1000), Fs[0], n_x_geom - n_x_1_2 + 2, 0.3,
                                            0)  # np.arange(x2, x_geom_max+(x_geom_max-x2)/1000/2, (x_geom_max-x2)/1000), Fs[0], n_x_geom-n_x_1_2+2, 0.3, 0)

            x1 = min(self.Xn[0])
            x2 = self.Xn[0][self.Yn[0].index(max(self.Yn[0]))]
            x0 = x1 - (x2 - x1)
            x5 = max(self.Xn[0])
            x6 = x_grid_max
            x7 = x8 = x6
            x9 = x5_bar = x5
            x15 = x_grid_min
            x16 = x17 = x15
            x0_bar = x14 = x0
            x1_bar = x13 = x1
            x2_bar = x12 = x2
            x3_bar = min(self.Xn[1])
            x4_bar = max(self.Xn[1])
            x3 = x3_bar
            x4 = x4_bar
            x10 = x4_bar
            x11 = x3_bar
            x0_bar = x0 - 0.15 * (x1 - x0)  # +0.5*(x1-x0)

            delta = self.Yn[1][self.Xn[1].index(x3_bar)] - Fs[0](x3_bar)

            y1 = r_grid_min
            y2 = Fs[0](x2)
            y3 = Fs[0](x3)
            y4 = Fs[0](x4)
            y0 = y5 = y6 = y17 = y1
            y8 = r_grid_max
            y9 = y10 = y11 = y12 = y13 = y14 = y15 = y8
            y2_bar = y2 + delta
            y3_bar = self.Yn[1][self.Xn[1].index(x3_bar)]
            y4_bar = self.Yn[1][self.Xn[1].index(x4_bar)]
            y5_bar = y7 = y4_bar
            y16 = 0.8 * y7  # r_grid_min + delta#y7
            y0_bar = y16  # y0 + delta#y7

            # generate bezier curve for front part of separated grids
            xb1 = x0_bar + 0.5 * (x2_bar - x0_bar)
            xb2 = x2_bar - 0.4 * (x2_bar - x0_bar)
            yb1 = y0_bar
            yb2 = y2_bar
            x_c = x0_bar + 0.5 * (x2_bar - x0_bar)
            y_c = 0.7 * (y2_bar - y0_bar) + y0_bar

            x_bez, y_bez = bezier_curve([[x0_bar, y0_bar], [xb1, yb1], [x_c, y_c], [xb2, yb2], [x2_bar, y2_bar]],
                                        nTimes=1000)
            Fs_bez = interpolate.UnivariateSpline(np.flip(x_bez), np.flip(y_bez), s=0)

            # generate curve for mid part of separated grids
            Y_in = [self.Yn[0][i] + delta for i in range(len(self.Yn[0]))]
            Fs_in = interpolate.UnivariateSpline(self.Xn[0], Y_in, s=0)

            x_a1 = x_geom_max - (x2 - self.Xn[0][0])
            y_a1 = r_geom_max + r_grid_max
            x_a2 = x_geom_max
            y_a2 = y_a1
            x_a3 = x_a2
            y_a3 = r_grid_max
            x_a4 = x_geom_max + (x2 - self.Xn[0][0])
            y_a4 = y_a3
            x_a0 = x_fuse_cent
            y_a0 = y_a1
            x_a5 = x_grid_max
            y_a5 = y_a4

            x_upper, y_upper = bezier_curve(
                [[x_a0, y_a0], [x_a1, y_a1], [x_a2, y_a2], [x_a3, y_a3], [x_a4, y_a4], [x_a5, y_a5]], nTimes=1000)
            Fs_upper = interpolate.UnivariateSpline(np.flip(x_upper), np.flip(y_upper), s=0)

            n_x_1_2 = max(int(8), int(round((x2 - x1) / (x5 - x1) * n_x_geom, 0)))
            n_x_3_4 = max(int(20), int(round((x4 - x3) / (x5 - x1) * n_x_geom, 0)))
            n_x_4_5 = max(int(10), int(round((x5 - x4) / (x5 - x1) * n_x_geom, 0)))
            n_x_2_3 = n_x_geom - n_x_1_2 - n_x_3_4 - n_x_4_5

            n_x_0_1 = 10  # max(int(14), int(round((x1 - x0) / (x1 - x17) * n_x_front, 0)))
            n_x_17_0 = n_x_front - n_x_0_1
            n_x_5_6 = n_x_rear

            if n_x_1_2 < 0 or n_x_3_4 < 0 or n_x_4_5 < 0 or n_x_2_3 < 0 or n_x_0_1 < 0 or n_x_17_0 < 0 or n_x_5_6 < 0:
                raise Warning("No. of x-nodes too small. Solution impossible.")

            n_r_geom_low = int(round(1 / 2 * self.n_y))
            n_r_geom_up = self.n_y - n_r_geom_low

            x_nacelle = np.linspace(x3, x4, 1000)
            spacing_nacelle_bottom, _ = paramSampling(x_nacelle, Fs[2], n_x_3_4 + 2, 1, 1)
            # ensure that FF stations are part of sample
            spacing_nacelle_bottom = find_nearest_replace(spacing_nacelle_bottom, self.fan_stations)
            spacing_nacelle_top = np.flip(spacing_nacelle_bottom)
            self.n_x += (len(spacing_nacelle_bottom) - 2 - n_x_3_4)
            n_x_3_4 = len(spacing_nacelle_bottom) - 2
            spacing_nose, _ = number_spacing(x1, x2, 1e-2, n_x_1_2 + 2, x1,
                                             0)  # paramSampling(np.linspace(x1,x2,1000),Fs[0](np.linspace(x1,x2,1000)), n_x_1_2 + 2, 0.1, 0)#number_spacing(x1, x2, 0.08, n_x_1_2 + 2, x1, 0)
            spacing_front, _ = number_spacing(x0_bar, x1, spacing_nose[1] - spacing_nose[0], n_x_0_1 + 1, x1, 0)
            spacing_front0, _ = number_spacing(x17, x0_bar, spacing_front[1] - spacing_front[0], n_x_17_0, x0_bar, 0)

            spacing_tail, _ = number_spacing(x4, x5, spacing_nacelle_bottom[-2] - spacing_nacelle_bottom[-1],
                                             n_x_4_5 + 1,
                                             x4)  # spacing_tail, _ = paramSampling(np.linspace(x4,x5,1000),Fs[0](np.linspace(x4,x5,1000)), n_x_4_5+1, 0.5, 0)
            spacing_back = number_spacing(x5, x6, spacing_tail[-1] - spacing_tail[-2], n_x_5_6, x5, 0)
            spacing_centre, _ = number_spacing(x_fuse_cent, x3,
                                               np.abs(spacing_nacelle_bottom[-1] - spacing_nacelle_bottom[-2]), n_x_2_3,
                                               x3, n_x_2_3 - 1)
            # calculate height of first cell based on y+
            if self.calc_first_cell == True:
                fuse_firstcell = first_cell_height_y_plus(self.first_cell_values[0], self.first_cell_values[1],
                                                          self.first_cell_values[2],
                                                          np.abs(self.Xn[0][-1] - self.Xn[0][0]))
                nac_firstcell = first_cell_height_y_plus(self.first_cell_values[0], self.first_cell_values[1],
                                                         self.first_cell_values[2],
                                                         np.abs(self.Xn[1][-1] - self.Xn[1][0]))
            else:
                fuse_firstcell = 1e-2
                nac_firstcell = 1e-2

            spacing_left_low, ratio0, ratio1 = number_spacing_bump(Fs[0](x_fuse_cent), Fs_in(x_fuse_cent),
                                                                   fuse_firstcell, nac_firstcell, n_r_geom_low + 1)
            spacing_left_up, ratio2 = number_spacing(Fs_in(x_fuse_cent), r_grid_max + r_geom_max,
                                                     spacing_left_low[-1] - spacing_left_low[-2], n_r_geom_up,
                                                     Fs_in(x_fuse_cent), 0)

            spacing_right_low, ratio_right_1, ratio_right_2 = number_spacing_bump(y6, y7, 1e-2, 1e-2, n_r_geom_low + 1)
            spacing_right_up = \
                number_spacing(y7, r_grid_max, spacing_left_up[1] - spacing_left_up[0], len(spacing_left_up), y7, 0)[0]
            y_pts1 = Fs_bez(spacing_front)
            y_pts2 = Fs_bez(spacing_nose[1:])

            def translate_points(x_node, x_low_old, x_up_old, x_low_new, x_up_new):
                return (x_node - x_low_old) / (x_up_old - x_low_old) * (x_up_new - x_low_new) + x_low_new

            spacing_front_inner = np.array([translate_points(spacing_front[i], spacing_front[0], spacing_front[-1],
                                                             spacing_front[0], spacing_front[-1] - (0) * (x1 - x0_bar))
                                            for i in
                                            range(0, len(spacing_front))])  # spacing_front[-1]-(1/10)*(x1-x0_bar)
            spacing_nose_inner = np.array([translate_points(spacing_nose[i], spacing_nose[0], spacing_nose[-1],
                                                            spacing_nose[0] - (0) * (x1 - x0_bar), spacing_nose[-1]) for
                                           i in range(0, len(spacing_nose))])  # spacing_nose[0]-(1/10)*(x1-x0_bar)

            c1 = np.array([np.zeros((self.n_x - n_x_1_2 - n_x_front)), np.zeros((self.n_x - n_x_1_2 - n_x_front))])
            c1_dup = np.array([np.concatenate((spacing_centre,
                                               np.flip(spacing_nacelle_bottom[1:]),
                                               spacing_tail,
                                               spacing_back[0])),
                               np.concatenate((Fs[0](spacing_centre),
                                               Fs[0](np.flip(spacing_nacelle_bottom[1:])),
                                               Fs[0](spacing_tail),
                                               np.concatenate(
                                                   np.full((1, n_x_5_6), y5))))])  # bottom (left to right)

            c2 = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
            c2_dup = np.array([np.concatenate((np.concatenate(np.full((1, n_r_geom_low + 1), x_fuse_cent)),
                                               np.concatenate(np.full((1, len(spacing_left_up)), x_fuse_cent)))),
                               np.concatenate((spacing_left_low,
                                               spacing_left_up))])
            c3 = np.array([np.zeros((self.n_x - n_x_1_2 - n_x_front)), np.zeros((self.n_x - n_x_1_2 - n_x_front))])
            c3_dup = np.array([np.concatenate((spacing_centre,
                                               spacing_nacelle_top[1:],
                                               spacing_tail,
                                               spacing_back[0])),
                               np.concatenate((Fs_upper(spacing_centre),
                                               Fs_upper(spacing_nacelle_top[1:]),
                                               Fs_upper(spacing_tail),
                                               Fs_upper(spacing_back[0])))])  # top (left to right)
            c4 = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
            c4_dup = np.array([np.concatenate((np.concatenate(np.full((1, n_r_geom_low + 1), x6)),
                                               np.concatenate(np.full((1, len(spacing_left_up)), x6)))),
                               np.concatenate((spacing_right_low,
                                               spacing_right_up))])

            c5 = np.array([np.zeros((self.n_x - n_x_1_2 - n_x_front)), np.zeros((self.n_x - n_x_1_2 - n_x_front))])
            c5_dup = np.array([np.concatenate((
                spacing_centre,
                np.flip(spacing_nacelle_bottom[1:]),
                spacing_tail,
                spacing_back[0])),
                np.concatenate((Fs_in(spacing_centre),  # np.linspace(x2, x3, n_x_2_3)),
                                Fs[2](np.flip(spacing_nacelle_bottom[1:])),
                                np.concatenate(np.full((1, n_x_4_5 + 1), y7)),
                                np.concatenate(np.full((1, n_x_5_6),
                                                       y7))))])  # bottom (left to right)
            c6 = np.array([np.zeros((self.n_x - n_x_1_2 - n_x_front)), np.zeros((self.n_x - n_x_1_2 - n_x_front))])
            c6_dup = np.array([np.concatenate((spacing_centre,
                                               spacing_nacelle_top,
                                               spacing_tail,
                                               spacing_back[0])),
                               np.concatenate((Fs_in(spacing_centre),  # np.linspace(x2, x3, n_x_2_3)),
                                               Fs[1](spacing_nacelle_top),
                                               np.concatenate(np.full((1, n_x_4_5 + 1), y7)),
                                               np.concatenate(
                                                   np.full((1, n_x_5_6), y7))))])  # lower top (left to right)
            # identify duplicates and remove from arrays
            c1_un = np.unique(c1_dup[0], return_index=True)
            c1[0] = c1_un[0]
            c1[1] = c1_dup[1][c1_un[1]]
            # c1[1][24]=0.2
            # c1[1][23]=0.1
            # c1[1][22]=0.05
            # c1[1][21]=0.025
            c3_un = np.unique(c3_dup[0], return_index=True)
            c3[0] = c3_un[0]
            c3[1] = c3_dup[1][c3_un[1]]
            c2_un = np.unique(c2_dup[1], return_index=True)
            c2[1] = c2_un[0]
            c2[0] = c2_dup[0][c2_un[1]]
            c4_un = np.unique(c4_dup[1], return_index=True)
            c4[1] = c4_un[0]
            c4[0] = c4_dup[0][c4_un[1]]
            c5_un = np.unique(c5_dup[0], return_index=True)
            c5[0] = c5_un[0]
            c5[1] = c5_dup[1][c5_un[1]]
            c6_un = np.unique(c6_dup[0], return_index=True)
            c6[0] = c6_un[0]
            c6[1] = c6_dup[1][c6_un[1]]
            boundaries = [c1, c2, c3, c4, c5, c6]
            n_space = [self.n_x, self.n_y]

            # flag matrix contains information about the node types
            # 0: inner point; 1: surface; 2: inlet; 3: outlet; 4: farfield; 5: center axis
            flags_1 = [5] * (n_x_17_0 + n_x_0_1 - 1) + [1] * (len(c1[0]) - n_x_17_0 - n_x_0_1 - n_x_5_6 + 2) + [5] * (
                        n_x_5_6 - 1)  # bottom
            flags_2 = [2] * len(c2[0])  # left / inlet
            flags_3 = [4] * len(c3[0])  # top
            flags_4 = [3] * len(c4[0])  # right / outlet
            flags_5 = [0] * (n_x_1_2 + n_x_2_3 - 1) + [1] * (len(c3[0]) - n_x_1_2 - n_x_2_3 - n_x_4_5 - n_x_5_6 + 1) + [
                0] * (n_x_4_5 + n_x_5_6)  # lower slit
            flags_6 = flags_5  # upper slit
            boundary_flags = [flags_1, flags_2, flags_3, flags_4, flags_5, flags_6]

            X_discret = [0]
            Y_discret = [0]

        elif self.grid_type == 'slit' and len(self.Xn) == 2:
            # calculate number of points in front of and behind whole geometry
            n_x_front = min(int(round(self.n_x * self.ext_front / (self.ext_front + self.ext_rear + 1), 0)), int(25))
            n_x_rear = min(int(round(self.n_x * self.ext_rear / (self.ext_front + self.ext_rear + 1), 0)), int(25))

            r_grid_min = min(list(map(min, self.Yn))) - r_geom_max * (1 + self.ext_rad)
            r_grid_max = max(list(map(max, self.Yn))) + r_geom_max * (1 + self.ext_rad)
            n_x_geom = self.n_x - n_x_rear - n_x_front
            if n_x_geom < 25:
                n_x_geom = 25
                warnings.warn(
                    f"Specified number of nodes too small. Increased to n_x={n_x_geom + n_x_rear + n_x_front}")
            self.n_x = n_x_geom + n_x_rear + n_x_front

            n_y_nacelle = int(round(self.n_y / 2, 0))

            spacing_nacelle_top, _ = paramSampling(np.linspace(x_geom_min, x_geom_max, 1000), Fs[0],
                                                   n_x_geom + 2, 1, 2)
            spacing_nacelle_bottom, _ = paramSampling(np.linspace(x_geom_min, x_geom_max, 1000), Fs[1],
                                                      n_x_geom + 2, 1, 1)

            front = number_spacing(x_geom_min, x_grid_min, spacing_nacelle_bottom[1] - spacing_nacelle_bottom[0],
                                   n_x_front, x_geom_min)
            rear = number_spacing(x_geom_max, x_grid_max, spacing_nacelle_bottom[-2] - spacing_nacelle_bottom[-1],
                                  n_x_rear, x_geom_max)

            # calculate height of first cell based on y+
            if self.calc_first_cell == True:
                nac_firstcell = first_cell_height_y_plus(self.first_cell_values[0], self.first_cell_values[1],
                                                         self.first_cell_values[2],
                                                         np.abs(self.Xn[1][-1] - self.Xn[1][0]))
            else:
                nac_firstcell = 1e-1  # 0.01

            c1 = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
            c1_dup = np.array([np.concatenate((front[0],
                                               np.flip(spacing_nacelle_bottom),
                                               rear[0])),
                               np.concatenate((np.full((1, n_x_front + n_x_geom + 2 + n_x_rear),
                                                       r_grid_min)))])  # bottom (left to right)
            c2 = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
            c2_dup = tuple(np.array([np.concatenate((np.full((1, self.n_y), x_grid_min))),
                                     # np.linspace(r_grid_min, r_grid_max, self.n_y)]))   # left (top to bottom)
                                     number_spacing(r_grid_min, r_grid_max, nac_firstcell, self.n_y,
                                                    Fs[1](min(self.Xn[0])), n_y_nacelle - 1)[0]]))
            c3 = np.array([np.zeros((self.n_x)), np.zeros((self.n_x))])
            c3_dup = np.array([np.concatenate((front[0],
                                               spacing_nacelle_top,
                                               rear[0])),
                               np.concatenate((np.full((1, self.n_x + 2), r_grid_max)))])  # top (left to right)
            c4 = np.array([np.zeros((self.n_y)), np.zeros((self.n_y))])
            c4_dup = tuple(np.array([np.concatenate((np.full((1, self.n_y), x_grid_max))),
                                     # np.linspace(r_grid_min, r_grid_max, self.n_y)]))   # right (top to bottom)
                                     number_spacing(r_grid_min, r_grid_max, nac_firstcell, self.n_y,
                                                    Fs[1](min(self.Xn[0])), n_y_nacelle - 1)[0]]))
            c5 = np.array([np.flip(spacing_nacelle_bottom),
                           Fs[1](np.flip(spacing_nacelle_bottom))])  # inner boundary (lower slit)
            c6 = np.array([spacing_nacelle_top,
                           Fs[0](spacing_nacelle_top)])  # upper boundary (lower slit)

            # identify duplicates and remove from arrays
            c1_un = np.unique(c1_dup[0], return_index=True)
            c1[0] = c1_un[0]
            c1[1] = c1_dup[1][c1_un[1]]
            c3_un = np.unique(c3_dup[0], return_index=True)
            c3[0] = c3_un[0]
            c3[1] = c3_dup[1][c3_un[1]]
            c2_un = np.unique(c2_dup[1], return_index=True)
            c2[1] = c2_un[0]
            c2[0] = c2_dup[0][c2_un[1]]
            c4_un = np.unique(c4_dup[1], return_index=True)
            c4[1] = c4_un[0]
            c4[0] = c4_dup[0][c4_un[1]]
            boundaries = [c1, c2, c3, c4, c5, c6]

            # flag matrix contains information about the node types
            # 0: inner point; 1: surface; 2: inlet; 3: outlet; 4: farfield; 5: center axis
            flags_1 = [4] * len(c1[0])  # bottom
            flags_2 = [2] * len(c2[0])  # left / inlet
            flags_3 = [4] * len(c3[0])  # top
            flags_4 = [3] * len(c4[0])  # right / outlet
            flags_5 = [1] * len(c5[0])  # slit
            flags_6 = [1] * len(c6[0])  # slit
            boundary_flags = [flags_1, flags_2, flags_3, flags_4, flags_5, flags_6]

            n_space = [self.n_x, self.n_y]
            X_discret = [c5[0], c6[0]]
            Y_discret = [c5[1], c6[1]]
        ### END: OLD, currently unused ###

        else:
            raise Warning("Mesh type not specified.")

        return boundaries, n_space, boundary_flags
