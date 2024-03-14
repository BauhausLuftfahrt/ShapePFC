"""Test the generation of meshes for a 2D planar nacelle (see below).

Author:  A. Habermann
"""


import numpy as np

# Own modules
from finite_differences.mesh.initialize_boundaries import InitBoundaries
from finite_differences.mesh.initialize_coordinates import InitCoordinates
from finite_differences.mesh.algebraic_grid import AlgebraicGrid
from finite_differences.mesh.rectangular_grid import RectGrid
from finite_differences.mesh.slab_grid import SlabGrid
from finite_differences.mesh.slit_grid import SlitGrid
from post_processing.finite_difference_post.plot_grid import plotBoundaries, plotGrid
from finite_differences.mesh.inner_boundaries import InnerBoundary
from geometry_generation.finite_difference_geometry.generate_fd_icst_geometry import GenerateGeomICST
import matplotlib.pyplot as plt


def gridgeneration_test(geometry: str, grid_type: str, geometry_type: str):
    if geometry == "Akron":
        # Geometry of Akron Airship
        delta = 0
        X = [
            [0, 0.05980176, 0.11960352, 0.17940528, 0.23920704, 0.2990088, 0.35881056, 0.41861232, 0.5980176, 0.8970264,
             1.1960352, 1.495044, 1.7940528, 2.0930616, 2.3920704, 2.6910792, 2.990088, 3.2890968, 3.5881056, 3.8871144,
             4.1861232, 4.485132, 4.7841408, 5.0831496, 5.3821584, 5.6811672, 5.980176]]
        X = [[X[0][i] + delta for i in range(0, len(X[0]))]]
        Y = [[0, 0.053821584, 0.125583696, 0.177611227, 0.219472459, 0.252961445, 0.281068272, 0.303792941, 0.360604613,
              0.422798443, 0.467051746, 0.485590291, 0.498148661, 0.504128837, 0.505324872, 0.505324872, 0.502932802,
              0.497550643, 0.485590291, 0.468845798, 0.444327077, 0.410240074, 0.366584789, 0.312165187, 0.243991181,
              0.165650875, 0]]
        settings = [90, 100, 2, 10, 5]  # settings = [105, 60, 2, 10, 5] (works well for rect-grid)
        firstcell = [3, 0.15, 2000]
        ratios = [1.2]
        domain = 'c-grid'

    elif geometry == "PFC fuselage":
        # PFC fuselage only
        CENTRELINE = GenerateGeomICST(0.9283, 6.09 / 2, 50.9, 0.465, 0.118, 4.341, 0.56, 0.51, 0.974, f_lint=0.3795,
                                      l_ff_stage=0.229,
                                      teta_f_cone=13.5, f_rho_le=0.67, f_l_nose=0.1975, ahi_athr=1.304,
                                      athr_a12=0.994, a18_a13=0.705, f_xthr=0.243, beta_ff_stage=-3.57, beta_te_up=15,
                                      beta_te_low=0, r_te_hub=0, f_r18hub=1.32, f_rthrtip=0.898, teta_int_in=9.4)
        fuselage, nacelle_top, nacelle_bottom, l_fuse, f_slr, tc_max, tc_max_x, c_nac, i_nac, teta_f_aft, Astar_A2, x_thr, x_12, x_13 \
            = CENTRELINE.build_geometry()
        delta = 0
        X = [[fuselage[i][0] + delta for i in range(0, len(fuselage))]]
        Y = [[fuselage[i][1] for i in range(0, len(fuselage))]]
        firstcell = [1, 0.7, 10000]
        ratios = [1.2]  # boundary layer cell width ratio for every single surface
        settings = [200, 80, 3, 5, 5]  # 30 for rect
        domain = 'c-grid'

    elif geometry == "PFC total":
        # PFC Geometry
        CENTRELINE = GenerateGeomICST(0.9283, 6.09 / 2, 50.9, 0.465, 0.118, 4.341, 0.56, 0.51, 0.974, f_lint=0.3795,
                                      l_ff_stage=0.229,
                                      teta_f_cone=13.5, f_rho_le=0.67, f_l_nose=0.1975, ahi_athr=1.304,
                                      athr_a12=0.994, a18_a13=0.705, f_xthr=0.243, beta_ff_stage=-3.57, beta_te_up=15,
                                      beta_te_low=0, r_te_hub=0, f_r18hub=1.32, f_rthrtip=0.898, teta_int_in=9.4)
        fuselage, nacelle_top, nacelle_bottom, l_fuse, f_slr, tc_max, tc_max_x, c_nac, i_nac, teta_f_aft, Astar_A2, x_thr, x_12, x_13 \
            = CENTRELINE.build_geometry()

        delta = 0
        X = [[fuselage[i][0] + delta for i in range(0, len(fuselage))],
             [nacelle_top[i][0] + delta for i in range(0, len(nacelle_top))],
             [nacelle_bottom[i][0] + delta for i in reversed(range(0, len(nacelle_bottom)))]]
        Y = [[fuselage[i][1] for i in range(0, len(fuselage))], [nacelle_top[i][1] for i in range(0, len(nacelle_top))],
             [nacelle_bottom[i][1] for i in reversed(range(0, len(nacelle_bottom)))]]
        # for PFC it is required to identify important FF stations by their x-coordinate in the following order:
        # [x_thr, x_rot,in , x_rot,out , x_stat_in , x_stat,out]
        stations = [x_thr + delta, x_12 + delta, x_12 + 0.4 * (x_13 - x_12) + delta, x_12 + 0.6 * (x_13 - x_12) + delta,
                    x_13 + delta]
        settings = [110, 180, 3, 1, 4]  # 91 for slit; 90 for rect
        firstcell = [1, 0.7, 10000]
        ratios = [1.2, 1.2, 1.2]  # boundary layer cell width ratio for every single surface
    elif geometry == "PFC nacelle":
        # PFC Nacelle only
        CENTRELINE = GenerateGeomICST(0.9283, 6.09 / 2, 50.9, 0.465, 0.118, 4.341, 0.56, 0.51, 0.974, f_lint=0.3795,
                                      l_ff_stage=0.229,
                                      teta_f_cone=13.5, f_rho_le=0.67, f_l_nose=0.1975, ahi_athr=1.304,
                                      athr_a12=0.994, a18_a13=0.705, f_xthr=0.243, beta_ff_stage=-3.57, beta_te_up=15,
                                      beta_te_low=0, r_te_hub=0, f_r18hub=1.32, f_rthrtip=0.898, teta_int_in=9.4)
        fuselage, nacelle_top, nacelle_bottom, l_fuse, f_slr, tc_max, tc_max_x, c_nac, i_nac, teta_f_aft, Astar_A2, x_thr, x_12, x_13 \
            = CENTRELINE.build_geometry()
        from misc_functions.body_force_model.blade_camber import rotate
        nacelle_top_rot_x, nacelle_top_rot_y = rotate(nacelle_top[0][0], nacelle_top[0][1],
                                                      [nacelle_top[i][0] for i in range(0, len(nacelle_top))],
                                                      [nacelle_top[i][1] for i in range(0, len(nacelle_top))],
                                                      np.deg2rad(-5.06))
        nacelle_bottom_rot_x, nacelle_bottom_rot_y = rotate(nacelle_bottom[-1][0], nacelle_bottom[-1][1],
                                                            [nacelle_bottom[i][0] for i in
                                                             range(0, len(nacelle_bottom))],
                                                            [nacelle_bottom[i][1] for i in
                                                             range(0, len(nacelle_bottom))], np.deg2rad(-5.06))
        delta = 10
        X = [nacelle_top_rot_x, np.flip(nacelle_bottom_rot_x)]
        Y = [nacelle_top_rot_y, np.flip(nacelle_bottom_rot_y)]
        Y = [Y[i] + delta for i in range(0, len(Y))]

        settings = [60, 60, 1.5, 1.5, 2]
        firstcell = [1, 0.7, 10000]
        ratios = [1.2, 1.2]
        domain = 'c-grid'
    else:
        raise Warning("Geometry type not available.")

    n1 = settings[0]
    n2 = settings[1]
    front = settings[2]
    rear = settings[3]
    rad = settings[4]
    max_it = 1000  # max no. of iterations 50
    type_grid = grid_type
    type_geom = geometry_type

    if len(X) == 3:
        Bounds = InitBoundaries(X, Y, n1, n2, front, rear, rad, type_grid, calc_first_cell=True,
                                fan_stations=stations, first_cell_values=firstcell, bl_ratio=ratios)
    else:
        Bounds = InitBoundaries(X, Y, n1, n2, front, rear, rad, type_grid, calc_first_cell=False,
                                first_cell_values=firstcell, bl_ratio=ratios, grid_shape=domain)
    boundaries, spacing, boundary_flags = Bounds.run()

    # plotBoundaries(boundaries)
    # plt.show()

    # identify number of subgrids
    if type_grid == 'slit':
        n = int(len(boundaries) / 6)
    else:
        n = int(len(boundaries) / 4)

    x_subgrid = [0] * n
    y_subgrid = [0] * n
    x_subinit = [0] * n
    y_subinit = [0] * n
    node_flags_sub = [0] * n
    alpha_subgrid = [0] * n
    beta_subgrid = [0] * n
    gamma_subgrid = [0] * n
    tau_subgrid = [0] * n
    omega_subgrid = [0] * n
    jac_subgrid = [0] * n
    coeffs = [0] * n
    x_grid = []
    y_grid = []
    x_init = []
    y_init = []
    alpha = []
    beta = []
    gamma = []
    tau = []
    omega = []
    jac = []
    node_flags = []

    for i in range(0, n):
        if (type_grid == 'rect' and len(
                X) == 3 and i == 2):  # or (type_grid == 'rect' and len(X) == 2 and i == 1): # use upper adapted boundary of lower subgrid as lower boundary of upper subgrid
            InnerBound = InnerBoundary(x_subgrid[i - 1], y_subgrid[i - 1], boundaries[i * 4], X, Y)
            boundaries[i * 4] = InnerBound.run()  # lower boundary
            boundaries[i * 4 + 2][0] = boundaries[i * 4][0]  # x- values of upper boundary same as of lower boundary

        if type_grid == 'slit':
            Coords = InitCoordinates(boundaries[i * 6:i * 6 + 6], spacing[i * 2:i * 2 + 2], type_grid, X, front, rear,
                                     rad, boundary_flags[i * 6:i * 6 + 6])
        else:
            Coords = InitCoordinates(boundaries[i * 4:i * 4 + 4], spacing[i * 2:i * 2 + 2], type_grid, X, front, rear,
                                     rad, boundary_flags[i * 4:i * 4 + 4])
        x_bound, y_bound, node_flags_sub[i] = Coords.run()

        # initialize grid with algebraic grid generation
        InitGrid = AlgebraicGrid(x_bound, y_bound, type_grid, X, Y)
        x_subinit[i], y_subinit[i] = InitGrid.run()

        # plotGrid(x_subinit[i], y_subinit[i],node_flags_sub[i])
        # plt.show()

        if type_grid == 'rect' and len(X) == 3 and i == 0:  # front subgrid
            Grid = RectGrid(x_subinit[i], y_subinit[i], max_it, type_geom, 'c-grid')
        elif type_grid == 'rect' and len(X) == 3 and i == 1:  # lower subgrid
            Grid = RectGrid(x_subinit[i], y_subinit[i], max_it, type_geom, 'bottom', 'rect-grid', X[1])
        elif type_grid == 'rect' and len(X) == 2 and i == 0:  # lower subgrid
            Grid = RectGrid(x_subinit[i], y_subinit[i], max_it, type_geom, 'rect-grid')  # , 'bottom', X[0])
        elif type_grid == 'rect' and len(X) == 3 and i == 2:  # upper subgrid
            Grid = RectGrid(x_subinit[i], y_subinit[i], max_it, type_geom, 'top', 'c-grid', X[2])
        elif type_grid == 'rect' and len(X) == 2 and i == 1:  # upper subgrid
            Grid = RectGrid(x_subinit[i], y_subinit[i], max_it, type_geom, 'c-grid')  # , 'top', X[1])
        elif type_grid == 'rect' and len(X) == 1 and i == 0:
            Grid = RectGrid(x_subinit[i], y_subinit[i], max_it, type_geom, domain)
        elif type_grid == 'slab':
            Grid = SlabGrid(x_subinit[i], y_subinit[i], spacing[i * 2], spacing[i * 2 + 1], max_it, type_geom, front,
                            rear)
        elif type_grid == 'slit':
            Grid = SlitGrid(x_subinit[i], y_subinit[i], spacing[i * 2], spacing[i * 2 + 1], max_it, type_geom, front,
                            rear, X)
        else:
            raise Warning("Mesh type not specified.")

        x_subgrid[i], y_subgrid[i], alpha_subgrid[i], beta_subgrid[i], gamma_subgrid[i], tau_subgrid[i], \
        omega_subgrid[i], jac_subgrid[i], _ = Grid.run()

        node_flags.insert(0, node_flags_sub[i])
        x_grid.insert(0, x_subgrid[i])
        y_grid.insert(0, y_subgrid[i])
        x_init.insert(0, x_subinit[i])
        y_init.insert(0, y_subinit[i])
        alpha.insert(0, alpha_subgrid[i])
        beta.insert(0, beta_subgrid[i])
        gamma.insert(0, gamma_subgrid[i])
        omega.insert(0, omega_subgrid[i])
        tau.insert(0, tau_subgrid[i])
        jac.insert(0, jac_subgrid[i])

    # plotBoundaries(boundaries)
    # plt.show()

    if n == 3:
        x_grid_tot = np.zeros((np.shape(x_grid[2])[0] + 1, np.shape(x_grid[0])[1] + np.shape(x_grid[2])[1] - 1))
        x_grid_tot[0:np.shape(x_grid[0])[0], 0:np.shape(x_grid[2])[1]] = x_grid[2][0:np.shape(x_grid[0])[0], :]
        x_grid_tot[np.shape(x_grid[0])[0]:, 0:np.shape(x_grid[2])[1]] = x_grid[2][np.shape(x_grid[0])[0] - 1:, :]
        x_grid_tot[0:np.shape(x_grid[0])[0], np.shape(x_grid[2])[1] - 1:] = x_grid[0]
        x_grid_tot[np.shape(x_grid[0])[0]:, np.shape(x_grid[2])[1] - 1:] = x_grid[1]

        y_grid_tot = np.zeros((np.shape(y_grid[2])[0] + 1, np.shape(y_grid[0])[1] + np.shape(y_grid[2])[1] - 1))
        y_grid_tot[0:np.shape(y_grid[0])[0], 0:np.shape(y_grid[2])[1]] = y_grid[2][0:np.shape(y_grid[0])[0], :]
        y_grid_tot[np.shape(y_grid[0])[0]:, 0:np.shape(y_grid[2])[1]] = y_grid[2][np.shape(y_grid[0])[0] - 1:, :]
        y_grid_tot[0:np.shape(y_grid[0])[0], np.shape(y_grid[2])[1] - 1:] = y_grid[0]
        y_grid_tot[np.shape(y_grid[0])[0]:, np.shape(y_grid[2])[1] - 1:] = y_grid[1]

        x_init_tot = np.zeros((np.shape(x_init[2])[0] + 1, np.shape(x_init[0])[1] + np.shape(x_init[2])[1] - 1))
        x_init_tot[0:np.shape(x_init[0])[0], 0:np.shape(x_init[2])[1]] = x_init[2][0:np.shape(x_init[0])[0], :]
        x_init_tot[np.shape(x_init[0])[0]:, 0:np.shape(x_init[2])[1]] = x_init[2][np.shape(x_init[0])[0] - 1:, :]
        x_init_tot[0:np.shape(x_init[0])[0], np.shape(x_init[2])[1] - 1:] = x_init[0]
        x_init_tot[np.shape(x_init[0])[0]:, np.shape(x_init[2])[1] - 1:] = x_init[1]

        y_init_tot = np.zeros((np.shape(y_init[2])[0] + 1, np.shape(y_init[0])[1] + np.shape(y_init[2])[1] - 1))
        y_init_tot[0:np.shape(y_init[0])[0], 0:np.shape(y_init[2])[1]] = y_init[2][0:np.shape(y_init[0])[0], :]
        y_init_tot[np.shape(y_init[0])[0]:, 0:np.shape(y_init[2])[1]] = y_init[2][np.shape(y_init[0])[0] - 1:, :]
        y_init_tot[0:np.shape(y_init[0])[0], np.shape(y_init[2])[1] - 1:] = y_init[0]
        y_init_tot[np.shape(y_init[0])[0]:, np.shape(y_init[2])[1] - 1:] = y_init[1]

        node_flags_tot = np.zeros(
            (np.shape(node_flags[2])[0] + 1, np.shape(node_flags[0])[1] + np.shape(node_flags[2])[1] - 1))
        node_flags_tot[0:np.shape(node_flags[0])[0], 0:np.shape(node_flags[2])[1]] = node_flags[2][
                                                                                     0:np.shape(node_flags[0])[0], :]
        node_flags_tot[np.shape(node_flags[0])[0]:, 0:np.shape(node_flags[2])[1]] = node_flags[2][
                                                                                    np.shape(node_flags[0])[0] - 1:, :]
        node_flags_tot[0:np.shape(node_flags[0])[0], np.shape(node_flags[2])[1] - 1:] = node_flags[0]
        node_flags_tot[np.shape(node_flags[0])[0]:, np.shape(node_flags[2])[1] - 1:] = node_flags[1]

    else:
        x_grid_tot = np.concatenate(x_grid)
        y_grid_tot = np.concatenate(y_grid)
        x_init_tot = np.concatenate(x_init)
        y_init_tot = np.concatenate(y_init)
        node_flags_tot = np.concatenate(node_flags)
        alpha = np.concatenate(alpha)
        beta = np.concatenate(beta)
        gamma = np.concatenate(gamma)
        tau = np.concatenate(tau)
        omega = np.concatenate(omega)
        jac = np.concatenate(jac)

    plotBoundaries(boundaries)
    plt.show()
    plotGrid(x_init_tot, y_init_tot, node_flags_tot)
    # plt.xlim((np.min(x_init),np.max(x_init)))
    plt.show()
    plotGrid(x_grid_tot, y_grid_tot, node_flags_tot)
    # plt.xlim((np.min(x_init),np.max(x_init)))
    # plt.ylim((np.min(y_init),np.max(y_init)))
    plt.show()

    # plotGrid(x_subgrid[0], y_subgrid[0])
    # plt.show()
    # plotGrid(x_subgrid[1], y_subgrid[1])
    # plt.show()

    return x_grid_tot, y_grid_tot, alpha, beta, gamma, tau, omega, jac, node_flags_tot, x_init_tot, y_init_tot


if __name__ == "__main__":
    x, y, alpha, beta, gamma, tau, omega, jac, node_matrix, x_init, y_init = gridgeneration_test("PFC nacelle", "rect", "planar")
    # x, y, alpha, beta, gamma, tau, omega, jac, node_matrix, x_init, y_init = gridgeneration_test("Akron", "rect", "axi")

    np.save('../test_functions/test_data/test_x_grid.npy', x)
    np.save('../test_functions/test_data/test_y_grid.npy', y)
    np.save('../test_functions/test_data/test_nodes.npy', node_matrix)
    np.save('../test_functions/test_data/alpha.npy', alpha)
    np.save('../test_functions/test_data/beta.npy', beta)
    np.save('../test_functions/test_data/gamma.npy', gamma)
    np.save('../test_functions/test_data/tau.npy', tau)
    np.save('../test_functions/test_data/omega.npy', omega)
    np.save('../test_functions/test_data/jac.npy', jac)
