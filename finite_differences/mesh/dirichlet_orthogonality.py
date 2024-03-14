"""Ensure Dirichlet orthogonality inside the whole grid. Based on initial algebraic grid.

Author: A. Habermann
Date:   05.11.2021

Sources:
    [1] Thompson, Joe F.; Soni, B. K.; Weatherill, N. P.: Handbook of grid generation, CRC Press, Boca Raton (1999), 
    Chapter 6.

"""

# Built-in/Generic Imports
import numpy as np
import scipy
from finite_differences.mesh.coefficients import *
from finite_differences.schemes.finite_difference_schemes import *


# x_it and y_it are the solutions of the current iteration
def dirichlet_control_functions(x_it, y_it, ghost_top, ghost_bottom, ghost_left, ghost_right, p_i, q_i,
                                orthogonality, delta, type_geom, type_grid, n_y_nacelle=0.0):
    p_o, q_o = orthogonal_control_functions(x_it, y_it, ghost_top, ghost_bottom, ghost_left, ghost_right, orthogonality
                                            , type_geom, type_grid, n_y_nacelle)
    p_dirich, q_dirich = blend_control_functions(p_i, q_i, p_o, q_o, delta)

    return p_dirich, q_dirich


# calculate control function for every point of the algebraic grid from the point in the algebraic grid
def algebraic_control_functions(x_alg, y_alg, type_geom: str, type_grid: str, n_y_nacelle=0):
    # initialize coefficients, Jacobi determinant and control functions
    alpha = np.zeros(np.shape(x_alg))
    beta = np.zeros(np.shape(x_alg))
    gamma = np.zeros(np.shape(x_alg))
    jac_det = np.zeros(np.shape(x_alg))
    p_a = np.zeros(np.shape(x_alg))
    q_a = np.zeros(np.shape(x_alg))
    n_y = np.shape(x_alg)[0]
    n_x = np.shape(x_alg)[1]

    for j in range(0, np.shape(x_alg)[1], 1):
        for i in range(0, np.shape(x_alg)[0], 1):
            # boundaries of the grid
            if (j == 0 and np.shape(x_alg)[0] - 1 > i > 0) and not (
                    type_grid == 'slit' and (i == n_y - n_y_nacelle - 2 or i == n_y - n_y_nacelle - 1)
                    and j == 0):  # left boundary
                alpha[i, j] = calc_alpha(x_alg, y_alg, i, j, 'right-sided')
                beta[i, j] = calc_beta(x_alg, y_alg, i, j, 'right-sided')
                gamma[i, j] = calc_gamma(x_alg, y_alg, i, j, 'right-sided')
                jac_det[i, j] = calc_jac_det(x_alg, y_alg, i, j, 'right-sided')

                if type_geom == 'planar':
                    # populate matrices for system of equations
                    a11 = alpha[i, j] * d1o2_cent('xi', x_alg, i, j)
                    a12 = gamma[i, j] * d1o2_right('eta', x_alg, i, j)
                    a21 = alpha[i, j] * d1o2_cent('xi', y_alg, i, j)
                    a22 = gamma[i, j] * d1o2_right('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_cent_right('eta', x_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_cent_right('eta', y_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', y_alg, i, j)

                elif type_geom == 'axi':
                    a11 = jac_det[i, j] ** 2 * d1o2_cent('xi', x_alg, i, j)
                    a12 = jac_det[i, j] ** 2 * d1o2_right('eta', x_alg, i, j)
                    a21 = jac_det[i, j] ** 2 * d1o2_cent('xi', y_alg, i, j)
                    a22 = jac_det[i, j] ** 2 * d1o2_right('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_cent_right('eta', x_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_cent_right('eta', y_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', y_alg, i, j) - jac_det[i, j] ** 2 / y_alg[i, j]

            elif (j == np.shape(x_alg)[1] - 1 and np.shape(x_alg)[0] - 1 > i > 0) and not (
                    type_grid == 'slit' and (i == n_y - n_y_nacelle - 2 or i == n_y - n_y_nacelle - 1)
                    and j == n_x - 1):  # right boundary
                alpha[i, j] = calc_alpha(x_alg, y_alg, i, j, 'left-sided')
                beta[i, j] = calc_beta(x_alg, y_alg, i, j, 'left-sided')
                gamma[i, j] = calc_gamma(x_alg, y_alg, i, j, 'left-sided')
                jac_det[i, j] = calc_jac_det(x_alg, y_alg, i, j, 'left-sided')

                if type_geom == 'planar':
                    # populate matrices for system of equations
                    a11 = alpha[i, j] * d1o2_cent('xi', x_alg, i, j)
                    a12 = gamma[i, j] * d1o2_left('eta', x_alg, i, j)
                    a21 = alpha[i, j] * d1o2_cent('xi', y_alg, i, j)
                    a22 = gamma[i, j] * d1o2_left('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_cent_left('eta', x_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_cent_left('eta', y_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', y_alg, i, j)

                elif type_geom == 'axi':
                    # populate matrices for system of equations
                    a11 = jac_det[i, j] ** 2 * d1o2_cent('xi', x_alg, i, j)
                    a12 = jac_det[i, j] ** 2 * d1o2_left('eta', x_alg, i, j)
                    a21 = jac_det[i, j] ** 2 * d1o2_cent('xi', y_alg, i, j)
                    a22 = jac_det[i, j] ** 2 * d1o2_left('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_cent_left('eta', x_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_cent_left('eta', y_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', y_alg, i, j) - jac_det[i, j] ** 2 / y_alg[i, j]

            elif (i == 0 and np.shape(x_alg)[1] - 1 > j > 0) or (
                    type_grid == 'slit' and i == n_y - n_y_nacelle - 1 and (
                    0 < j < n_x - 1)):  # top of grid and lower side of slit
                alpha[i, j] = calc_alpha(x_alg, y_alg, i, j, 'bottom-sided')
                beta[i, j] = calc_beta(x_alg, y_alg, i, j, 'bottom-sided')
                gamma[i, j] = calc_gamma(x_alg, y_alg, i, j, 'bottom-sided')
                jac_det[i, j] = calc_jac_det(x_alg, y_alg, i, j, 'bottom-sided')

                if type_geom == 'planar':
                    # populate matrices for system of equations
                    a11 = alpha[i, j] * d1o2_right('xi', x_alg, i, j)
                    a12 = gamma[i, j] * d1o2_cent('eta', x_alg, i, j)
                    a21 = alpha[i, j] * d1o2_right('xi', y_alg, i, j)
                    a22 = gamma[i, j] * d1o2_cent('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_cent_right('xi', x_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_cent_right('xi', y_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', y_alg, i, j)

                elif type_geom == 'axi':
                    # populate matrices for system of equations
                    a11 = jac_det[i, j] ** 2 * d1o2_right('xi', x_alg, i, j)
                    a12 = jac_det[i, j] ** 2 * d1o2_cent('eta', x_alg, i, j)
                    a21 = jac_det[i, j] ** 2 * d1o2_right('xi', y_alg, i, j)
                    a22 = jac_det[i, j] ** 2 * d1o2_cent('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_cent_right('xi', x_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_cent_right('xi', y_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', y_alg, i, j) - jac_det[i, j] ** 2 / y_alg[i, j]

            elif (i == np.shape(x_alg)[0] - 1 and np.shape(x_alg)[1] - 1 > j > 0) or (
                    type_grid == 'slit' and i == n_y - n_y_nacelle - 2 and (
                    0 < j < n_x - 1)):  # bottom of grid and upper side of slit
                alpha[i, j] = calc_alpha(x_alg, y_alg, i, j, 'top-sided')
                beta[i, j] = calc_beta(x_alg, y_alg, i, j, 'top-sided')
                gamma[i, j] = calc_gamma(x_alg, y_alg, i, j, 'top-sided')
                jac_det[i, j] = calc_jac_det(x_alg, y_alg, i, j, 'top-sided')

                if type_geom == 'planar':
                    # populate matrices for system of equations
                    a11 = alpha[i, j] * d1o2_left('xi', x_alg, i, j)
                    a12 = gamma[i, j] * d1o2_cent('eta', x_alg, i, j)
                    a21 = alpha[i, j] * d1o2_left('xi', y_alg, i, j)
                    a22 = gamma[i, j] * d1o2_cent('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_cent_left('xi', x_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_cent_left('xi', y_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', y_alg, i, j)

                elif type_geom == 'axi':
                    # populate matrices for system of equations
                    a11 = jac_det[i, j] ** 2 * d1o2_left('xi', x_alg, i, j)
                    a12 = jac_det[i, j] ** 2 * d1o2_cent('eta', x_alg, i, j)
                    a21 = jac_det[i, j] ** 2 * d1o2_left('xi', y_alg, i, j)
                    a22 = jac_det[i, j] ** 2 * d1o2_cent('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_cent_left('xi', x_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_cent_left('xi', y_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', y_alg, i, j) - jac_det[i, j] ** 2 / y_alg[i, j]

            # corners of the grid
            elif (i == 0 and j == 0) or (type_grid == 'slit' and i == n_y - n_y_nacelle - 1
                                         and j == 0):  # top left and "dummy" slit line bottom left
                alpha[i, j] = calc_alpha(x_alg, y_alg, i, j, 'right-sided')
                beta[i, j] = calc_beta(x_alg, y_alg, i, j, 'bottom-right-sided')
                gamma[i, j] = calc_gamma(x_alg, y_alg, i, j, 'bottom-sided')
                jac_det[i, j] = calc_jac_det(x_alg, y_alg, i, j, 'bottom-right-sided')

                if type_geom == 'planar':
                    # populate matrices for system of equations
                    a11 = alpha[i, j] * d1o2_right('xi', x_alg, i, j)
                    a12 = gamma[i, j] * d1o2_right('eta', x_alg, i, j)
                    a21 = alpha[i, j] * d1o2_right('xi', y_alg, i, j)
                    a22 = gamma[i, j] * d1o2_right('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_right_right(x_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_right_right(y_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', y_alg, i, j)

                elif type_geom == 'axi':
                    a11 = jac_det[i, j] ** 2 * d1o2_right('xi', x_alg, i, j)
                    a12 = jac_det[i, j] ** 2 * d1o2_right('eta', x_alg, i, j)
                    a21 = jac_det[i, j] ** 2 * d1o2_right('xi', y_alg, i, j)
                    a22 = jac_det[i, j] ** 2 * d1o2_right('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_right_right(x_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_right_right(y_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', y_alg, i, j) - jac_det[i, j] ** 2 / y_alg[i, j]

            elif (i == 0 and j == np.shape(x_alg)[1] - 1) or (type_grid == 'slit' and i == n_y - n_y_nacelle - 1
                                                              and j == n_x - 1):  # top right and "dummy" slit line bottom right
                alpha[i, j] = calc_alpha(x_alg, y_alg, i, j, 'left-sided')
                beta[i, j] = calc_beta(x_alg, y_alg, i, j, 'bottom-left-sided')
                gamma[i, j] = calc_gamma(x_alg, y_alg, i, j, 'bottom-sided')
                jac_det[i, j] = calc_jac_det(x_alg, y_alg, i, j, 'bottom-left-sided')

                if type_geom == 'planar':
                    a11 = alpha[i, j] * d1o2_right('xi', x_alg, i, j)
                    a12 = gamma[i, j] * d1o2_left('eta', x_alg, i, j)
                    a21 = alpha[i, j] * d1o2_right('xi', y_alg, i, j)
                    a22 = gamma[i, j] * d1o2_left('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_left_right('xi', x_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_left_right('xi', y_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', y_alg, i, j)

                elif type_geom == 'axi':
                    a11 = jac_det[i, j] ** 2 * d1o2_right('xi', x_alg, i, j)
                    a12 = jac_det[i, j] ** 2 * d1o2_left('eta', x_alg, i, j)
                    a21 = jac_det[i, j] ** 2 * d1o2_right('xi', y_alg, i, j)
                    a22 = jac_det[i, j] ** 2 * d1o2_left('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_left_right('xi', x_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_left_right('xi', y_alg, i, j) \
                         - alpha[i, j] * d2o2_right('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', y_alg, i, j) - jac_det[i, j] ** 2 / y_alg[i, j]

            elif (i == np.shape(x_alg)[0] - 1 and j == 0) or (type_grid == 'slit' and i == n_y - n_y_nacelle - 2
                                                              and j == 0):  # bottom left and "dummy" slit line top left
                alpha[i, j] = calc_alpha(x_alg, y_alg, i, j, 'right-sided')
                beta[i, j] = calc_beta(x_alg, y_alg, i, j, 'top-right-sided')
                gamma[i, j] = calc_gamma(x_alg, y_alg, i, j, 'top-sided')
                jac_det[i, j] = calc_jac_det(x_alg, y_alg, i, j, 'top-right-sided')

                if type_geom == 'planar':
                    a11 = alpha[i, j] * d1o2_left('xi', x_alg, i, j)
                    a12 = gamma[i, j] * d1o2_right('eta', x_alg, i, j)
                    a21 = alpha[i, j] * d1o2_left('xi', y_alg, i, j)
                    a22 = gamma[i, j] * d1o2_right('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_left_right('eta', x_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_left_right('eta', y_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', y_alg, i, j)

                elif type_geom == 'axi':
                    a11 = jac_det[i, j] ** 2 * d1o2_left('xi', x_alg, i, j)
                    a12 = jac_det[i, j] ** 2 * d1o2_right('eta', x_alg, i, j)
                    a21 = jac_det[i, j] ** 2 * d1o2_left('xi', y_alg, i, j)
                    a22 = jac_det[i, j] ** 2 * d1o2_right('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_left_right('eta', x_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_left_right('eta', y_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_right('eta', y_alg, i, j) - jac_det[i, j] ** 2 / y_alg[i, j]

            elif (i == np.shape(x_alg)[0] - 1 and j == np.shape(x_alg)[1] - 1) or (
                    type_grid == 'slit' and i == n_y - n_y_nacelle - 2
                    and j == n_x - 1):  # bottom right and "dummy" slit line top right
                alpha[i, j] = calc_alpha(x_alg, y_alg, i, j, 'left-sided')
                beta[i, j] = calc_beta(x_alg, y_alg, i, j, 'top-left-sided')
                gamma[i, j] = calc_gamma(x_alg, y_alg, i, j, 'top-sided')
                jac_det[i, j] = calc_jac_det(x_alg, y_alg, i, j, 'top-left-sided')

                if type_geom == 'planar':
                    a11 = alpha[i, j] * d1o2_left('xi', x_alg, i, j)
                    a12 = gamma[i, j] * d1o2_left('eta', x_alg, i, j)
                    a21 = alpha[i, j] * d1o2_left('xi', y_alg, i, j)
                    a22 = gamma[i, j] * d1o2_left('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_left_left(x_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_left_left(y_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', y_alg, i, j)

                elif type_geom == 'axi':
                    a11 = jac_det[i, j] ** 2 * d1o2_left('xi', x_alg, i, j)
                    a12 = jac_det[i, j] ** 2 * d1o2_left('eta', x_alg, i, j)
                    a21 = jac_det[i, j] ** 2 * d1o2_left('xi', y_alg, i, j)
                    a22 = jac_det[i, j] ** 2 * d1o2_left('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_left_left(x_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_left_left(y_alg, i, j) \
                         - alpha[i, j] * d2o2_left('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_left('eta', y_alg, i, j) - jac_det[i, j] ** 2 / y_alg[i, j]

            # all points inside the grid
            else:
                alpha[i, j] = calc_alpha(x_alg, y_alg, i, j, 'centered')
                beta[i, j] = calc_beta(x_alg, y_alg, i, j, 'center-center')
                gamma[i, j] = calc_gamma(x_alg, y_alg, i, j, 'centered')
                jac_det[i, j] = calc_jac_det(x_alg, y_alg, i, j, 'centered')

                if type_geom == 'planar':
                    # populate matrices for system of equations
                    a11 = alpha[i, j] * d1o2_cent('xi', x_alg, i, j)
                    a12 = gamma[i, j] * d1o2_cent('eta', x_alg, i, j)
                    a21 = alpha[i, j] * d1o2_cent('xi', y_alg, i, j)
                    a22 = gamma[i, j] * d1o2_cent('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_cent(x_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_cent(y_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', y_alg, i, j)

                elif type_geom == 'axi':
                    a11 = jac_det[i, j] ** 2 * d1o2_cent('xi', x_alg, i, j)
                    a12 = jac_det[i, j] ** 2 * d1o2_cent('eta', x_alg, i, j)
                    a21 = jac_det[i, j] ** 2 * d1o2_cent('xi', y_alg, i, j)
                    a22 = jac_det[i, j] ** 2 * d1o2_cent('eta', y_alg, i, j)

                    b1 = 2 * beta[i, j] * d202_mix_cent(x_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', x_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', x_alg, i, j)

                    b2 = 2 * beta[i, j] * d202_mix_cent(y_alg, i, j) \
                         - alpha[i, j] * d2o2_cent('xi', y_alg, i, j) \
                         - gamma[i, j] * d2o2_cent('eta', y_alg, i, j) - jac_det[i, j] ** 2 / y_alg[i, j]

            # solve linear system of equations
            a = np.array([[a11, a12], [a21, a22]])
            b = np.array([b1, b2])
            p_a[i, j], q_a[i, j] = np.linalg.solve(a, b)  # [1], eq. 6.14

    return p_a, q_a


# iteratively smooth control functions (only necessary inside grid, not at boundaries/corners) (?)
def initial_control_functions(p_a, q_a, type_geom: str, type_grid: str, n_y_nacelle=0):
    p_i = np.copy(p_a)
    q_i = np.copy(q_a)

    max_it = 100  # no. of iterations
    it = 0
    err_p = 1
    err_q = 1

    if type_geom == 'planar':
        while it < max_it and (err_p > 10e-4 or err_q > 10e-4):
            for j in range(2, np.shape(p_a)[1] - 2, 1):
                for i in range(2, np.shape(p_a)[0] - 2, 1):
                    p_i[i, j] = 0.4 * (p_a[i, j + 1] + p_a[i, j - 1]) + 0.1 * (
                            p_a[i, j + 2] + p_a[i, j - 2])  # [1], eq. 6.15
                    if type_grid == 'slit' and i == n_y_nacelle:  # dummy slit line bottom
                        q_i[i, j] = 0.4 * (q_a[i + 1, j] + q_a[i, j]) + 0.1 * (q_a[i + 3, j] + q_a[i + 2, j])
                    elif type_grid == 'slit' and i == n_y_nacelle - 1:  # dummy slit line top
                        q_i[i, j] = 0.4 * (q_a[i, j] + q_a[i - 1, j]) + 0.1 * (q_a[i - 2, j] + q_a[i - 3, j])
                    else:
                        q_i[i, j] = 0.4 * (q_a[i + 1, j] + q_a[i - 1, j]) + 0.1 * (q_a[i + 2, j] + q_a[i - 2, j])
            err_p = np.amax(abs((p_i - p_a) / p_a))
            err_q = np.amax(abs((q_i - q_a) / q_a))
            p_a = np.copy(p_i)
            q_a = np.copy(q_i)
            it += 1
    elif type_geom == 'axi':
        while it < max_it and (err_p > 10e-4 or err_q > 10e-4):
            for j in range(2, np.shape(p_a)[1] - 2, 1):
                for i in range(2, np.shape(p_a)[0] - 2, 1):
                    if type_grid == 'rect' and j == 2 and (
                            np.any(q_a[:, 0] != q_a[:, 0]) or np.any(np.isinf(q_a[:, 0]))):
                        p_i[i, j] = 0.4 * p_a[i, j] + 0.3 * (p_a[i, j + 1] + p_a[i, j - 1])  # [1], eq. 6.15
                    else:
                        p_i[i, j] = 0.4 * p_a[i, j] + 0.15 * (p_a[i, j + 1] + p_a[i, j - 1]) + 0.15 * (
                                p_a[i, j + 2] + p_a[i, j - 2])  # [1], eq. 6.15
                    if type_grid == 'slit' and i == n_y_nacelle:  # dummy slit line bottom
                        q_i[i, j] = 0.4 * (q_a[i + 1, j] + q_a[i, j]) + 0.1 * (q_a[i + 3, j] + q_a[i + 2, j])
                    elif type_grid == 'slit' and i == n_y_nacelle - 1:  # dummy slit line top
                        q_i[i, j] = 0.4 * (q_a[i, j] + q_a[i - 1, j]) + 0.1 * (q_a[i - 2, j] + q_a[i - 3, j])
                    elif type_grid == 'rect' and i == (np.shape(p_a)[0] - 3) and np.any(q_a[-1, :] != q_a[-1, :]):
                        q_i[i, j] = 0.4 * q_a[i, j] + 0.3 * (q_a[i + 1, j] + q_a[i - 1, j])
                    else:
                        q_i[i, j] = 0.4 * q_a[i, j] + 0.15 * (q_a[i + 1, j] + q_a[i - 1, j]) + 0.15 * (
                                q_a[i + 2, j] + q_a[i - 2, j])
            errp = abs((p_i - p_a) / p_a)
            errq = abs((q_i - q_a) / q_a)
            err_p = np.nanmax(errp[errp != np.inf])
            err_q = np.nanmax(errq[errq != np.inf])
            p_a = np.copy(p_i)
            q_a = np.copy(q_i)
            it += 1

    return p_i, q_i


# calculate orthogonal control functions
def orthogonal_control_functions(x_it, y_it, ghost_top, ghost_bottom, ghost_left, ghost_right, orthogonality: str
                                 , type_geom: str, type_grid: str, n_y_nacelle=0):
    p_o = np.zeros(np.shape(x_it))
    q_o = np.zeros(np.shape(x_it))
    gamma = np.zeros(np.shape(x_it))
    alpha = np.zeros(np.shape(x_it))
    x_g = np.zeros(np.shape(x_it))
    y_g = np.zeros(np.shape(x_it))
    x_xi = np.zeros(np.shape(x_it))
    y_xi = np.zeros(np.shape(x_it))
    x_xi_xi = np.zeros(np.shape(x_it))
    y_xi_xi = np.zeros(np.shape(x_it))
    x_eta = np.zeros(np.shape(x_it))
    y_eta = np.zeros(np.shape(x_it))
    x_eta_eta = np.zeros(np.shape(x_it))
    y_eta_eta = np.zeros(np.shape(x_it))
    jac_det = np.zeros(np.shape(x_it))
    n_y = np.shape(x_it)[0]
    n_x = np.shape(x_it)[1]

    # calculate orthogonal control functions at boundaries
    # left boundary
    n = 0
    j = n

    for i in range(1, np.shape(x_it)[0] - 1, 1):
        if not (type_grid == 'slit' and (i == n_y - n_y_nacelle - 2 or i == n_y - n_y_nacelle - 1)):
            alpha[i, n] = ghost_left[6][i]
            x_g[i, n] = ghost_left[0][i]
            y_g[i, n] = ghost_left[1][i]
            x_xi[i, n] = d1o2_cent('xi', x_it, i, j)
            y_xi[i, n] = d1o2_cent('xi', y_it, i, j)
            x_xi_xi[i, n] = d2o2_cent('xi', x_it, i, j)
            y_xi_xi[i, n] = d2o2_cent('xi', y_it, i, j)
            x_eta[i, n] = d1o1_right('eta', x_it, i, j)
            y_eta[i, n] = d1o1_right('eta', y_it, i, j)
            x_eta_eta[i, n] = x_it[i, n + 1] - 2 * x_it[i, n] + x_g[i, n]
            y_eta_eta[i, n] = y_it[i, n + 1] - 2 * y_it[i, n] + y_g[i, n]
            gamma[i, n] = x_xi[i, n] ** 2 + y_xi[i, n] ** 2
            jac_det[i, j] = calc_jac_det(x_it, y_it, i, j, 'right-sided')
            if orthogonality == 'xi':
                p_o[i, j] = 0
                q_o[i, j] = 0
            else:
                if type_geom == 'planar':
                    p_o[i, j] = (-1 / gamma[i, j]) * (x_xi[i, j] * x_xi_xi[i, j] + (y_xi[i, j] * y_xi_xi[i, j])) - \
                                (1 / alpha[i, j]) * (x_xi[i, j] * x_eta_eta[i, j] + y_xi[i, j] * y_eta_eta[i, j])
                    q_o[i, j] = (-1 / alpha[i, j]) * (x_eta[i, j] * x_eta_eta[i, j] + (y_eta[i, j] * y_eta_eta[i, j])) - \
                                (1 / gamma[i, j]) * (x_eta[i, j] * x_xi_xi[i, j] + y_eta[i, j] * y_xi_xi[i, j])
                elif type_geom == 'axi':
                    p_o[i, j], q_o[i, j] = solve_axi_ortho(x_xi[i, j], y_xi[i, j], x_eta[i, j], y_eta[i, j],
                                                           x_xi_xi[i, j],
                                                           y_xi_xi[i, j], x_eta_eta[i, j], y_eta_eta[i, j],
                                                           jac_det[i, j],
                                                           alpha[i, j], gamma[i, j], y_it[i, j])

    # right boundary
    n = np.shape(x_it)[1] - 1
    j = n

    for i in range(1, np.shape(x_it)[0] - 1, 1):
        if not (type_grid == 'slit' and (i == n_y - n_y_nacelle - 2 or i == n_y - n_y_nacelle - 1)):
            alpha[i, n] = ghost_right[6][i]
            x_g[i, n] = ghost_right[0][i]
            y_g[i, n] = ghost_right[1][i]
            x_xi[i, n] = d1o2_cent('xi', x_it, i, j)
            y_xi[i, n] = d1o2_cent('xi', y_it, i, j)
            x_xi_xi[i, n] = d2o2_cent('xi', x_it, i, j)
            y_xi_xi[i, n] = d2o2_cent('xi', y_it, i, j)
            x_eta[i, n] = d1o1_left('eta', x_it, i, j)
            y_eta[i, n] = d1o1_left('eta', y_it, i, j)
            x_eta_eta[i, n] = x_g[i, n] - 2 * x_it[i, n] + x_it[i, n - 1]
            y_eta_eta[i, n] = y_g[i, n] - 2 * y_it[i, n] + y_it[i, n - 1]
            gamma[i, n] = x_xi[i, n] ** 2 + y_xi[i, n] ** 2
            jac_det[i, j] = calc_jac_det(x_it, y_it, i, j, 'left-sided')
            if orthogonality == 'xi':
                p_o[i, j] = 0
                q_o[i, j] = 0
            else:
                if type_geom == 'planar':
                    p_o[i, j] = (-1 / gamma[i, j]) * (x_xi[i, j] * x_xi_xi[i, j] + (y_xi[i, j] * y_xi_xi[i, j])) - \
                                (1 / alpha[i, j]) * (x_xi[i, j] * x_eta_eta[i, j] + y_xi[i, j] * y_eta_eta[i, j])
                    q_o[i, j] = (-1 / alpha[i, j]) * (x_eta[i, j] * x_eta_eta[i, j] + (y_eta[i, j] * y_eta_eta[i, j])) - \
                                (1 / gamma[i, j]) * (x_eta[i, j] * x_xi_xi[i, j] + y_eta[i, j] * y_xi_xi[i, j])
                elif type_geom == 'axi':
                    p_o[i, j], q_o[i, j] = solve_axi_ortho(x_xi[i, j], y_xi[i, j], x_eta[i, j], y_eta[i, j],
                                                           x_xi_xi[i, j],
                                                           y_xi_xi[i, j], x_eta_eta[i, j], y_eta_eta[i, j],
                                                           jac_det[i, j],
                                                           alpha[i, j], gamma[i, j], y_it[i, j])

    # top boundary
    m = 0
    i = m
    for j in range(1, np.shape(x_it)[1] - 1, 1):
        gamma[i, j] = ghost_top[7][j]
        x_g[i, j] = ghost_top[0][j]
        y_g[i, j] = ghost_top[1][j]
        x_xi[i, j] = d1o1_right('xi', x_it, i, j)
        y_xi[i, j] = d1o1_right('xi', y_it, i, j)
        x_xi_xi[i, j] = x_it[i + 1, j] - 2 * x_it[i, j] + x_g[i, j]
        y_xi_xi[i, j] = y_it[i + 1, j] - 2 * y_it[i, j] + y_g[i, j]
        x_eta[i, j] = d1o2_cent('eta', x_it, i, j)
        y_eta[i, j] = d1o2_cent('eta', y_it, i, j)
        x_eta_eta[i, j] = d2o2_cent('eta', x_it, i, j)
        y_eta_eta[i, j] = d2o2_cent('eta', y_it, i, j)
        alpha[i, j] = x_eta[i, j] ** 2 + y_eta[i, j] ** 2
        jac_det[i, j] = calc_jac_det(x_it, y_it, i, j, 'bottom-sided')
        if orthogonality == 'eta':
            p_o[i, j] = 0
            q_o[i, j] = 0
        else:
            if type_geom == 'planar':
                p_o[i, j] = (-1 / gamma[i, j]) * (x_xi[i, j] * x_xi_xi[i, j] + (y_xi[i, j] * y_xi_xi[i, j])) - \
                            (1 / alpha[i, j]) * (x_xi[i, j] * x_eta_eta[i, j] + y_xi[i, j] * y_eta_eta[i, j])
                q_o[i, j] = (-1 / alpha[i, j]) * (x_eta[i, j] * x_eta_eta[i, j] + (y_eta[i, j] * y_eta_eta[i, j])) - \
                            (1 / gamma[i, j]) * (x_eta[i, j] * x_xi_xi[i, j] + y_eta[i, j] * y_xi_xi[i, j])
            elif type_geom == 'axi':
                p_o[i, j], q_o[i, j] = solve_axi_ortho(x_xi[i, j], y_xi[i, j], x_eta[i, j], y_eta[i, j], x_xi_xi[i, j],
                                                       y_xi_xi[i, j], x_eta_eta[i, j], y_eta_eta[i, j], jac_det[i, j],
                                                       alpha[i, j], gamma[i, j], y_it[i, j])

    # bottom boundary
    m = np.shape(x_it)[0] - 1
    i = m
    for j in range(1, np.shape(x_it)[1] - 1, 1):
        gamma[i, j] = ghost_bottom[7][j]
        x_g[i, j] = ghost_bottom[0][j]
        y_g[i, j] = ghost_bottom[1][j]
        x_xi[i, j] = d1o1_left('xi', x_it, i, j)  # x_it[m,j]-x_it[m-1,j]#ghost_bottom[2][j]
        y_xi[i, j] = d1o1_left('xi', y_it, i, j)  # y_it[m,j]-y_it[m-1,j]#ghost_bottom[3][j]
        x_xi_xi[i, j] = x_g[i, j] - 2 * x_it[i, j] + x_it[i - 1, j]
        y_xi_xi[i, j] = y_g[i, j] - 2 * y_it[i, j] + y_it[i - 1, j]
        x_eta[i, j] = d1o2_cent('eta', x_it, i, j)  # 0.5*(x_it[m,j+1]-x_it[m,j-1])#ghost_top[4][j]
        y_eta[i, j] = d1o2_cent('eta', y_it, i, j)  # 0.5*(y_it[m,j+1]-y_it[m,j-1])#ghost_top[5][j]
        x_eta_eta[i, j] = d2o2_cent('eta', x_it, i, j)  # x_it[m,j+1]-2*x_it[m,j]+x_it[m,j-1]
        y_eta_eta[i, j] = d2o2_cent('eta', y_it, i, j)  # x_it[m,j+1]-2*x_it[m,j]+x_it[m,j-1]
        alpha[i, j] = x_eta[i, j] ** 2 + y_eta[i, j] ** 2
        jac_det[i, j] = calc_jac_det(x_it, y_it, i, j, 'top-sided')
        if orthogonality == 'eta':
            p_o[i, j] = 0
            q_o[i, j] = 0
        else:
            if type_geom == 'planar':
                p_o[i, j] = (-1 / gamma[i, j]) * (x_xi[i, j] * x_xi_xi[i, j] + (y_xi[i, j] * y_xi_xi[i, j])) - \
                            (1 / alpha[i, j]) * (x_xi[i, j] * x_eta_eta[i, j] + y_xi[i, j] * y_eta_eta[i, j])
                q_o[i, j] = (-1 / alpha[i, j]) * (x_eta[i, j] * x_eta_eta[i, j] + (y_eta[i, j] * y_eta_eta[i, j])) - \
                            (1 / gamma[i, j]) * (x_eta[i, j] * x_xi_xi[i, j] + y_eta[i, j] * y_xi_xi[i, j])
            elif type_geom == 'axi':
                p_o[i, j], q_o[i, j] = solve_axi_ortho(x_xi[i, j], y_xi[i, j], x_eta[i, j], y_eta[i, j], x_xi_xi[i, j],
                                                       y_xi_xi[i, j], x_eta_eta[i, j], y_eta_eta[i, j], jac_det[i, j],
                                                       alpha[i, j], gamma[i, j], y_it[i, j])

    # corners using one-sided difference formulas (no orthogonality, only conformity)
    # top left corner
    x_xi[0, 0] = d1o1_right('xi', x_it, 0, 0)
    y_xi[0, 0] = d1o1_right('xi', y_it, 0, 0)
    x_eta[0, 0] = d1o1_right('eta', x_it, 0, 0)
    y_eta[0, 0] = d1o1_right('eta', y_it, 0, 0)
    x_xi_xi[0, 0] = d2o2_right('xi', x_it, 0, 0)
    y_xi_xi[0, 0] = d2o2_right('xi', y_it, 0, 0)
    x_eta_eta[0, 0] = d2o2_right('eta', x_it, 0, 0)
    y_eta_eta[0, 0] = d2o2_right('eta', y_it, 0, 0)
    gamma[0, 0] = x_xi[0, 0] ** 2 + y_xi[0, 0] ** 2
    alpha[0, 0] = x_eta[0, 0] ** 2 + y_eta[0, 0] ** 2
    jac_det[0, 0] = calc_jac_det(x_it, y_it, 0, 0, 'bottom-right-sided')

    # top right corner
    k = np.shape(x_it)[1] - 1
    x_xi[0, k] = d1o1_right('xi', x_it, 0, k)
    y_xi[0, k] = d1o1_right('xi', y_it, 0, k)
    x_eta[0, k] = d1o1_left('eta', x_it, 0, k)
    y_eta[0, k] = d1o1_left('eta', y_it, 0, k)
    x_xi_xi[0, k] = d2o2_right('xi', x_it, 0, k)
    y_xi_xi[0, k] = d2o2_right('xi', y_it, 0, k)
    x_eta_eta[0, k] = d2o2_left('eta', x_it, 0, k)
    y_eta_eta[0, k] = d2o2_left('eta', y_it, 0, k)
    gamma[0, k] = x_xi[0, k] ** 2 + y_xi[0, k] ** 2
    alpha[0, k] = x_eta[0, k] ** 2 + y_eta[0, k] ** 2
    jac_det[0, k] = calc_jac_det(x_it, y_it, 0, k, 'bottom-left-sided')

    # bottom left corner
    k = np.shape(x_it)[0] - 1
    x_xi[k, 0] = d1o1_left('xi', x_it, k, 0)
    y_xi[k, 0] = d1o1_left('xi', y_it, k, 0)
    x_eta[k, 0] = d1o1_right('eta', x_it, k, 0)
    y_eta[k, 0] = d1o1_right('eta', y_it, k, 0)
    x_xi_xi[k, 0] = d2o2_left('xi', x_it, k, 0)
    y_xi_xi[k, 0] = d2o2_left('xi', y_it, k, 0)
    x_eta_eta[k, 0] = d2o2_right('eta', x_it, k, 0)
    y_eta_eta[k, 0] = d2o2_right('eta', y_it, k, 0)
    gamma[k, 0] = x_xi[k, 0] ** 2 + y_xi[k, 0] ** 2
    alpha[k, 0] = x_eta[k, 0] ** 2 + y_eta[k, 0] ** 2
    jac_det[k, 0] = calc_jac_det(x_it, y_it, k, 0, 'top-right-sided')

    # bottom right corner
    k = np.shape(x_it)[0] - 1
    m = np.shape(x_it)[1] - 1
    x_xi[k, m] = d1o1_left('xi', x_it, k, m)
    y_xi[k, m] = d1o1_left('xi', y_it, k, m)
    x_eta[k, m] = d1o1_left('eta', x_it, k, m)
    y_eta[k, m] = d1o1_left('eta', y_it, k, m)
    x_xi_xi[k, m] = d2o2_left('xi', x_it, k, m)
    y_xi_xi[k, m] = d2o2_left('xi', y_it, k, m)
    x_eta_eta[k, m] = d2o2_left('eta', x_it, k, m)
    y_eta_eta[k, m] = d2o2_left('eta', y_it, k, m)
    gamma[k, m] = x_xi[k, m] ** 2 + y_xi[k, m] ** 2
    alpha[k, m] = x_eta[k, m] ** 2 + y_eta[k, m] ** 2
    jac_det[k, m] = calc_jac_det(x_it, y_it, k, m, 'top-left-sided')

    if type_grid == 'slit':
        # bottom left corner
        k = n_y - n_y_nacelle - 1
        m = 0
        x_xi[k, m] = d1o1_right('xi', x_it, k, m)
        y_xi[k, m] = d1o1_right('xi', y_it, k, m)
        x_eta[k, m] = d1o1_right('eta', x_it, k, m)
        y_eta[k, m] = d1o1_right('eta', y_it, k, m)
        x_xi_xi[k, m] = d2o2_right('xi', x_it, k, m)
        y_xi_xi[k, m] = d2o2_right('xi', y_it, k, m)
        x_eta_eta[k, m] = d2o2_right('eta', x_it, k, m)
        y_eta_eta[k, m] = d2o2_right('eta', y_it, k, m)
        gamma[k, m] = x_xi[k, m] ** 2 + y_xi[k, m] ** 2
        alpha[k, m] = x_eta[k, m] ** 2 + y_eta[k, m] ** 2
        jac_det[k, m] = calc_jac_det(x_it, y_it, k, m, 'bottom-right-sided')

        # bottom right corner
        k = n_y - n_y_nacelle - 1
        m = n_x - 1
        x_xi[k, m] = d1o1_right('xi', x_it, k, m)
        y_xi[k, m] = d1o1_right('xi', y_it, k, m)
        x_eta[k, m] = d1o1_left('eta', x_it, k, m)
        y_eta[k, m] = d1o1_left('eta', y_it, k, m)
        x_xi_xi[k, m] = d2o2_right('xi', x_it, k, m)
        y_xi_xi[k, m] = d2o2_right('xi', y_it, k, m)
        x_eta_eta[k, m] = d2o2_left('eta', x_it, k, m)
        y_eta_eta[k, m] = d2o2_left('eta', y_it, k, m)
        gamma[k, m] = x_xi[k, m] ** 2 + y_xi[k, m] ** 2
        alpha[k, m] = x_eta[k, m] ** 2 + y_eta[k, m] ** 2
        jac_det[k, m] = calc_jac_det(x_it, y_it, k, m, 'bottom-left-sided')

        # top left corner
        k = n_y - n_y_nacelle - 2
        m = 0
        x_xi[k, m] = d1o1_left('xi', x_it, k, m)
        y_xi[k, m] = d1o1_left('xi', y_it, k, m)
        x_eta[k, m] = d1o1_right('eta', x_it, k, m)
        y_eta[k, m] = d1o1_right('eta', y_it, k, m)
        x_xi_xi[k, m] = d2o2_left('xi', x_it, k, m)
        y_xi_xi[k, m] = d2o2_left('xi', y_it, k, m)
        x_eta_eta[k, m] = d2o2_right('eta', x_it, k, m)
        y_eta_eta[k, m] = d2o2_right('eta', y_it, k, m)
        gamma[k, m] = x_xi[k, m] ** 2 + y_xi[k, m] ** 2
        alpha[k, m] = x_eta[k, m] ** 2 + y_eta[k, m] ** 2
        jac_det[k, m] = calc_jac_det(x_it, y_it, k, m, 'top-right-sided')

        # top right corner
        k = n_y - n_y_nacelle - 2
        m = n_x - 1
        x_xi[k, m] = d1o1_left('xi', x_it, k, m)
        y_xi[k, m] = d1o1_left('xi', y_it, k, m)
        x_eta[k, m] = d1o1_left('eta', x_it, k, m)
        y_eta[k, m] = d1o1_left('eta', y_it, k, m)
        x_xi_xi[k, m] = d2o2_left('xi', x_it, k, m)
        y_xi_xi[k, m] = d2o2_left('xi', y_it, k, m)
        x_eta_eta[k, m] = d2o2_left('eta', x_it, k, m)
        y_eta_eta[k, m] = d2o2_left('eta', y_it, k, m)
        gamma[k, m] = x_xi[k, m] ** 2 + y_xi[k, m] ** 2
        alpha[k, m] = x_eta[k, m] ** 2 + y_eta[k, m] ** 2
        jac_det[k, m] = calc_jac_det(x_it, y_it, k, m, 'top-left-sided')

    if type_grid == 'slit':
        list_corners = [0, n_y - n_y_nacelle - 2, n_y - n_y_nacelle - 1, np.shape(x_it)[0] - 1]
    else:
        list_corners = [0, np.shape(x_it)[0] - 1]

    if type_geom == 'planar':
        for i in list_corners:
            for j in [0, np.shape(x_it)[1] - 1]:
                p_o[i, j] = (-1 / gamma[i, j]) * (x_xi[i, j] * x_xi_xi[i, j] + (y_xi[i, j] * y_xi_xi[i, j])) - \
                            (1 / alpha[i, j]) * (x_xi[i, j] * x_eta_eta[i, j] + y_xi[i, j] * y_eta_eta[i, j])
                q_o[i, j] = (-1 / alpha[i, j]) * (x_eta[i, j] * x_eta_eta[i, j] + (y_eta[i, j] * y_eta_eta[i, j])) - \
                            (1 / gamma[i, j]) * (x_eta[i, j] * x_xi_xi[i, j] + y_eta[i, j] * y_xi_xi[i, j])
    elif type_geom == 'axi':
        for i in list_corners:
            for j in [0, np.shape(x_it)[1] - 1]:
                p_o[i, j], q_o[i, j] = solve_axi_ortho(x_xi[i, j], y_xi[i, j], x_eta[i, j], y_eta[i, j], x_xi_xi[i, j],
                                                       y_xi_xi[i, j], x_eta_eta[i, j], y_eta_eta[i, j], jac_det[i, j],
                                                       alpha[i, j], gamma[i, j], y_it[i, j])

    # interpolate to interior grid using linear transfinite interpolation (TFI)
    eta = np.zeros(np.shape(x_it)[1])
    xi = np.zeros(np.shape(x_it)[0])
    idx = -1

    for k in range(0, np.shape(x_it)[1]):
        val = (x_it[0, k] - np.nanmin(x_it[0, :])) / (np.nanmax(x_it[0, :]) - np.nanmin(x_it[0, :]))
        eta[k] = val
    for k in range(0, np.shape(x_it)[0]):
        val = (y_it[k, -1] - np.nanmin(y_it[:, -1])) / (np.nanmax(y_it[:, -1]) - np.nanmin(y_it[:, -1]))
        xi[k] = val

    p_o[p_o == np.inf] = 0
    p_o[p_o == -np.inf] = 0
    q_o[q_o == np.inf] = 0
    q_o[q_o == -np.inf] = 0
    p_o[np.isnan(p_o)] = 0

    if type_grid == 'slit':
        idx_i1 = n_y - n_y_nacelle - 3
        idx_i2 = n_y - n_y_nacelle - 2
        eta_slit = np.zeros(np.shape(x_it)[0])
        for k in range(0, idx_i1 + 1):
            eta_slit[k] = -1 / (idx_i1 + 1) * k + 1
        for k in range(idx_i1 + 2, np.shape(x_it)[0]):
            eta_slit[k] = -1 / (np.shape(x_it)[0] - 2 - idx_i2) * k + (idx_i2 + 1) / (
                    np.shape(x_it)[0] - 2 - idx_i2) + 1
        # orthogonality at slit
        p_o[n_y - n_y_nacelle - 1, :] = 0
        q_o[n_y - n_y_nacelle - 1, :] = 0
        p_o[n_y - n_y_nacelle - 2, :] = 0
        q_o[n_y - n_y_nacelle - 2, :] = 0

    from finite_differences.mesh.algebraic_grid import AlgebraicGrid
    calc_funcs = AlgebraicGrid(p_o, q_o, 'rect', None, None)
    p_o_test, q_o_test = calc_funcs.run()

    for j in range(1, np.shape(p_o)[1] - 1, 1):
        for i in range(1, np.shape(p_o)[0] - 1, 1):
            if orthogonality == 'eta':
                p_o[i, j] = (1 - eta[j]) * p_o[i, 0] + eta[j] * p_o[i, -1]
                q_o[i, j] = (1 - eta[j]) * q_o[i, 0] + eta[j] * q_o[i, -1]
            elif orthogonality == 'xi':
                p_o[i, j] = (1 - xi[i]) * p_o[idx, j] + xi[i] * p_o[0, j]
                q_o[i, j] = (1 - xi[i]) * q_o[idx, j] + xi[i] * q_o[0, j]
            elif orthogonality == 'xi_and_eta':
                p_o[i, j] = (1 - eta[j]) * p_o[i, 0] + eta[j] * p_o[i, -1] + \
                            (1 - xi[i]) * p_o[idx, j] + xi[i] * p_o[0, j] + \
                            (1 - eta[j]) * p_o[i, 0] + eta[j] * p_o[i, -1] - (eta[j] * xi[i] * p_o[0, -1] + eta[j]
                                                                              * (1 - xi[i]) * p_o[idx, -1] + xi[i]
                                                                              * (1 - eta[j]) * p_o[0, 0] + (1 - eta[j])
                                                                              * (1 - xi[i]) * p_o[idx, 0])
                q_o[i, j] = (1 - eta[j]) * q_o[i, 0] + eta[j] * q_o[i, -1] + \
                            (1 - xi[i]) * q_o[idx, j] + xi[i] * q_o[0, j] + \
                            (1 - eta[j]) * q_o[i, 0] + eta[j] * q_o[i, -1] - (eta[j] * xi[i] * q_o[0, -1] + eta[j]
                                                                              * (1 - xi[i]) * q_o[idx, -1] + xi[i]
                                                                              * (1 - eta[j]) * q_o[0, 0] + (1 - eta[j])
                                                                              * (1 - xi[i]) * q_o[idx, 0])
            else:
                raise Warning('Direction of orthogonality not specified.')

    return p_o, q_o


# Function to calculate ghost points for calculation of orthogonal control functions. Calculate only once at beginning.
# Don't calculate corners.
def ghost_points(x_alg, y_alg, domain='rect-grid'):
    xl = np.zeros(np.shape(x_alg)[0])
    yl = np.zeros(np.shape(x_alg)[0])
    xr = np.zeros(np.shape(x_alg)[0])
    yr = np.zeros(np.shape(x_alg)[0])
    xt = np.zeros(np.shape(x_alg)[1])
    yt = np.zeros(np.shape(x_alg)[1])
    xb = np.zeros(np.shape(x_alg)[1])
    yb = np.zeros(np.shape(x_alg)[1])
    gammat = np.zeros(np.shape(x_alg)[1])
    gammab = np.zeros(np.shape(x_alg)[1])
    gammal = np.zeros(np.shape(x_alg)[0])
    gammar = np.zeros(np.shape(x_alg)[0])
    alphat = np.zeros(np.shape(x_alg)[1])
    alphab = np.zeros(np.shape(x_alg)[1])
    alphal = np.zeros(np.shape(x_alg)[0])
    alphar = np.zeros(np.shape(x_alg)[0])
    x_xi_l = np.zeros(np.shape(x_alg)[0])
    y_xi_l = np.zeros(np.shape(x_alg)[0])
    x_eta_l = np.zeros(np.shape(x_alg)[0])
    y_eta_l = np.zeros(np.shape(x_alg)[0])
    x_xi_r = np.zeros(np.shape(x_alg)[0])
    y_xi_r = np.zeros(np.shape(x_alg)[0])
    x_eta_r = np.zeros(np.shape(x_alg)[0])
    y_eta_r = np.zeros(np.shape(x_alg)[0])
    x_xi_t = np.zeros(np.shape(x_alg)[1])
    y_xi_t = np.zeros(np.shape(x_alg)[1])
    x_eta_t = np.zeros(np.shape(x_alg)[1])
    y_eta_t = np.zeros(np.shape(x_alg)[1])
    x_xi_b = np.zeros(np.shape(x_alg)[1])
    y_xi_b = np.zeros(np.shape(x_alg)[1])
    x_eta_b = np.zeros(np.shape(x_alg)[1])
    y_eta_b = np.zeros(np.shape(x_alg)[1])

    # left boundary
    n = 0
    j = n
    for i in range(1, np.shape(x_alg)[0] - 1, 1):
        x_xi_l[i] = d1o2_cent('xi', x_alg, i, j)
        y_xi_l[i] = d1o2_cent('xi', y_alg, i, j)
        gammal[i] = x_xi_l[i] ** 2 + y_xi_l[i] ** 2
        x_eta_l[i] = d1o1_right('eta', x_alg, i, j)
        y_eta_l[i] = d1o1_right('eta', y_alg, i, j)
        if domain == 'c-grid':
            xl[i] = 2 * x_alg[i, j] - x_alg[i, j + 1]
            yl[i] = 2 * y_alg[i, j] - y_alg[i, j + 1]
        else:
            xl[i] = x_alg[i, n] - ((-y_xi_l[i] / gammal[i] * (-y_xi_l[i] * x_eta_l[i] + x_xi_l[i] * y_eta_l[i])))
            yl[i] = y_alg[i, n] - ((x_xi_l[i] / gammal[i] * (-y_xi_l[i] * x_eta_l[i] + x_xi_l[i] * y_eta_l[i])))
        alphal[i] = ((x_xi_l[i] / gammal[i] * (-y_xi_l[i] * x_eta_l[i] + x_xi_l[i] * y_eta_l[i]))) ** 2 + \
                    ((-y_xi_l[i] / gammal[i] * (-y_xi_l[i] * x_eta_l[i] + x_xi_l[i] * y_eta_l[i]))) ** 2

    # right boundary
    n = np.shape(x_alg)[1] - 1
    j = n
    for i in range(1, np.shape(x_alg)[0] - 1, 1):
        x_xi_r[i] = d1o2_cent('xi', x_alg, i, j)
        y_xi_r[i] = d1o2_cent('xi', y_alg, i, j)
        gammar[i] = x_xi_r[i] ** 2 + y_xi_r[i] ** 2
        x_eta_r[i] = d1o1_left('eta', x_alg, i, j)
        y_eta_r[i] = d1o1_left('eta', y_alg, i, j)
        if domain == 'c-grid':
            xr[i] = 2 * x_alg[i, j] - x_alg[i, j - 1]
            yr[i] = 2 * y_alg[i, j] - y_alg[i, j - 1]
        else:
            xr[i] = x_alg[i, n] + ((-y_xi_r[i] / gammar[i] * (-y_xi_r[i] * x_eta_r[i] + x_xi_r[i] * y_eta_r[i])))
            yr[i] = y_alg[i, n] + ((x_xi_r[i] / gammar[i] * (-y_xi_r[i] * x_eta_r[i] + x_xi_r[i] * y_eta_r[i])))
        alphar[i] = ((x_xi_r[i] / gammar[i] * (-y_xi_r[i] * x_eta_r[i] + x_xi_r[i] * y_eta_r[i]))) ** 2 + \
                    ((-y_xi_r[i] / gammar[i] * (-y_xi_r[i] * x_eta_r[i] + x_xi_r[i] * y_eta_r[i]))) ** 2

    # top boundary
    m = 0
    i = m
    for j in range(1, np.shape(x_alg)[1] - 1, 1):
        x_eta_t[j] = d1o2_cent('eta', x_alg, i, j)
        y_eta_t[j] = d1o2_cent('eta', y_alg, i, j)
        alphat[j] = x_eta_t[j] ** 2 + y_eta_t[j] ** 2
        x_xi_t[j] = d1o1_right('xi', x_alg, i, j)
        y_xi_t[j] = d1o1_right('xi', y_alg, i, j)
        if domain == 'c-grid':
            xt[j] = 2 * x_alg[i, j] - x_alg[i + 1, j]
            yt[j] = 2 * y_alg[i, j] - y_alg[i + 1, j]
        else:
            xt[j] = x_alg[m, j] - ((y_eta_t[j]) / alphat[j] * ((y_eta_t[j] * x_xi_t[j] - x_eta_t[j] * y_xi_t[j])))
            yt[j] = y_alg[m, j] - ((-x_eta_t[j]) / alphat[j] * ((y_eta_t[j] * x_xi_t[j] - x_eta_t[j] * y_xi_t[j])))
        gammat[j] = ((-x_eta_t[j]) / alphat[j] * ((y_eta_t[j] * x_xi_t[j] - x_eta_t[j] * y_xi_t[j]))) ** 2 + \
                    ((y_eta_t[j]) / alphat[j] * ((y_eta_t[j] * x_xi_t[j] - x_eta_t[j] * y_xi_t[j]))) ** 2

    # bottom boundary
    m = np.shape(x_alg)[0] - 1
    i = m
    for j in range(1, np.shape(x_alg)[1] - 1, 1):
        x_eta_b[j] = d1o2_cent('eta', x_alg, i, j)  # p. 204
        y_eta_b[j] = d1o2_cent('eta', y_alg, i, j)  # p. 204
        alphab[j] = x_eta_b[j] ** 2 + y_eta_b[j] ** 2
        x_xi_b[j] = d1o1_left('xi', x_alg, i, j)  # p. 204
        y_xi_b[j] = d1o1_left('xi', y_alg, i, j)  # p. 204
        if domain == 'c-grid':
            xb[j] = 2 * x_alg[i, j] - x_alg[i - 1, j]
            yb[j] = 2 * y_alg[i, j] - y_alg[i - 1, j]
        else:
            xb[j] = x_alg[m, j] + ((y_eta_b[j]) / alphab[j] * (
                (y_eta_b[j] * x_xi_b[j] - x_eta_b[j] * y_xi_b[j])))  # p. 205, eq. 6.12 and following
            yb[j] = y_alg[m, j] + ((-x_eta_b[j]) / alphab[j] * ((y_eta_b[j] * x_xi_b[j] - x_eta_b[j] * y_xi_b[j])))
        gammab[j] = ((-x_eta_b[j]) / alphab[j] * ((y_eta_b[j] * x_xi_b[j] - x_eta_b[j] * y_xi_b[j]))) ** 2 + \
                    ((y_eta_b[j]) / alphab[j] * ((y_eta_b[j] * x_xi_b[j] - x_eta_b[j] * y_xi_b[j]))) ** 2

    top = [xt, yt, x_xi_t, y_xi_t, x_eta_t, y_eta_t, alphat, gammat]
    bottom = [xb, yb, x_xi_b, y_xi_b, x_eta_b, y_eta_b, alphab, gammab]
    left = [xl, yl, x_xi_l, y_xi_l, x_eta_l, y_eta_l, alphal, gammal]
    right = [xr, yr, x_xi_r, y_xi_r, x_eta_r, y_eta_r, alphar, gammar]

    return top, bottom, left, right


# blend initial and orthogonal control functions to calculate control functions for Dirichlet orthogonality
def blend_control_functions(p_i, q_i, p_o, q_o, delta):
    # blending function; delta is a positive number, which controls the exponential decay of the blending function.
    # 0 < delta < 1
    # smaller delta leads to faster convergence (but can't be 0!). suggested value: 0.02

    m = np.shape(p_i)[0]
    n = np.shape(p_i)[1]

    # [1], eq. 6.16
    b = np.array([[np.exp(-(1 / delta) * (i / m) * (j / n) * ((m - 1) / m) * ((n - j) / n))
                   for j in range(0, np.shape(p_i)[1], 1)]
                  for i in range(0, np.shape(p_i)[0], 1)])
    p_b = np.array([[b[i, j] * p_o[i, j] + (1 - b[i, j]) * p_i[i, j]
                     for j in range(0, np.shape(p_i)[1], 1)]
                    for i in range(0, np.shape(p_i)[0], 1)])
    q_b = np.array([[b[i, j] * q_o[i, j] + (1 - b[i, j]) * q_i[i, j]
                     for j in range(0, np.shape(p_i)[1], 1)]
                    for i in range(0, np.shape(p_i)[0], 1)])

    return p_b, q_b


# solve linear system of equations for axisymmetric cases
def solve_axi_ortho(x_xi, y_xi, x_eta, y_eta, x_xi_xi, y_xi_xi, x_eta_eta, y_eta_eta, jac_det, alpha, gamma, y_it):
    a11 = jac_det ** 2 * x_xi
    a12 = jac_det ** 2 * x_eta
    a21 = jac_det ** 2 * y_xi
    a22 = jac_det ** 2 * y_eta
    b1 = - alpha * x_xi_xi - gamma * x_eta_eta
    # if y_it == 0:
    #     b2 = - alpha * y_xi_xi - gamma * y_eta_eta
    # else:
    #     b2 = - alpha * y_xi_xi - gamma * y_eta_eta - jac_det ** 2 / y_it
    b2 = - alpha * y_xi_xi - gamma * y_eta_eta - jac_det ** 2 / y_it
    # solve linear system of equations
    a = np.array([[a11, a12], [a21, a22]])
    b = np.array([b1, b2])
    p, q = np.linalg.solve(a, b)

    return p, q
