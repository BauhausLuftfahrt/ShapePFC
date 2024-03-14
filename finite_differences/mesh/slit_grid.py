"""Generates a body-fitted curvilinear coordinate system (mesh) around the geometry for the finite difference
calculation. Based on elliptic PDEs. Transforms coordinates to a grid with a slit for an inner geometry (see 
[4])

Author:  A. Habermann

 Args:
    x_init                array initialized x-coordinates of transformed grid
    y_init                array initialized y-coordinates of transformed grid
    n_x             [-]     Number of x-ccordinates (transformed grid)
    n_y             [-]     Number of y-ccordinates (transformed grid)
    ext_front       [-]     Extension of grid in front of geometry (in percent of max. body length)
    ext_rear       [-]     Extension of grid behind geometry (in percent of max. body length)
    it_max          [-]     Maximum number of iterations
    type: str               Type of grid. Options: (2D) "axi" or "planar"

Returns:
    x                       array final x-coordinates of transformed grid
    y                       array final y-coordinates of transformed grid    

Sources:
    [1] Thompson, Joe F.; Thames, Frank C.; Mastin, C.Wayne: Automatic numerical generation of body-fitted curvilinear 
        coordinate system for field containing any number of arbitrary two-dimensional bodies.
        Journal of Computational Physics 15:3 (1974), 299 - 319.
    [2] Uchikawa, S.: Generation of boundary-fitted curvilinear coordinate systems for a two-dimensional axisymmetric 
        flow problem. Journal of Computational Physics 50:2 (1983), 316 - 321.
    [3] Thompson, Joe F.; Soni, B. K.; Weatherill, N. P.: Handbook of grid generation, CRC Press, Boca Raton (1999).
    [4] Thompson, Joe F.; Warsi, Z. U.; Mastin, C. W.: Numerical grid generation - Foundations and applications, 
        North-Holland, New York (1985).
    [5] Häuser, J.; Xia, Y.; Modern Introduction to Grid Generation. COSMASE Shortcourse Notes. EPF Lausanne (1996).

"""

import numpy as np
import math
import copy
from finite_differences.mesh.dirichlet_orthogonality import algebraic_control_functions, initial_control_functions, \
    dirichlet_control_functions, ghost_points
from finite_differences.mesh.coefficients import calc_alpha, calc_beta, calc_gamma, calc_jac_det, calc_tau, calc_omega, \
    calc_dx, calc_dy


class SlitGrid:

    def __init__(self, x_init, y_init, n_x, n_y, it_max: int, type: str, ext_front: float, ext_rear: float, Xn):
        self.x = x_init
        self.y = y_init
        self.n_x = np.shape(self.x)[1]
        self.n_y = np.shape(self.x)[0]
        self.it_max = it_max
        self.type = type
        self.ext_front = ext_front
        self.ext_rear = ext_rear
        self.Xn = Xn

    def run(self):

        # initialize coefficients, Jacobi determinant and control functions
        alpha = np.zeros((self.n_y, self.n_x))
        beta = np.zeros((self.n_y, self.n_x))
        gamma = np.zeros((self.n_y, self.n_x))
        tau = np.zeros((self.n_y, self.n_x))
        omega = np.zeros((self.n_y, self.n_x))
        jac_det = np.zeros((self.n_y, self.n_x))
        dx = np.zeros((self.n_y, self.n_x))
        dy = np.zeros((self.n_y, self.n_x))

        if len(self.Xn) == 3:
            n_y_nacelle = int(round(1 / 2 * self.n_y)) - 1
        elif len(self.Xn) == 2:
            n_y_nacelle = int(round(self.n_y / 2, 0))

        new_x = np.copy(self.x)
        new_y = np.copy(self.y)
        err1 = np.zeros((1, self.it_max))
        err2 = np.zeros((1, self.it_max))
        err1[0, 0] = 1
        err2[0, 0] = 1
        t = 1

        # calculate control functions for algebraic grid and initial grid once as input to Dirichlet control function calculation later
        x_alg = copy.deepcopy(self.x)
        y_alg = copy.deepcopy(self.y)
        p_a, q_a = algebraic_control_functions(x_alg, y_alg, self.type, 'slit', n_y_nacelle)
        p_i, q_i = initial_control_functions(p_a, q_a, self.type, 'slit', n_y_nacelle)
        # calculate ghost points once
        ghost_top, ghost_bottom, ghost_left, ghost_right = ghost_points(x_alg, y_alg)

        errmax = 10e-5
        # solution of elliptic PDE in finite difference form with 2nd order central differences
        # elliptic PDE are Laplace equations with control functions p and q
        # acc. to Thompson et al 1977
        while (err1[0, t - 1] > errmax and err2[0, t - 1] > errmax) and t < self.it_max:
            p, q = dirichlet_control_functions(new_x, new_y, ghost_top, ghost_bottom, ghost_left, ghost_right, p_i, q_i,
                                               'eta', 0.5, self.type, 'slit',
                                               n_y_nacelle)  # orthogonal_control_functions(new_x, new_y, ghost_top, ghost_bottom, ghost_left, ghost_right, 'eta', self.type, 'slit', n_y_nacelle)
            idx1 = int(np.where(new_x[self.n_y - n_y_nacelle - 2, :] == self.Xn[1][0])[0])  # first point of nacelle
            idx2 = int(np.where(new_x[self.n_y - n_y_nacelle - 2, :] == self.Xn[1][-1])[0])  # last point of nacelle
            # calculate coefficients
            alpha[1:self.n_y - 1, 1:self.n_x - 1] = np.array(
                [[calc_alpha(self.x, self.y, i, j, 'centered') for j in range(1, self.n_x - 1, 1)] for i in
                 range(1, self.n_y - 1, 1)])
            beta[1:self.n_y - 1, 1:self.n_x - 1] = np.array(
                [[calc_beta(self.x, self.y, i, j, 'center-center') for j in range(1, self.n_x - 1, 1)] for i in
                 range(1, self.n_y - 1, 1)])
            gamma[1:self.n_y - 1, 1:self.n_x - 1] = np.array(
                [[calc_gamma(self.x, self.y, i, j, 'centered') for j in range(1, self.n_x - 1, 1)] for i in
                 range(1, self.n_y - 1, 1)])
            dx[1:self.n_y - 1, 1:self.n_x - 1] = np.array(
                [[calc_dx(self.x, self.y, i, j, 'centered', alpha, beta, gamma) for j in range(1, self.n_x - 1, 1)] for
                 i in range(1, self.n_y - 1, 1)])
            dy[1:self.n_y - 1, 1:self.n_x - 1] = np.array(
                [[calc_dy(self.x, self.y, i, j, 'centered', alpha, beta, gamma) for j in range(1, self.n_x - 1, 1)] for
                 i in range(1, self.n_y - 1, 1)])
            jac_det[1:self.n_y - 1, 1:self.n_x - 1] = np.array(
                [[calc_jac_det(self.x, self.y, i, j, 'centered') for j in range(1, self.n_x - 1, 1)] for i in
                 range(1, self.n_y - 1, 1)])
            tau[1:self.n_y - 1, 1:self.n_x - 1] = np.array(
                [[calc_tau(new_x, new_y, i, j, 'centered', jac_det, dx, dy) for j in range(1, self.n_x - 1, 1)] for i in
                 range(1, self.n_y - 1, 1)])
            omega[1:self.n_y - 1, 1:self.n_x - 1] = np.array(
                [[calc_omega(new_x, new_y, i, j, 'centered', jac_det, dx, dy) for j in range(1, self.n_x - 1, 1)] for i
                 in range(1, self.n_y - 1, 1)])

            if len(self.Xn) == 2:
                for j in range(1, self.n_x - 1, 1):
                    for i in range(1, self.n_y - 1, 1):
                        # inner and outer surface of nacelle
                        if i == self.n_y - n_y_nacelle - 1 and (0 < j < idx1 or idx2 < j < self.n_x - 1):
                            new_x[i, j] = new_x[i - 1, j]
                            new_y[i, j] = new_y[i - 1, j]
                        elif (i == self.n_y - n_y_nacelle - 2 or i == self.n_y - n_y_nacelle - 1) and (
                                idx1 <= j <= idx2):
                            new_x[i, j] = new_x[i, j]
                            new_y[i, j] = new_y[i, j]
                        elif i == self.n_y - n_y_nacelle - 2 and (0 < j < idx1 or idx2 < j < self.n_x - 1):
                            # for inner boundary, calculate values using next+1 row
                            beta[i, j] = (1 / 4) * (
                                        ((new_x[i + 2, j] - new_x[i - 1, j]) * (new_x[i, j + 1] - new_x[i, j - 1]))
                                        + ((new_y[i + 2, j] - new_y[i - 1, j]) * (new_y[i, j + 1] - new_y[i, j - 1])))
                            gamma[i, j] = (1 / 4) * ((new_x[i + 1, j] - new_x[i - 1, j]) ** 2 + (
                                        new_y[i + 2, j] - new_y[i - 1, j]) ** 2)
                            jac_det[i, j] = (1 / 4) * (
                                    (new_x[i + 2, j] - new_x[i - 1, j]) * (new_y[i, j + 1] - new_y[i, j - 1])
                                    - (new_x[i, j + 1] - new_x[i, j - 1]) * (new_y[i + 2, j] - new_y[i - 1, j]))
                            if self.type == 'planar':
                                # 2D planar acc. to [3], eq. 6.1
                                new_x[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * \
                                              (alpha[i, j] * (new_x[i + 2, j] + new_x[i - 1, j] + 0.5 * p[i, j] * (
                                                          new_x[i + 2, j] - new_x[i - 1, j])) +
                                               gamma[i, j] * (new_x[i, j + 1] + new_x[i, j - 1] + 0.5 * q[i, j] * (
                                                                  new_x[i, j + 1] - new_x[i, j - 1])) -
                                               0.5 * beta[i, j] * (new_x[i + 2, j + 1] - new_x[i + 2, j - 1] +
                                                                   new_x[i - 1, j - 1] - new_x[i - 1, j + 1]))
                                new_y[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * \
                                              (alpha[i, j] * (new_y[i + 2, j] + new_y[i - 1, j] + 0.5 * p[i, j] * (
                                                          new_y[i + 2, j] - new_y[i - 1, j])) +
                                               gamma[i, j] * (new_y[i, j + 1] + new_y[i, j - 1] + 0.5 * q[i, j] * (
                                                                  new_y[i, j + 1] - new_y[i, j - 1])) -
                                               0.5 * beta[i, j] * (new_y[i + 2, j + 1] - new_y[i + 2, j - 1] +
                                                                   new_y[i - 1, j - 1] - new_y[i - 1, j + 1]))
                            elif self.type == 'axi':
                                # axisymmetric (x = z, y = r) acc. to [2]
                                delta = -1
                                c1 = -2 * (alpha[i, j] + gamma[i, j])
                                c2 = alpha[i, j] * (new_y[i + 2, j] + new_y[i - 1, j]) - 0.5 \
                                     * beta[i, j] * (
                                             new_y[i + 2, j + 1] - new_y[i + 2, j - 1] + new_y[i - 1, j - 1] - new_y[
                                         i - 1, j + 1]) \
                                     + gamma[i, j] * (new_y[i, j + 1] + new_y[i, j - 1]) \
                                     + (jac_det[i, j] ** 2 / 2) * (p[i, j] * (new_y[i + 2, j] - new_y[i - 1, j])
                                                                   + q[i, j] * (new_y[i, j + 1] - new_y[i, j - 1]))
                                c3 = -delta * jac_det[i, j] ** 2

                                new_x[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * (
                                        alpha[i, j] * (new_x[i + 2, j] + new_x[i - 1, j])
                                        + gamma[i, j] * (new_x[i, j + 1] + new_x[i, j - 1])
                                        - 0.5 * beta[i, j] * (new_x[i + 2, j + 1] - new_x[i + 2, j - 1]
                                                              + new_x[i - 1, j - 1] - new_x[i - 1, j + 1])) \
                                              + (jac_det[i, j] ** 2 / (4 * (alpha[i, j] + gamma[i, j]))) * (
                                                      p[i, j] * (new_x[i + 2, j] - new_x[i - 1, j])
                                                      + q[i, j] * (new_x[i, j + 1] - new_x[i, j - 1]))
                                # one solution of quadratic solution for r, i.e. new_y
                                # solution of quadratic solution for r, i.e. new_y different for y > 0 and y < 0
                                if new_y[i, j] < 0:
                                    new_y[i, j] = -c2 / (2 * c1) - np.sqrt((c2 / (2 * c1)) ** 2 - c3 / c1)
                                else:
                                    new_y[i, j] = -c2 / (2 * c1) + np.sqrt((c2 / (2 * c1)) ** 2 - c3 / c1)
                        else:
                            if self.type == 'planar':
                                # 2D planar acc. to [3], eq. 6.1
                                new_x[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * \
                                              (alpha[i, j] * (new_x[i + 1, j] + new_x[i - 1, j] + 0.5 * p[i, j] * (
                                                          new_x[i + 1, j] - new_x[i - 1, j])) +
                                               gamma[i, j] * (new_x[i, j + 1] + new_x[i, j - 1] + 0.5 * q[i, j] * (
                                                                  new_x[i, j + 1] - new_x[i, j - 1])) -
                                               0.5 * beta[i, j] * (new_x[i + 1, j + 1] - new_x[i + 1, j - 1] +
                                                                   new_x[i - 1, j - 1] - new_x[i - 1, j + 1]))
                                new_y[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * \
                                              (alpha[i, j] * (new_y[i + 1, j] + new_y[i - 1, j] + 0.5 * p[i, j] * (
                                                          new_y[i + 1, j] - new_y[i - 1, j])) +
                                               gamma[i, j] * (new_y[i, j + 1] + new_y[i, j - 1] + 0.5 * q[i, j] * (
                                                                  new_y[i, j + 1] - new_y[i, j - 1])) -
                                               0.5 * beta[i, j] * (new_y[i + 1, j + 1] - new_y[i + 1, j - 1] +
                                                                   new_y[i - 1, j - 1] - new_y[i - 1, j + 1]))
                            elif self.type == 'axi':
                                # axisymmetric (x = z, y = r) acc. to [2]
                                delta = -1
                                c1 = -2 * (alpha[i, j] + gamma[i, j])
                                c2 = alpha[i, j] * (new_y[i + 1, j] + new_y[i - 1, j]) - 0.5 \
                                     * beta[i, j] * (new_y[i + 1, j + 1] - new_y[i + 1, j - 1] + new_y[i - 1, j - 1] -
                                                     new_y[
                                                         i - 1, j + 1]) \
                                     + gamma[i, j] * (new_y[i, j + 1] + new_y[i, j - 1]) \
                                     + (jac_det[i, j] ** 2 / 2) * (p[i, j] * (new_y[i + 1, j] - new_y[i - 1, j])
                                                                   + q[i, j] * (new_y[i, j + 1] - new_y[i, j - 1]))
                                c3 = -delta * jac_det[i, j] ** 2

                                new_x[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * (
                                        alpha[i, j] * (new_x[i + 1, j] + new_x[i - 1, j])
                                        + gamma[i, j] * (new_x[i, j + 1] + new_x[i, j - 1])
                                        - 0.5 * beta[i, j] * (new_x[i + 1, j + 1] - new_x[i + 1, j - 1]
                                                              + new_x[i - 1, j - 1] - new_x[i - 1, j + 1])) \
                                              + (jac_det[i, j] ** 2 / (4 * (alpha[i, j] + gamma[i, j]))) * (
                                                      p[i, j] * (new_x[i + 1, j] - new_x[i - 1, j])
                                                      + q[i, j] * (new_x[i, j + 1] - new_x[i, j - 1]))
                                # solution of quadratic solution for r, i.e. new_y different for y > 0 and y < 0
                                if new_y[i, j] < 0:
                                    new_y[i, j] = -c2 / (2 * c1) - np.sqrt((c2 / (2 * c1)) ** 2 - c3 / c1)
                                else:
                                    new_y[i, j] = -c2 / (2 * c1) + np.sqrt((c2 / (2 * c1)) ** 2 - c3 / c1)

            elif len(self.Xn) == 3:
                for j in range(1, self.n_x - 1, 1):
                    for i in range(1, self.n_y - 1, 1):
                        # inner and outer surface of nacelle
                        if i == self.n_y - n_y_nacelle - 1 and (0 < j < idx1 or idx2 < j < self.n_x - 1):
                            new_x[i, j] = new_x[i - 1, j]
                            new_y[i, j] = new_y[i - 1, j]
                        elif (i == self.n_y - n_y_nacelle - 2 or i == self.n_y - n_y_nacelle - 1) and (
                                idx1 <= j <= idx2):
                            new_x[i, j] = new_x[i, j]
                            new_y[i, j] = new_y[i, j]
                        elif i == self.n_y - n_y_nacelle - 2 and (0 < j < idx1 or idx2 < j < self.n_x - 1):
                            # for inner boundary, calculate values using next+1 row
                            beta[i, j] = (1 / 4) * (
                                        ((new_x[i + 2, j] - new_x[i - 1, j]) * (new_x[i, j + 1] - new_x[i, j - 1]))
                                        + ((new_y[i + 2, j] - new_y[i - 1, j]) * (new_y[i, j + 1] - new_y[i, j - 1])))
                            gamma[i, j] = (1 / 4) * ((new_x[i + 2, j] - new_x[i - 1, j]) ** 2 + (
                                        new_y[i + 2, j] - new_y[i - 1, j]) ** 2)
                            jac_det[i, j] = (1 / 4) * (
                                    (new_x[i + 2, j] - new_x[i - 1, j]) * (new_y[i, j + 1] - new_y[i, j - 1])
                                    - (new_x[i, j + 1] - new_x[i, j - 1]) * (new_y[i + 2, j] - new_y[i - 1, j]))
                            if self.type == 'planar':
                                # 2D planar acc. to [3], eq. 6.1
                                new_x[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * \
                                              (alpha[i, j] * (new_x[i + 2, j] + new_x[i - 1, j] + 0.5 * p[i, j] * (
                                                          new_x[i + 2, j] - new_x[i - 1, j])) +
                                               gamma[i, j] * (new_x[i, j + 1] + new_x[i, j - 1] + 0.5 * q[i, j] * (
                                                                  new_x[i, j + 1] - new_x[i, j - 1])) -
                                               0.5 * beta[i, j] * (new_x[i + 2, j + 1] - new_x[i + 2, j - 1] +
                                                                   new_x[i - 1, j - 1] - new_x[i - 1, j + 1]))
                                new_y[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * \
                                              (alpha[i, j] * (new_y[i + 2, j] + new_y[i - 1, j] + 0.5 * p[i, j] * (
                                                          new_y[i + 2, j] - new_y[i - 1, j])) +
                                               gamma[i, j] * (new_y[i, j + 1] + new_y[i, j - 1] + 0.5 * q[i, j] * (
                                                                  new_y[i, j + 1] - new_y[i, j - 1])) -
                                               0.5 * beta[i, j] * (new_y[i + 2, j + 1] - new_y[i + 2, j - 1] +
                                                                   new_y[i - 1, j - 1] - new_y[i - 1, j + 1]))
                            elif self.type == 'axi':
                                # axisymmetric (x = z, y = r) acc. to Uchikawa 1983
                                delta = -1
                                c1 = -2 * (alpha[i, j] + gamma[i, j])
                                c2 = alpha[i, j] * (new_y[i + 2, j] + new_y[i - 1, j]) - 0.5 \
                                     * beta[i, j] * (
                                             new_y[i + 2, j + 1] - new_y[i + 2, j - 1] + new_y[i - 1, j - 1] - new_y[
                                         i - 1, j + 1]) \
                                     + gamma[i, j] * (new_y[i, j + 1] + new_y[i, j - 1]) \
                                     + (jac_det[i, j] ** 2 / 2) * (p[i, j] * (new_y[i + 2, j] - new_y[i - 1, j])
                                                                   + q[i, j] * (new_y[i, j + 1] - new_y[i, j - 1]))
                                c3 = -delta * jac_det[i, j] ** 2

                                new_x[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * (
                                        alpha[i, j] * (new_x[i + 2, j] + new_x[i - 1, j])
                                        + gamma[i, j] * (new_x[i, j + 1] + new_x[i, j - 1])
                                        - 0.5 * beta[i, j] * (new_x[i + 2, j + 1] - new_x[i + 2, j - 1]
                                                              + new_x[i - 1, j - 1] - new_x[i - 1, j + 1])) \
                                              + (jac_det[i, j] ** 2 / (4 * (alpha[i, j] + gamma[i, j]))) * (
                                                      p[i, j] * (new_x[i + 2, j] - new_x[i - 1, j])
                                                      + q[i, j] * (new_x[i, j + 1] - new_x[i, j - 1]))
                                # one solution of quadratic solution for r, i.e. y
                                new_y[i, j] = -c2 / (2 * c1) + np.sqrt((c2 / (2 * c1)) ** 2 - c3 / c1)
                        else:
                            if self.type == 'planar':
                                # 2D planar acc. to [3], eq. 6.1
                                new_x[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * \
                                              (alpha[i, j] * (new_x[i + 1, j] + new_x[i - 1, j] + 0.5 * p[i, j] * (
                                                          new_x[i + 1, j] - new_x[i - 1, j])) +
                                               gamma[i, j] * (new_x[i, j + 1] + new_x[i, j - 1] + 0.5 * q[i, j] * (
                                                                  new_x[i, j + 1] - new_x[i, j - 1])) -
                                               0.5 * beta[i, j] * (new_x[i + 1, j + 1] - new_x[i + 1, j - 1] +
                                                                   new_x[i - 1, j - 1] - new_x[i - 1, j + 1]))
                                new_y[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * \
                                              (alpha[i, j] * (new_y[i + 1, j] + new_y[i - 1, j] + 0.5 * p[i, j] * (
                                                          new_y[i + 1, j] - new_y[i - 1, j])) +
                                               gamma[i, j] * (new_y[i, j + 1] + new_y[i, j - 1] + 0.5 * q[i, j] * (
                                                                  new_y[i, j + 1] - new_y[i, j - 1])) -
                                               0.5 * beta[i, j] * (new_y[i + 1, j + 1] - new_y[i + 1, j - 1] +
                                                                   new_y[i - 1, j - 1] - new_y[i - 1, j + 1]))
                            elif self.type == 'axi':
                                # axisymmetric (x = z, y = r) acc. to Uchikawa 1983
                                delta = -1
                                c1 = -2 * (alpha[i, j] + gamma[i, j])
                                c2 = alpha[i, j] * (new_y[i + 1, j] + new_y[i - 1, j]) - 0.5 \
                                     * beta[i, j] * (
                                             new_y[i + 1, j + 1] - new_y[i + 1, j - 1] + new_y[i - 1, j - 1] - new_y[
                                         i - 1, j + 1]) \
                                     + gamma[i, j] * (new_y[i, j + 1] + new_y[i, j - 1]) \
                                     + (jac_det[i, j] ** 2 / 2) * (p[i, j] * (new_y[i + 1, j] - new_y[i - 1, j])
                                                                   + q[i, j] * (new_y[i, j + 1] - new_y[i, j - 1]))
                                c3 = -delta * jac_det[i, j] ** 2

                                new_x[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * (
                                        alpha[i, j] * (new_x[i + 1, j] + new_x[i - 1, j])
                                        + gamma[i, j] * (new_x[i, j + 1] + new_x[i, j - 1])
                                        - 0.5 * beta[i, j] * (new_x[i + 1, j + 1] - new_x[i + 1, j - 1]
                                                              + new_x[i - 1, j - 1] - new_x[i - 1, j + 1])) \
                                              + (jac_det[i, j] ** 2 / (4 * (alpha[i, j] + gamma[i, j]))) * (
                                                      p[i, j] * (new_x[i + 1, j] - new_x[i - 1, j])
                                                      + q[i, j] * (new_x[i, j + 1] - new_x[i, j - 1]))
                                # one solution of quadratic solution for r, i.e. y
                                new_y[i, j] = -c2 / (2 * c1) + np.sqrt((c2 / (2 * c1)) ** 2 - c3 / c1)

            # Neumann boundary condition on left and right boundary-> only required if no control function p, q is applied
            if np.all((p[1:-2, 1:-2] == 0)):
                new_y[:, -1] = new_y[:, -2]  # right
                new_y[:, 0] = new_y[:, 1]  # left
            if len(self.Xn) == 3:
                new_x[0, :] = new_x[1, :]
                new_y[:, -1] = new_y[:, -2]  # right
                new_y[:, 0] = new_y[:, 1]  # left

            # successive over-relaxation (SOR) to speed up convergence [5]
            overrelaxation = 1.5  # overrelaxation factor 1 <= omega <= 2
            new_x = self.x + overrelaxation * (new_x - self.x)
            new_y = self.y + overrelaxation * (new_y - self.y)

            err_x = abs(new_x - self.x)
            err_y = abs(new_y - self.y)
            err1[0, t] = np.amax(err_x)
            err2[0, t] = np.amax(err_y)

            self.y = np.copy(new_y)
            self.x = np.copy(new_x)
            t += 1

        if t == self.it_max:
            print("Convergence not reached.")
            status = 'n'
        elif t < self.it_max and (np.amax(new_x) > 10000 or np.amax(new_y) > 10000) or \
                (math.isnan(np.amax(new_x)) or math.isnan(np.amax(new_y))):
            status = 'd'
        else:
            print("Convergence reached after %s timesteps." % t)
            status = 'c'

        return self.x, self.y, alpha, beta, gamma, tau, omega, jac_det, status
