"""Generates a body-fitted curvilinear coordinate system (mesh) around the geometry for the finite difference
calculation. Based on elliptic PDEs. Transforms coordinates to a rectangular grid (see [4])

Author:  A. Habermann

 Args:
    x_init                array initialized x-coordinates of transformed grid
    y_init                array initialized y-coordinates of transformed grid
    n_x             [-]     Number of x-ccordinates (transformed grid)
    n_y             [-]     Number of y-ccordinates (transformed grid)
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
from finite_differences.mesh.coefficients import calc_coeffs_centered, calc_coeffs_all
from finite_differences.mesh.dirichlet_orthogonality import algebraic_control_functions, initial_control_functions, \
    dirichlet_control_functions, ghost_points
from finite_differences.mesh.line_attraction import line_attraction_cgrid, line_attraction_rectgrid, \
    merge_control_functions_fuselage, \
    merge_control_functions_nacelle


class RectGrid:

    def __init__(self, x_init, y_init, it_max: int, symmetry: str, domain: str, position='', x_nacelle=None):
        self.x = x_init
        self.y = y_init
        self.n_x = np.shape(x_init)[1]
        self.n_y = np.shape(x_init)[0]
        self.it_max = 2  # it_max
        self.symm = symmetry
        self.domain = domain
        self.position = position
        self.x_nacelle = x_nacelle

    def run(self):

        # initialize coefficients, Jacobi determinant and control functions
        alpha = np.zeros((self.n_y, self.n_x))
        beta = np.zeros((self.n_y, self.n_x))
        gamma = np.zeros((self.n_y, self.n_x))
        tau = np.zeros((self.n_y, self.n_x))
        omega = np.zeros((self.n_y, self.n_x))
        jac_det = np.zeros((self.n_y, self.n_x))
        c1 = np.zeros((self.n_y, self.n_x))
        c3 = np.zeros((self.n_y, self.n_x))

        errmax = 1e-5
        new_x = np.copy(self.x)
        new_y = np.copy(self.y)
        err1 = np.zeros((1, self.it_max))
        err2 = np.zeros((1, self.it_max))
        err1[0, 0] = 1
        err2[0, 0] = 1
        t = 1

        # calculate control functions for algebraic grid and initial grid once as input to Dirichlet control function
        # calculation later
        x_alg = copy.deepcopy(self.x)
        y_alg = copy.deepcopy(self.y)
        p_a, q_a = algebraic_control_functions(x_alg, y_alg, self.symm, 'rect')
        p_i, q_i = p_a, q_a  # initial_control_functions(p_a, q_a, self.type, 'rect') -> increases speed to convergence
        # calculate ghost points once
        ghost_top, ghost_bottom, ghost_left, ghost_right = ghost_points(x_alg, y_alg, self.domain)

        if self.domain == 'c-grid' or self.position == 'bottom':
            p_attr, q_attr = line_attraction_rectgrid(new_x, new_y)
            l_ref = x_alg[-1, np.where((y_alg[-1, :] != 0) == True)[0][-1]] - \
                    x_alg[-1, np.where((y_alg[-1, :] != 0) == True)[0][0]]
        elif self.position == 'top':
            p_attr, q_attr = line_attraction_rectgrid(new_x, new_y)
            l_ref2 = self.x_nacelle[-1]
            l_ref1 = self.x_nacelle[0]

        # solution of elliptic PDE in finite difference form with 2nd order central differences
        # elliptic PDE are Laplace equations with control functions p and q
        # acc. to Thompson et al 1977
        while (err1[0, t - 1] > errmax and err2[0, t - 1] > errmax) and t < self.it_max:
            # calculate control functions at every iteration
            if t < 3:
                delta_cont = 0.2
            else:
                delta_cont = 0.05

            p_dir, q_dir = dirichlet_control_functions(new_x, new_y, ghost_top, ghost_bottom, ghost_left, ghost_right,
                                                       p_i, q_i, 'eta', delta_cont, self.symm, 'rect', self.domain)
            # if self.domain != 'c-grid':
            #     q = q_dir
            if self.position == 'bottom':
                p = p_dir
                q = q_dir
            elif self.position == 'top':
                p, q = merge_control_functions_nacelle(p_dir, q_dir, p_attr, q_attr, new_x, l_ref1, l_ref2)
            else:
                p = p_dir
                q = q_dir

            p = p_dir
            q = q_dir

            # calculate coefficients
            coeffs = np.array(
                [calc_coeffs_all(self.x, self.y, i, j) for i in range(0, self.n_y, 1) for j in range(0, self.n_x, 1)])
            alpha = np.reshape([item[0] for item in coeffs], (self.n_y, self.n_x))
            beta = np.reshape([item[1] for item in coeffs], (self.n_y, self.n_x))
            gamma = np.reshape([item[2] for item in coeffs], (self.n_y, self.n_x))
            tau = np.reshape([item[3] for item in coeffs], (self.n_y, self.n_x))
            omega = np.reshape([item[4] for item in coeffs], (self.n_y, self.n_x))
            jac_det = np.reshape([item[5] for item in coeffs], (self.n_y, self.n_x))

            if self.symm == 'axi':
                delta = -1
                c1[1:self.n_y - 1, 1:self.n_x - 1] = np.reshape(
                    [-2 * (alpha[i, j] + gamma[i, j]) for i in range(1, self.n_y - 1, 1) for j in
                     range(1, self.n_x - 1, 1)], (self.n_y - 2, self.n_x - 2))
                c3[1:self.n_y - 1, 1:self.n_x - 1] = np.reshape(
                    [-delta * jac_det[i, j] ** 2 for i in range(1, self.n_y - 1, 1) for j in range(1, self.n_x - 1, 1)],
                    (self.n_y - 2, self.n_x - 2))

            for j in reversed(range(1, self.n_x - 1, 1)):
                for i in reversed(range(1, self.n_y - 1, 1)):
                    if self.symm == 'planar':
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
                    elif self.symm == 'axi':
                        # axisymmetric (x = z, y = r) acc. to [2]
                        c2 = alpha[i, j] * (new_y[i + 1, j] + new_y[i - 1, j]) - 0.5 \
                             * beta[i, j] * (new_y[i + 1, j + 1] - new_y[i + 1, j - 1] + new_y[i - 1, j - 1] - new_y[
                            i - 1, j + 1]) \
                             + gamma[i, j] * (new_y[i, j + 1] + new_y[i, j - 1]) \
                             + (jac_det[i, j] ** 2 / 2) * (p[i, j] * (new_y[i + 1, j] - new_y[i - 1, j])
                                                           + q[i, j] * (new_y[i, j + 1] - new_y[i, j - 1]))

                        new_x[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * (
                                alpha[i, j] * (new_x[i + 1, j] + new_x[i - 1, j])
                                + gamma[i, j] * (new_x[i, j + 1] + new_x[i, j - 1])
                                - 0.5 * beta[i, j] * (new_x[i + 1, j + 1] - new_x[i + 1, j - 1]
                                                      + new_x[i - 1, j - 1] - new_x[i - 1, j + 1])) \
                                      + (jac_det[i, j] ** 2 / (4 * (alpha[i, j] + gamma[i, j]))) * (
                                              p[i, j] * (new_x[i + 1, j] - new_x[i - 1, j])
                                              + q[i, j] * (new_x[i, j + 1] - new_x[i, j - 1]))
                        if np.abs(new_y[i, j]) <= 1e-6:
                            new_y[i, j] = 0
                        elif new_y[i, j] < 0:
                            # one solution of quadratic solution for r, i.e. self.y
                            new_y[i, j] = -c2 / (2 * c1[i, j]) - np.sqrt(
                                (c2 / (2 * c1[i, j])) ** 2 - c3[i, j] / c1[i, j])
                        else:
                            # one solution of quadratic solution for r, i.e. self.y
                            new_y[i, j] = -c2 / (2 * c1[i, j]) + np.sqrt(
                                (c2 / (2 * c1[i, j])) ** 2 - c3[i, j] / c1[i, j])

            # Neumann boundary condition on left and right boundary-> only required if no control function p, q is applied
            # if np.all((p[1:-2,1:-2] == 0)):
            #     new_y[:, -1] = new_y[:, -2]  # right
            #     new_y[:, 0] = new_y[:, 1]  # left

            # optional adaption of upper boundary of lower subgrid with Neumann condition, keeps nacelle and jet shape
            # intact
            if self.position == 'bottom':
                idx1 = int(np.where(self.x[0, :] == min(self.x_nacelle))[0])
                idx2 = int(np.where(self.x[0, :] == max(self.x_nacelle))[0])
                new_x[0, 0:idx1] = new_x[1, 0:idx1]
                # new_x[0,idx2+1:] = new_x[1,idx2+1:]
            if (self.position == 'top' or self.position == '') and self.domain == 'rect-grid':
                new_x[0, :] = new_x[1, :]

            # successive over-relaxation (SOR) to speed up convergence [5]
            if self.symm == 'planar':
                overrelaxation = 1.1  # overrelaxation factor 1 <= overrelaxation <= 2
            elif self.symm == 'axi':
                overrelaxation = 1.2

            new_x = self.x + overrelaxation * (new_x - self.x)
            new_y = self.y + overrelaxation * (new_y - self.y)

            err_x = abs((new_x - self.x) / (np.max(self.x) - np.min(self.x)))
            err_y = abs((new_y - self.y) / (np.max(self.x) - np.min(self.x)))

            err1[0, t] = np.amax(err_x)
            err2[0, t] = np.amax(err_y)

            self.y = np.copy(new_y)
            self.x = np.copy(new_x)
            t += 1

        if t == self.it_max:
            # print("Convergence not reached.")
            status = 'n'
        elif t < self.it_max and (np.amax(new_x) > 10000 or np.amax(new_y) > 10000) or \
                (math.isnan(np.amax(new_x)) or math.isnan(np.amax(new_y))):
            status = 'd'
        else:
            # print("Convergence reached after %s timesteps." % t)
            status = 'c'

        return self.x, self.y, alpha, beta, gamma, tau, omega, jac_det, status
