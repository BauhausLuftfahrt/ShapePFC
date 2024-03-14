"""Generates a body-fitted curvilinear coordinate system (mesh) around the geometry for the finite difference
calculation. Based on elliptic PDEs. Transforms coordinates to a single or multiple connected rectangular grids (see [4])

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

import math
import copy
import numpy as np
from finite_differences.mesh.coefficients import calc_alpha, calc_beta, calc_gamma, calc_jac_det, calc_tau, calc_omega, \
    calc_dx, \
    calc_dy
from finite_differences.mesh.dirichlet_orthogonality import algebraic_control_functions, initial_control_functions, \
    dirichlet_control_functions, ghost_points


class SlabGrid:

    def __init__(self, x_init, y_init, n_x, n_y, it_max: int, type: str, ext_front: float, ext_rear: float):
        self.x = x_init
        self.y = y_init
        self.n_x = n_x
        self.n_y = n_y
        self.it_max = it_max
        self.type = type
        self.ext_front = ext_front
        self.ext_rear = ext_rear

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

        errmax = 10e-5

        # introduce slab to corodinate system. todo: already calculated above, harmonize
        n_x_front = min(int(round(self.n_x * self.ext_front / (self.ext_front + self.ext_rear + 1), 0)), int(25))
        n_x_rear = 0  # min(int(round(self.n_x * self.ext_rear / (self.ext_front + self.ext_rear + 1), 0)), int(20))
        n_x_geom = self.n_x - n_x_rear - n_x_front
        n_y_geom = int(round(0.3 * self.n_y))

        # fill slab cells with nan
        self.x[self.n_y - n_y_geom:self.n_y, n_x_front:n_x_front + n_x_geom] = 0
        self.y[self.n_y - n_y_geom:self.n_y, n_x_front:n_x_front + n_x_geom] = 0

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
        p_a, q_a = algebraic_control_functions(x_alg, y_alg, self.type, 'slab')
        p_i, q_i = initial_control_functions(p_a, q_a, self.type, 'slab')
        # calculate ghost points once
        ghost_top, ghost_bottom, ghost_left, ghost_right = ghost_points(x_alg, y_alg)

        # solution of elliptic PDE in finite difference form with 2nd order central differences
        # elliptic PDE are Laplace equations with control functions p and q
        # acc. to Thompson et al 1977
        while (err1[0, t - 1] > errmax and err2[0, t - 1] > errmax) and t < self.it_max:
            # calculate control functions at every iteration
            p, q = dirichlet_control_functions(new_x, new_y, ghost_top, ghost_bottom, ghost_left, ghost_right, p_i, q_i,
                                               'eta', 0.5, self.type, 'rect')
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

            for j in range(1, self.n_x - 1, 1):
                for i in range(1, self.n_y - 1, 1):
                    # ignore coordinates in slab
                    if (n_x_front - 2 < j < n_x_front + n_x_geom + 1) and (self.n_y - n_y_geom - 2 < i < self.n_y):
                        new_x[i, j] = new_x[i, j]
                        new_y[i, j] = new_y[i, j]
                    else:
                        alpha[i, j] = (1 / 4) * (
                                    (new_x[i, j + 1] - new_x[i, j - 1]) ** 2 + (new_y[i, j + 1] - new_y[i, j - 1]) ** 2)
                        beta[i, j] = (1 / 4) * (
                                    ((new_x[i + 1, j] - new_x[i - 1, j]) * (new_x[i, j + 1] - new_x[i, j - 1]))
                                    + ((new_y[i + 1, j] - new_y[i - 1, j]) * (new_y[i, j + 1] - new_y[i, j - 1])))
                        gamma[i, j] = (1 / 4) * (
                                    (new_x[i + 1, j] - new_x[i - 1, j]) ** 2 + (new_y[i + 1, j] - new_y[i - 1, j]) ** 2)
                        jac_det[i, j] = (1 / 4) * (
                                (new_x[i + 1, j] - new_x[i - 1, j]) * (new_y[i, j + 1] - new_y[i, j - 1])
                                - (new_x[i, j + 1] - new_x[i, j - 1]) * (new_y[i + 1, j] - new_y[i - 1, j]))
                        # # control functions
                        # # todo: find well usable control functions, for boundary layer especially
                        # p[i,j] = 0
                        # q[i,j] = 0

                        # for some reason, it doesn't work with axisymmetric equations acc. to Uchikawa
                        # 2D planar acc. to Thompson et al 1977
                        new_x[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * (
                                    alpha[i, j] * (new_x[i + 1, j] + new_x[i - 1, j])
                                    + gamma[i, j] * (new_x[i, j + 1] + new_x[i, j - 1])
                                    - 0.5 * beta[i, j] * (new_x[i + 1, j + 1] - new_x[i + 1, j - 1]
                                                          + new_x[i - 1, j - 1] - new_x[i - 1, j + 1])) \
                                      + (jac_det[i, j] ** 2 / (4 * (alpha[i, j] + gamma[i, j]))) * (
                                                  p[i, j] * (new_x[i + 1, j] - new_x[i - 1, j])
                                                  + q[i, j] * (new_x[i, j + 1] - new_x[i, j - 1]))
                        new_y[i, j] = (1 / (2 * alpha[i, j] + 2 * gamma[i, j])) * (
                                    alpha[i, j] * (new_y[i + 1, j] + new_y[i - 1, j])
                                    + gamma[i, j] * (new_y[i, j + 1] + new_y[i, j - 1])
                                    - 0.5 * beta[i, j] * (new_y[i + 1, j + 1] - new_y[i + 1, j - 1]
                                                          + new_y[i - 1, j - 1] - new_y[i - 1, j + 1])) \
                                      + (jac_det[i, j] ** 2 / (4 * (alpha[i, j] + gamma[i, j]))) * (
                                                  p[i, j] * (new_y[i + 1, j] - new_y[i - 1, j])
                                                  + q[i, j] * (new_y[i, j + 1] - new_y[i, j - 1]))

            # Neumann boundary condition on left and right boundary-> only required if no control function p, q is applied
            if np.all((p[1:-2, 1:-2] == 0)):
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

        # fill slab cells with nan
        self.x[self.n_y - n_y_geom:self.n_y, n_x_front:n_x_front + n_x_geom] = np.nan
        self.y[self.n_y - n_y_geom:self.n_y, n_x_front:n_x_front + n_x_geom] = np.nan

        if t == self.it_max:
            print("Convergence not reached.")
            status = 'n'
        elif t < self.it_max and (np.amax(new_x) > 10000 or np.amax(new_y) > 10000) or \
                (math.isnan(np.amax(new_x)) or math.isnan(np.amax(new_y))):
            status = 'd'
        else:
            print("Convergence reached after %s timesteps." % t)
            status = 'c'

        return self.x, self.y, alpha, beta, gamma, tau, omega, status
