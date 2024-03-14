"""Calculate coefficients for grid generation using different FD schemes

Author:  A. Habermann

Sources:
    [1] Thompson, Joe F.; Soni, B. K.; Weatherill, N. P.: Handbook of grid generation, CRC Press, Boca Raton (1999),
    Chapter 6.

"""

from finite_differences.schemes.finite_difference_schemes import *
import numpy as np


def calc_coeffs_centered(x, y, i, j):
    d1o2_eta_x = d1o2_cent('eta', x, i, j)
    d1o2_eta_y = d1o2_cent('eta', y, i, j)
    d1o2_cent_xi_x = d1o2_cent('xi', x, i, j)
    d1o2_cent_eta_x = d1o2_cent('eta', x, i, j)
    d1o2_cent_xi_y = d1o2_cent('xi', y, i, j)
    d1o2_cent_eta_y = d1o2_cent('eta', y, i, j)

    d2o2_cent_xi_x = d2o2_cent("xi", x, i, j)
    d2o2_cent_mix_x = d202_mix_cent(x, i, j)
    d2o2_cent_eta_x = d2o2_cent("eta", x, i, j)

    d2o2_cent_xi_y = d2o2_cent("xi", y, i, j)
    d2o2_cent_mix_y = d202_mix_cent(y, i, j)
    d2o2_cent_eta_y = d2o2_cent("eta", y, i, j)

    alpha = d1o2_eta_x ** 2 + d1o2_eta_y ** 2
    beta = d1o2_cent_xi_x * d1o2_cent_eta_x + d1o2_cent_xi_y * d1o2_cent_eta_y
    gamma = d1o2_cent_xi_x ** 2 + d1o2_cent_xi_y ** 2
    dx = alpha * d2o2_cent_xi_x - 2 * beta * d2o2_cent_mix_x + gamma * d2o2_cent_eta_x
    dy = alpha * d2o2_cent_xi_y - 2 * beta * d2o2_cent_mix_y + gamma * d2o2_cent_eta_y
    jac_det = d1o2_cent_xi_x * d1o2_cent_eta_y - d1o2_cent_eta_x * d1o2_cent_xi_y
    tau = 1 / jac_det * (d1o2_cent_eta_x * dy - d1o2_cent_eta_y * dx)
    omega = 1 / jac_det * (d1o2_cent_xi_y * dx - d1o2_cent_xi_x * dy)

    return alpha, beta, gamma, tau, omega, jac_det


def calc_coeffs_all(x, y, i, j):
    if i == 0:
        d1o2_cent_xi_x = d1o2_right('xi', x, i, j)
        d1o2_cent_xi_y = d1o2_right('xi', y, i, j)
        d2o2_cent_xi_x = d2o2_right("xi", x, i, j)
        d2o2_cent_xi_y = d2o2_right("xi", y, i, j)
        if j == 0:
            d1o2_cent_eta_x = d1o2_right('eta', x, i, j)
            d1o2_cent_eta_y = d1o2_right('eta', y, i, j)
            d2o2_cent_eta_x = d2o1_right("eta", x, i, j)
            d2o2_cent_eta_y = d2o1_right("eta", y, i, j)
            d2o2_cent_mix_x = d202_mix_right_right(x, i, j)
            d2o2_cent_mix_y = d202_mix_right_right(y, i, j)
        elif j == np.shape(x)[1] - 1:
            d1o2_cent_eta_x = d1o2_left('eta', x, i, j)
            d1o2_cent_eta_y = d1o2_left('eta', y, i, j)
            d2o2_cent_eta_x = d2o1_left("eta", x, i, j)
            d2o2_cent_eta_y = d2o1_left("eta", y, i, j)
            d2o2_cent_mix_x = d202_mix_left_right('xi', x, i, j)
            d2o2_cent_mix_y = d202_mix_left_right('xi', y, i, j)
        else:
            d1o2_cent_eta_x = d1o2_cent('eta', x, i, j)
            d1o2_cent_eta_y = d1o2_cent('eta', y, i, j)
            d2o2_cent_eta_x = d2o2_cent("eta", x, i, j)
            d2o2_cent_eta_y = d2o2_cent("eta", y, i, j)
            d2o2_cent_mix_x = d202_mix_cent_right('xi', x, i, j)
            d2o2_cent_mix_y = d202_mix_cent_right('xi', y, i, j)

    elif i == np.shape(x)[0] - 1:
        d1o2_cent_xi_x = d1o2_left('xi', x, i, j)
        d1o2_cent_xi_y = d1o2_left('xi', y, i, j)
        d2o2_cent_xi_x = d2o2_left("xi", x, i, j)
        d2o2_cent_xi_y = d2o2_left("xi", y, i, j)
        if j == 0:
            d1o2_cent_eta_x = d1o2_right('eta', x, i, j)
            d1o2_cent_eta_y = d1o2_right('eta', y, i, j)
            d2o2_cent_eta_x = d2o1_right("eta", x, i, j)
            d2o2_cent_eta_y = d2o1_right("eta", y, i, j)
            d2o2_cent_mix_x = d202_mix_left_right('eta', x, i, j)
            d2o2_cent_mix_y = d202_mix_left_right('eta', y, i, j)
        elif j == np.shape(x)[1] - 1:
            d1o2_cent_eta_x = d1o2_left('eta', x, i, j)
            d1o2_cent_eta_y = d1o2_left('eta', y, i, j)
            d2o2_cent_eta_x = d2o1_left("eta", x, i, j)
            d2o2_cent_eta_y = d2o1_left("eta", y, i, j)
            d2o2_cent_mix_x = d202_mix_left_left(x, i, j)
            d2o2_cent_mix_y = d202_mix_left_left(y, i, j)
        else:
            d1o2_cent_eta_x = d1o2_cent('eta', x, i, j)
            d1o2_cent_eta_y = d1o2_cent('eta', y, i, j)
            d2o2_cent_eta_x = d2o2_cent("eta", x, i, j)
            d2o2_cent_eta_y = d2o2_cent("eta", y, i, j)
            d2o2_cent_mix_x = d202_mix_cent_left('xi', x, i, j)
            d2o2_cent_mix_y = d202_mix_cent_left('xi', y, i, j)
    elif j == 0 and i != (0 and np.shape(x)[0] - 1):
        d1o2_cent_eta_x = d1o2_right('eta', x, i, j)
        d1o2_cent_eta_y = d1o2_right('eta', y, i, j)
        d2o2_cent_eta_x = d2o1_right("eta", x, i, j)
        d2o2_cent_eta_y = d2o1_right("eta", y, i, j)

        d1o2_cent_xi_x = d1o2_cent('xi', x, i, j)
        d1o2_cent_xi_y = d1o2_cent('xi', y, i, j)
        d2o2_cent_xi_x = d2o2_cent("xi", x, i, j)
        d2o2_cent_xi_y = d2o2_cent("xi", y, i, j)

        d2o2_cent_mix_x = d202_mix_cent_right('eta', x, i, j)
        d2o2_cent_mix_y = d202_mix_cent_right('eta', y, i, j)

    elif j == np.shape(x)[1] - 1 and i != (0 and np.shape(x)[0] - 1):
        d1o2_cent_eta_x = d1o2_left('eta', x, i, j)
        d1o2_cent_eta_y = d1o2_left('eta', y, i, j)
        d2o2_cent_eta_x = d2o1_left("eta", x, i, j)
        d2o2_cent_eta_y = d2o1_left("eta", y, i, j)

        d1o2_cent_xi_x = d1o2_cent('xi', x, i, j)
        d1o2_cent_xi_y = d1o2_cent('xi', y, i, j)
        d2o2_cent_xi_x = d2o2_cent("xi", x, i, j)
        d2o2_cent_xi_y = d2o2_cent("xi", y, i, j)

        d2o2_cent_mix_x = d202_mix_cent_left('eta', x, i, j)
        d2o2_cent_mix_y = d202_mix_cent_left('eta', y, i, j)

    else:
        d1o2_cent_xi_x = d1o2_cent('xi', x, i, j)
        d1o2_cent_eta_x = d1o2_cent('eta', x, i, j)
        d1o2_cent_xi_y = d1o2_cent('xi', y, i, j)
        d1o2_cent_eta_y = d1o2_cent('eta', y, i, j)

        d2o2_cent_xi_x = d2o2_cent("xi", x, i, j)
        d2o2_cent_mix_x = d202_mix_cent(x, i, j)
        d2o2_cent_eta_x = d2o2_cent("eta", x, i, j)

        d2o2_cent_xi_y = d2o2_cent("xi", y, i, j)
        d2o2_cent_mix_y = d202_mix_cent(y, i, j)
        d2o2_cent_eta_y = d2o2_cent("eta", y, i, j)

    alpha = d1o2_cent_eta_x ** 2 + d1o2_cent_eta_y ** 2
    beta = d1o2_cent_xi_x * d1o2_cent_eta_x + d1o2_cent_xi_y * d1o2_cent_eta_y
    gamma = d1o2_cent_xi_x ** 2 + d1o2_cent_xi_y ** 2
    dx = alpha * d2o2_cent_xi_x - 2 * beta * d2o2_cent_mix_x + gamma * d2o2_cent_eta_x
    dy = alpha * d2o2_cent_xi_y - 2 * beta * d2o2_cent_mix_y + gamma * d2o2_cent_eta_y
    jac_det = d1o2_cent_xi_y * d1o2_cent_eta_x - d1o2_cent_eta_y * d1o2_cent_xi_x
    tau = 1 / jac_det * (d1o2_cent_eta_x * dy - d1o2_cent_eta_y * dx)
    omega = 1 / jac_det * (d1o2_cent_xi_y * dx - d1o2_cent_xi_x * dy)

    return alpha, beta, gamma, tau, omega, jac_det


# second order centered finite difference calculation of coefficients
def calc_alpha(x, y, i, j, fd_type):
    if fd_type == 'centered' or fd_type == 'top-sided' or fd_type == 'bottom-sided':
        alpha = d1o2_cent('eta', x, i, j) ** 2 + d1o2_cent('eta', y, i, j) ** 2
    elif fd_type == 'right-sided':
        alpha = d1o2_right('eta', x, i, j) ** 2 + d1o2_right('eta', y, i, j) ** 2
    elif fd_type == 'left-sided':
        alpha = d1o2_left('eta', x, i, j) ** 2 + d1o2_left('eta', y, i, j) ** 2
    else:
        raise Warning("Finite difference type not specified.")
    return alpha


def calc_beta(x, y, i, j, fd_type):
    if fd_type == 'center-center':
        beta = d1o2_cent('xi', x, i, j) * d1o2_cent('eta', x, i, j) + d1o2_cent('xi', y, i, j) * d1o2_cent('eta', y, i,
                                                                                                           j)
    elif fd_type == 'left-sided':
        beta = d1o2_cent('xi', x, i, j) * d1o2_left('eta', x, i, j) + d1o2_cent('xi', y, i, j) * d1o2_left('eta', y, i,
                                                                                                           j)
    elif fd_type == 'right-sided':
        beta = d1o2_cent('xi', x, i, j) * d1o2_right('eta', x, i, j) + d1o2_cent('xi', y, i, j) * d1o2_right('eta', y,
                                                                                                             i, j)
    elif fd_type == 'top-sided':
        beta = d1o2_left('xi', x, i, j) * d1o2_cent('eta', x, i, j) + d1o2_left('xi', y, i, j) * d1o2_cent('eta', y, i,
                                                                                                           j)
    elif fd_type == 'bottom-sided':
        beta = d1o2_right('xi', x, i, j) * d1o2_cent('eta', x, i, j) + d1o2_right('xi', y, i, j) * d1o2_cent('eta', y,
                                                                                                             i, j)
    elif fd_type == 'top-left-sided':  # for bottom right corner
        beta = d1o2_left('xi', x, i, j) * d1o2_left('eta', x, i, j) + d1o2_left('xi', y, i, j) * d1o2_left('eta', y, i,
                                                                                                           j)
    elif fd_type == 'top-right-sided':  # for bottom left corner
        beta = d1o2_left('xi', x, i, j) * d1o2_right('eta', x, i, j) + d1o2_left('xi', y, i, j) * d1o2_right('eta', y,
                                                                                                             i, j)
    elif fd_type == 'bottom-left-sided':  # for top right corner
        beta = d1o2_right('xi', x, i, j) * d1o2_left('eta', x, i, j) + d1o2_right('xi', y, i, j) * d1o2_left('eta', y,
                                                                                                             i, j)
    elif fd_type == 'bottom-right-sided':  # for top left corner
        beta = d1o2_right('xi', x, i, j) * d1o2_right('eta', x, i, j) + d1o2_right('xi', y, i, j) * d1o2_right('eta', y,
                                                                                                               i, j)
    else:
        raise Warning("Finite difference type not specified.")
    return beta


def calc_gamma(x, y, i, j, fd_type):
    if fd_type == 'centered' or fd_type == 'left-sided' or fd_type == 'right-sided':
        gamma = d1o2_cent('xi', x, i, j) ** 2 + d1o2_cent('xi', y, i, j) ** 2
    elif fd_type == 'top-sided':
        gamma = d1o2_left('xi', x, i, j) ** 2 + d1o2_left('xi', y, i, j) ** 2
    elif fd_type == 'bottom-sided':
        gamma = d1o2_right('xi', x, i, j) ** 2 + d1o2_right('xi', y, i, j) ** 2
    else:
        raise Warning("Finite difference type not specified.")
    return gamma


def calc_tau(x, y, i, j, fd_type, jac_det, dx, dy):
    if fd_type == 'centered':
        x_eta = d1o2_cent("eta", x, i, j)
        y_eta = d1o2_cent("eta", y, i, j)
    else:
        raise Warning("Calculation method for this finite difference type not yet implemented.")
    tau = 1 / jac_det[i, j] * (x_eta * dy[i, j] - y_eta * dx[i, j])
    return tau


def calc_omega(x, y, i, j, fd_type, jac_det, dx, dy):
    if fd_type == 'centered':
        x_xi = d1o2_cent("xi", x, i, j)
        y_xi = d1o2_cent("xi", y, i, j)
    else:
        raise Warning("Calculation method for this finite difference type not yet implemented.")
    omega = 1 / jac_det[i, j] * (y_xi * dx[i, j] - x_xi * dy[i, j])
    return omega


def calc_jac_det(x, y, i, j, fd_type):
    if fd_type == 'centered':
        jac_det = d1o2_cent('xi', y, i, j) * d1o2_cent('eta', x, i, j) - d1o2_cent('eta', y, i, j) * d1o2_cent('xi', x,
                                                                                                               i, j)
    elif fd_type == 'left-sided':
        jac_det = d1o2_cent('xi', y, i, j) * d1o2_left('eta', x, i, j) - d1o2_left('eta', y, i, j) * d1o2_cent('xi', x,
                                                                                                               i, j)
    elif fd_type == 'right-sided':
        jac_det = d1o2_cent('xi', y, i, j) * d1o2_right('eta', x, i, j) - d1o2_right('eta', y, i, j) * d1o2_cent('xi',
                                                                                                                 x, i,
                                                                                                                 j)
    elif fd_type == 'top-sided':
        jac_det = d1o2_left('xi', y, i, j) * d1o2_cent('eta', x, i, j) - d1o2_left('xi', x, i, j) * d1o2_cent('eta', y,
                                                                                                              i, j)
    elif fd_type == 'bottom-sided':
        jac_det = d1o2_right('xi', y, i, j) * d1o2_cent('eta', x, i, j) - d1o2_right('xi', x, i, j) * d1o2_cent('eta',
                                                                                                                y, i, j)
    elif fd_type == 'top-left-sided':  # for bottom right corner
        jac_det = d1o2_left('xi', y, i, j) * d1o2_left('eta', x, i, j) - d1o2_left('xi', x, i, j) * d1o2_left('eta', y,
                                                                                                              i, j)
    elif fd_type == 'top-right-sided':  # for bottom left corner
        jac_det = d1o2_left('xi', y, i, j) * d1o2_right('eta', x, i, j) - d1o2_left('xi', x, i, j) * d1o2_right('eta',
                                                                                                                y,
                                                                                                                i, j)
    elif fd_type == 'bottom-left-sided':  # for top right corner
        jac_det = d1o2_right('xi', y, i, j) * d1o2_left('eta', x, i, j) - d1o2_right('xi', x, i, j) * d1o2_left('eta',
                                                                                                                y,
                                                                                                                i, j)
    elif fd_type == 'bottom-right-sided':  # for top left corner
        jac_det = d1o2_right('xi', y, i, j) * d1o2_right('eta', x, i, j) - d1o2_right('xi', x, i, j) * d1o2_right('eta',
                                                                                                                  y,
                                                                                                                  i, j)
    else:
        raise Warning("Finite difference type not specified.")
    return jac_det


def calc_dx(x, y, i, j, fd_type, alpha, beta, gamma):
    if fd_type == "centered":
        x_xi_xi = d2o2_cent("xi", x, i, j)
        x_xi_eta = d202_mix_cent(x, i, j)
        x_eta_eta = d2o2_cent("eta", x, i, j)
        dx = alpha[i, j] * x_xi_xi - 2 * beta[i, j] * x_xi_eta + gamma[i, j] * x_eta_eta
    else:
        raise Warning("Calculation method for this finite difference type not yet implemented.")
    return dx


def calc_dy(x, y, i, j, fd_type, alpha, beta, gamma):
    if fd_type == "centered":
        y_xi_xi = d2o2_cent("xi", y, i, j)
        y_xi_eta = d202_mix_cent(y, i, j)
        y_eta_eta = d2o2_cent("eta", y, i, j)
        dy = alpha[i, j] * y_xi_xi - 2 * beta[i, j] * y_xi_eta + gamma[i, j] * y_eta_eta
    else:
        raise Warning("Calculation method for this finite difference type not yet implemented.")
    return dy
