"""
Functions required for preparation of parameterized grid generation scripts.

Author:  A. Habermann
"""


import numpy as np
from scipy.optimize import fsolve
from scipy.special import ellipe
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad


def rel_value(n_loc_orig, n_tot_orig, l_delta_orig, l_tot_orig, n_tot_targ, l_delta_targ, l_tot_targ):
    return round((n_loc_orig / n_tot_orig) * (l_tot_orig / l_delta_orig) * (l_delta_targ / l_tot_targ) * n_tot_targ)


def calc_progression_first_length(coeff, curvelength, point_number):
    return curvelength * (coeff - 1) / (coeff ** (point_number - 1) - 1)


def equ_solve_progr_coeff(coeff, first_length, curvelength, point_number):
    return curvelength * (coeff - 1) / (coeff ** (point_number - 1) - 1) - first_length


def calc_progression_coeff(first_length, curvelength, point_number, init_guess=np.array([0.9])):
    coeff = fsolve(equ_solve_progr_coeff, init_guess, args=(first_length, curvelength, point_number))
    return coeff


def quart_ellipse_arclength(a, b):
    eccentricity = 1 - (b / a) ** 2
    return a * ellipe(eccentricity ** 2)


def spline_length(x, y):
    spline = UnivariateSpline(x, y, s=0)
    integrand = lambda t: np.sqrt(1 + spline.derivative()(t) ** 2)
    curve_length, _ = quad(integrand, x[0], x[-1])
    return curve_length


def calc_bump_first_length(coeff, curvelength, point_number):
    if coeff > 1.0:
        a = (-4 * np.sqrt(coeff - 1) * np.arctan2(1, np.sqrt(coeff - 1))) / (curvelength * point_number)
    else:
        a = (2 * np.sqrt(1 - coeff) * np.log(np.abs((1 + 1 / np.sqrt(1 - coeff)) / (1 - 1 / np.sqrt(1 - coeff))))) / (
                    curvelength * point_number)
    b = -a * curvelength ** 2 / (4 * (coeff - 1))
    d = 20  # norm(der)# magnitude of the first derivative of the curve at given position t_. rate of change or slope of the curve at the given position t_. takes into account local steepnass / curvature of the curve
    t = 0  # (t_local-t_begin)/(t_end-t_begin) # curvilinear abscissa; the position along the curve. range: [0,1]. should be 0 for first cell (?)
    x = d / (-a * (t * (curvelength - (curvelength) * 0.5) ** 2) + b)
    return b


# following functions are unused in current mesh generation
def equ_solve_bump_coeff(coeff, first_length, curvelength, point_number):
    return -(2 * np.sqrt(1 - coeff) * np.log(np.abs((1 + 1 / np.sqrt(1 - coeff)) / (1 - 1 / np.sqrt(1 - coeff))))) / \
           (curvelength * point_number) * curvelength ** 2 / (4 * (coeff - 1)) - first_length


def equ_solve_bump_coeff2(coeff, first_length, curvelength, point_number):
    return -(-4 * np.sqrt(coeff - 1) * np.arctan2(1, np.sqrt(coeff - 1))) / (curvelength * point_number) * \
           curvelength ** 2 / (4 * (coeff - 1)) - first_length


def calc_bump_coeff(first_length, curvelength, point_number, init_guess=np.array([0.1])):
    try:
        coeff = fsolve(equ_solve_bump_coeff, init_guess, args=(first_length, curvelength, point_number))
    except:
        coeff = fsolve(equ_solve_bump_coeff2, np.array([1]), args=(first_length, curvelength, point_number))
    return coeff
