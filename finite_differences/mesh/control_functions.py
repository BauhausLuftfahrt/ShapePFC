"""Control functions for grid control.

Author:  A. Habermann

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

# Built-in/Generic Imports
import numpy as np


def thompson_single_line(line_cont, line_i, coeff_a, coeff_c):
    """
        a:              amplitude
        c:              decay coefficient
        line_cont:      control line
        line_i:         controlled line
    """
    return -coeff_a * np.sign(line_i - line_cont) * np.exp(-coeff_c * np.abs(line_i - line_cont))


def thompson_single_line_tan(line_cont, line_i, coeff_a, coeff_c):
    """
        a:              amplitude
        c:              decay coefficient
        line_cont:      control line
        line_i:         controlled line
    """
    return -coeff_a * (2 / np.pi) * np.arctan(100000 * (line_i - line_cont)) * np.exp(
        -coeff_c * np.abs(line_i - line_cont))


def thompson_single_point(point_cont, point_i, coeff_a, coeff_c):
    """
        a:              amplitude
        c:              decay coefficient
        point_cont:      control point
        point_i:         controlled point
    """
    # returns two different values for the p and q function
    return -coeff_a * np.sign(point_i[0] - point_cont[0]) * np.exp(
        -coeff_c * ((point_i[0] - point_cont[0]) ** 2 + (point_i[1] - point_cont[1]) ** 2) ** 0.5), \
           -coeff_a * np.sign(point_i[1] - point_cont[1]) * np.exp(
               -coeff_c * ((point_i[1] - point_cont[1]) ** 2 + (point_i[0] - point_cont[0]) ** 2) ** 0.5)
