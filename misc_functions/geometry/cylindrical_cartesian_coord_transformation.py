"""
Transformation of coordiantes from cylindrical to cartesian coordinate system.

Author:  A. Habermann
"""

import numpy as np


def cylindrical_to_cartesian(r, theta, z):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


def cylindrical_to_cartesian_rotate(r, theta, z):
    x_new = z
    z_new = r * np.cos(theta)
    y_new = r * np.sin(theta)
    return x_new, y_new, z_new
