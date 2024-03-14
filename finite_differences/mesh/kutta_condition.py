"""
Calculate y-coordinates behind sharp trailing edge of lifting body to ensure Kutta condition for potential flow calculations.
"""

import numpy as np


def kutta_y(x_up, x_low, y_up, y_low, x_kutta):
    """
    x_up / x_low        X-coordinates of nacelle upper and lower side
    y_up / y_low        Y-coordinates of nacelle upper and lower side
    x_kutta:            Array of y-coordinates of grid for which Kutta condition should be applied
"""
    alpha_up = np.arctan((y_up[-2] - y_up[-1]) / (x_up[-1] - x_up[-2]))
    alpha_low = np.arctan((y_low[-2] - y_low[-1]) / (x_low[-1] - x_low[-2]))

    alpha_kutta = (alpha_up + alpha_low) / 2
    y_kutta = [0] * len(x_kutta)

    x_kutta = np.insert(x_kutta, 0, x_low[-1])

    for i in range(1, len(x_kutta)):
        delta_x = x_kutta[i] - x_kutta[0]
        y_kutta[i - 1] = y_low[-1] - delta_x * np.tan(alpha_kutta)

    return y_kutta
