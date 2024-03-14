"""Calculate grid spacing.

Author:  A. Habermann

 Args:
    x               array x-coordinates of grid [m]
    y               array y-coordinates of grid [m]

Returns:
    delta_x_low           array spacing of x-coordinates to lower coordinates (left)  [m]
    delta_x_up            array spacing of x-coordinates to upper coordinates (right)  [m]
    delta_y_low           array spacing of y-coordinates to lower coordinates  [m]
    delta_y_up            array spacing of y-coordinates to upper coordinates  [m]

Sources:

"""

# Built-in/Generic Imports
import numpy as np


def calc_spacing(x, y):
    delta_x_low = np.zeros((np.shape(x)[0], np.shape(x)[1]))
    delta_x_up = np.zeros((np.shape(x)[0], np.shape(x)[1]))
    delta_y_low = np.zeros((np.shape(x)[0], np.shape(x)[1]))
    delta_y_up = np.zeros((np.shape(x)[0], np.shape(x)[1]))

    # boundaries, which cannot be calculated
    delta_x_low[:, 0] = np.nan
    delta_x_up[:, -1] = np.nan
    delta_y_low[-1, :] = np.nan
    delta_y_up[0, :] = np.nan

    # inside
    delta_x_low[0:np.shape(x)[0], 1:np.shape(x)[1]] = [[np.abs(x[i, j] - x[i, j - 1]) for j in range(1, np.shape(x)[1])]
                                                       for i in range(0, np.shape(x)[0])]
    delta_x_up[0:np.shape(x)[0], 0:np.shape(x)[1] - 1] = [
        [np.abs(x[i, j + 1] - x[i, j]) for j in range(0, np.shape(x)[1] - 1)]
        for i in range(0, np.shape(x)[0])]
    delta_y_low[0:np.shape(y)[0] - 1, 0:np.shape(y)[1]] = [
        [np.abs(y[i + 1, j] - y[i, j]) for j in range(0, np.shape(y)[1])]
        for i in range(0, np.shape(y)[0] - 1)]
    delta_y_up[1:np.shape(y)[0], 0:np.shape(y)[1]] = [[np.abs(y[i, j] - y[i - 1, j]) for j in range(0, np.shape(y)[1])]
                                                      for i in range(1, np.shape(y)[0])]

    return delta_x_low, delta_x_up, delta_y_low, delta_y_up


if __name__ == "__main__":
    x_test = np.array(
        ([0, 1, 2, 2, 3], [0, 1, 2, 2, 3], [0, 1, 2, 2, 3], [0, 1, 2, 2, 3], [1, 2, 3, 3, 3.5], [1, 2, 3, 3, 4]))
    y_test = np.array(
        ([5, 5, 5, 5, 5], [4, 4, 4, 4, 4], [3, 3, 3, 3, 3], [2, 2, 2, 2, 2], [1, 1, 1.5, 1.5, 1], [0, 0, 1, 1, 0]))
    x_l, x_u, y_l, y_u = calc_spacing(x_test, y_test)
