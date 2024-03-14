"""
Calculate progressions of points along a line. Required for preparation of parameterized grid generation scripts.

Author:  A. Habermann
"""

import numpy as np
from sympy import symbols, Eq, solve
from scipy.optimize import root


def calc_len_from_no_and_first_cell(no_cells, first_cell_height, height_ratio):
    return first_cell_height * (height_ratio ** no_cells - 1) / (height_ratio - 1)


def calc_points(no_cells, first_cell_height, height_ratio):
    x = [0]
    for i in range(0, no_cells):
        x.append(first_cell_height * height_ratio ** i + x[-1])
    return x


def calc_ratio_and_cellno_from_first_and_last_cell(height_first_cell, height_last_cell, total_length):
    # height_ratio, no_cells
    r, n = symbols('r n')

    # Define the equations
    equation1 = Eq(height_first_cell * (r ** n - 1) / (r - 1), total_length)
    equation2 = Eq(height_first_cell * r ** n, height_last_cell)

    # Solve the system of equations
    solution = solve((equation1, equation2), (r, n))
    no_cells = int(solution[0][1])

    # correct height_ratio, so that total_length constraint is kept (will change heights of first and last cells a little)
    height_ratio = calc_ratio_from_cellno_and_first_cell(height_first_cell, no_cells, total_length,
                                                         float(solution[0][0]))

    return height_ratio, no_cells


def calc_ratio_from_cellno_and_first_cell(height_first_cell, no_cells, total_length, initial_value):
    def equation(height_ratio):
        return height_first_cell * (height_ratio ** no_cells - 1) / (height_ratio - 1) - total_length

    solution = root(equation, x0=initial_value, method='lm')

    height_ratio = solution.x[0]
    return height_ratio


def calc_last_cell_height(height_first_cell, height_ratio, no_cells):
    return height_first_cell * height_ratio ** no_cells


def calculate_arc_length(x_coordinates, y_coordinates):
    if len(x_coordinates) != len(y_coordinates):
        raise ValueError("Number of x-coordinates and y-coordinates should be the same.")

    n = len(x_coordinates)
    arc_length = 0.0

    for i in range(1, n):
        x1, y1 = x_coordinates[i - 1], y_coordinates[i - 1]
        x2, y2 = x_coordinates[i], y_coordinates[i]

        dx = x2 - x1
        dy = y2 - y1

        segment_length = np.sqrt(dx ** 2 + dy ** 2)
        arc_length += segment_length

    return arc_length
