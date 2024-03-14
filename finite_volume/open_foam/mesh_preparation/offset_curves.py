"""
Offset curves by provided distance. Required for preparation of parameterized grid generation scripts.

Author:  A. Habermann
"""

from shapely.geometry import LineString
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import shapely


def find_intersections(x1, y1, x2, y2):
    curve1 = LineString(zip(x1, y1))
    curve2 = LineString(zip(x2, y2))

    intersections = curve1.intersection(curve2)

    if intersections.is_empty:
        return []

    if intersections.geom_type == "Point":
        return [(intersections.x, intersections.y)]


def offset_curve(x, y, h):
    # Calculate the tangent vector
    dx = np.gradient(x)
    dy = np.gradient(y)

    tangent_angle = np.arctan2(dy, dx)
    # Calculate the unit normal vector
    normal_angle = tangent_angle + np.pi / 2
    normal_vector = np.column_stack((np.cos(normal_angle), np.sin(normal_angle)))
    # Calculate the offset curve
    offset_x = x + h * normal_vector[:, 0]
    offset_y = y + h * normal_vector[:, 1]

    return offset_x, offset_y


def extrapolate_curve_fuselage(x, y, x_end):
    le_angle = np.arctan((x[1] - x[0]) / (y[1] - y[0]))
    y_le = 0.
    x_le = x[1] - np.tan(le_angle) * (y[1] - y_le)

    interp = interp1d(x, y)
    y_end = interp(x_end)

    idx = 0
    for i in range(0, len(x)):
        if x[i] > x_end:
            idx = i
            break

    x = x[0:idx]
    y = y[0:idx]

    x = np.insert(x, 0, x_le)
    x = np.append(x, x_end)
    y = np.insert(y, 0, y_le)
    y = np.append(y, y_end)

    return x, y


def extrapolate_curve_inlet(x, y):
    le_angle = np.arctan((x[1] - x[0]) / (y[1] - y[0]))
    y_le = 0.
    x_le = x[1] - np.tan(le_angle) * (y[1] - y_le)

    x = np.insert(x, 0, x_le)
    y = np.insert(y, 0, y_le)

    return x, y


def extrapolate_curve_rear(x, y, x_end):
    interp = interp1d(x, y)
    y_end = interp(x_end)
    idx = 0
    for i in range(0, len(x)):
        if x[i] > x_end:
            idx = i
            break

    x = x[0:idx]
    y = y[0:idx]
    x = np.append(x, x_end)
    y = np.append(y, y_end)
    return x, y


def interpolate_between_curves(x1, y1, x2, y2):
    interp1 = interp1d(x1, y1)
    interp2 = interp1d(x2, y2)
    x_new = np.linspace(max(min(x1), min(x2)), min(max(x1), max(x2)))
    y_interp = interp1(x_new) + 0.5 * (interp2(x_new) - interp1(x_new))
    return x_new, y_interp


def scale_normalize_span(x, y):
    y_end = 1.
    y_start = 0.
    y_scale = [(y[i] - y[0]) / (y[-1] - y[0]) * (y_end - y_start) for i in range(0, len(y))]
    x_scale = x
    return x_scale, y_scale


def scale(x, x_start, x_end):
    return [(x[i] - x[0]) * (x_end - x_start) + x_start for i in range(0, len(x))]


def interpolate_between_curves_vertical(x1, y1, x2, y2, y_start, y_end):
    x_scale_1, y_scale_1 = scale_normalize_span(x1, y1)
    x_scale_2, y_scale_2 = scale_normalize_span(x2, y2)
    interp1 = interp1d(y_scale_1, x_scale_1)
    interp2 = interp1d(y_scale_2, x_scale_2)
    y_new = np.linspace(0, 1, 100)
    x_interp = interp1(y_new) + 0.5 * (interp2(y_new) - interp1(y_new))
    y_interp = scale(y_new, y_start, y_end)
    return x_interp, y_interp


def insert_coordinate(x, y, coord):
    # Calculate the Euclidean distances between each curve point and the provided coordinate
    distances = cdist(np.array([coord]), np.column_stack((x, y)))
    distances = distances.flatten()

    # Find the indices of the two closest points
    closest_indices = np.argsort(distances)[:2]

    # Insert the provided coordinate at the appropriate position
    x_insert = np.insert(x, closest_indices[1], coord[0])
    y_insert = np.insert(y, closest_indices[1], coord[1])

    x_insert, y_insert = sort_arrays(x_insert, y_insert)

    return list(x_insert), list(y_insert)


def insert_and_cut_array(x, y, coord):
    x_ins, y_ins = insert_coordinate(x, y, coord)
    x_ins, y_ins = sort_arrays(x_ins, y_ins)
    idx_insert = int(np.where(x_ins == coord[0])[0][0])
    x_cut = x_ins[0:idx_insert + 1]
    y_cut = y_ins[0:idx_insert + 1]
    return x_cut, y_cut


def insert_and_cut_array_reverse(x, y, coord):
    x_ins, y_ins = insert_coordinate(x, y, coord)
    x_ins, y_ins = sort_arrays(x_ins, y_ins)
    idx_insert = int(np.where(x_ins == coord[0])[0][0])
    x_cut = x_ins[idx_insert:]
    y_cut = y_ins[idx_insert:]
    return x_cut, y_cut


def sort_arrays(array1, array2):
    combined = list(zip(array1, array2))
    sorted_combined = sorted(combined, key=lambda x: x[0])
    sorted_array1, sorted_array2 = zip(*sorted_combined)
    return sorted_array1, sorted_array2


def interpolate_and_cut_section(x, y, x_start, x_end, no_points):
    interp = interp1d(x, y)
    x_new = np.linspace(x_start, x_end, no_points)
    y_new = interp(x_new)
    return x_new, y_new


def intersect_and_cut_ff_stage(horiz_line_1, horiz_line_2, list_vert_lines):
    lines_vert = []
    intersections = []
    for i in range(0, len(list_vert_lines)):
        int_up = find_intersections(horiz_line_1[0], horiz_line_1[1], list_vert_lines[i][0], list_vert_lines[i][1])[0]
        int_bot = find_intersections(horiz_line_2[0], horiz_line_2[1], list_vert_lines[i][0], list_vert_lines[i][1])[0]
        x_ins_up, y_ins_up = insert_coordinate(list_vert_lines[i][0], list_vert_lines[i][1], int_up)
        y_ins_up, x_ins_up = sort_arrays(y_ins_up, x_ins_up)
        x_ins_bot, y_ins_bot = insert_coordinate(x_ins_up, y_ins_up, int_bot)
        y_ins_bot, x_ins_bot = sort_arrays(y_ins_bot, x_ins_bot)
        idx_bot = int(np.where(y_ins_bot == int_bot[1])[0][0])
        idx_up = int(np.where(y_ins_bot == int_up[1])[0][0])
        if idx_bot < idx_up:
            y_new = y_ins_bot[idx_bot:idx_up]
            x_new = x_ins_bot[idx_bot:idx_up]
        else:
            y_new = y_ins_bot[idx_up:idx_bot]
            x_new = x_ins_bot[idx_up:idx_bot]
        lines_vert.append([x_new, y_new])
        horiz_line_1[0], horiz_line_1[1] = insert_coordinate(horiz_line_1[0], horiz_line_1[1], int_up)
        horiz_line_2[0], horiz_line_2[1] = insert_coordinate(horiz_line_2[0], horiz_line_2[1], int_bot)
        intersections.append([int_up, int_bot])

    horiz_line_1_x, horiz_line_1_y = sort_arrays(horiz_line_1[0], horiz_line_1[1])
    horiz_line_2_x, horiz_line_2_y = sort_arrays(horiz_line_2[0], horiz_line_2[1])

    indices_up = []
    indices_bot = []

    for i in intersections:
        indices_up.append(int(np.where(horiz_line_1_x == i[0][0])[0][0]))
        indices_bot.append(int(np.where(horiz_line_2_x == i[1][0])[0][0]))

    return lines_vert, horiz_line_1_x, horiz_line_1_y, horiz_line_2_x, horiz_line_2_y, indices_up, indices_bot
