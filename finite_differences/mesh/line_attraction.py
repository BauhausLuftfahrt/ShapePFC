"""
Line attraction, especially used within boundary layers.
"""

import numpy as np


def line_attraction_cgrid(x, y):
    eta_attract = np.shape(x)[0]
    xi_attract = 0
    xi_attract2 = int(np.shape(x)[1] / 2)

    xi_p_attract = np.shape(x)[0]
    eta_p_attract = 0

    p = np.zeros((np.shape(x)[0], np.shape(x)[1]))
    q = np.zeros((np.shape(x)[0], np.shape(x)[1]))

    # line attraction
    # c1 = 5000
    idx_i = np.shape(x)[0]
    c1 = np.zeros(idx_i)
    c_start = 2000
    c_stop = 50000
    c3 = 200
    c2 = 0.5  # 0 < c2 < 1
    c4 = 1

    # point attraction
    b1 = 10000
    d1 = 1

    for i in range(0, np.shape(x)[0]):
        c1[i] = c_start + (c_stop - c_start) / (idx_i - 0) * i
        for j in range(0, np.shape(x)[1]):
            # line only
            # q[i,j] = -c1[i]*np.sign(j-xi_attract)*np.exp(-c2*np.abs(j-xi_attract))
            # p[i,j] = 0#-c3*np.sign(i-eta_attract)*np.exp(-c4*np.abs(i-eta_attract))
            # point only
            # q[i,j] = -b1*np.sign(j-xi_p_attract)*np.exp(-d1*np.sqrt((j-xi_p_attract)**2+(i-eta_p_attract)**2))
            # p[i,j] = 0#-b1*np.sign(i-eta_p_attract)*np.exp(-d1*np.sqrt((j-xi_p_attract)**2+(i-eta_p_attract)**2))
            # point and line mix
            # q[i,j] = -c1*np.sign(j-xi_attract)*np.exp(-c2*np.abs(j-xi_attract))-b1*np.sign(j-xi_p_attract)*np.exp(-d1*np.sqrt((j-xi_p_attract)**2+(i-eta_p_attract)**2))
            # p[i,j] = 0#-c3*np.sign(i-eta_attract)*np.exp(-c4*np.abs(i-eta_attract))-b1*np.sign(i-eta_p_attract)*np.exp(-d1*np.sqrt((j-xi_p_attract)**2+(i-eta_p_attract)**2))

            q[i, j] = -c1[i] * np.sign(j - xi_attract2) * np.exp(-c2 * np.abs(j - xi_attract2))
            p[i, j] = 0

    return p, q


def line_attraction_rectgrid(x, y):
    eta_attract = np.shape(x)[0]
    xi_attract = int(np.shape(x)[0] / 2)

    p = np.zeros((np.shape(x)[0], np.shape(x)[1]))
    q = np.zeros((np.shape(x)[0], np.shape(x)[1]))

    # line attraction
    # c1 = 5000
    idx_i = np.shape(x)[0]
    c1 = 1000
    c3 = 35000  # 80000
    c2 = 1.0  # 0 < c2 < 1
    c4 = 1

    for i in range(0, np.shape(x)[0]):
        for j in range(0, np.shape(x)[1]):
            # line only
            q[i, j] = 0  # -c1*np.sign(j-xi_attract)*np.exp(-c2*np.abs(j-xi_attract))
            p[i, j] = -c3 * np.sign(i - eta_attract) * np.exp(-c4 * np.abs(i - eta_attract))

    return p, q


def merge_control_functions_fuselage(p1, q1, p2, q2, x, l_ref):
    # l_ref - reference length, i.e. length of axisymmetric body
    p = np.zeros((np.shape(x)[0], np.shape(x)[1]))
    q = np.zeros((np.shape(x)[0], np.shape(x)[1]))

    idx00 = find_nearest(x[-1, :], -0.75 * l_ref)  # p = p1
    idx01 = find_nearest(x[-1, :], -0.5 * l_ref)  # p = p2
    idx10 = find_nearest(x[-1, :], 1 / 6 * l_ref)  # p = p2
    idx11 = find_nearest(x[-1, :], 1 / 3 * l_ref)  # p = p1

    for j in range(0, np.shape(x)[1]):
        for i in range(0, np.shape(x)[0]):
            if j < idx00 or j > idx11:
                p[i, j] = p1[i, j]
                q[i, j] = q1[i, j]
            elif idx00 <= j <= idx01:
                k = (x[i, j] - x[i, idx00]) / (x[i, idx01] - x[i, idx00])
                p[i, j] = p1[i, j] * (1 - k) + p2[i, j] * k
                q[i, j] = q1[i, j] * (1 - k) + q2[i, j] * k
            elif idx01 < j < idx10:
                p[i, j] = p2[i, j]
                q[i, j] = q2[i, j]
            elif idx10 <= j <= idx11:
                k = (x[i, j] - x[i, idx10]) / (x[i, idx11] - x[i, idx10])
                p[i, j] = p1[i, j] * k + p2[i, j] * (1 - k)
                q[i, j] = q1[i, j] * k + q2[i, j] * (1 - k)

    return p, q


def merge_control_functions_nacelle(p1, q1, p2, q2, x, x_ref_1, x_ref_2):
    # todo: adapt for nacelle with x_refs!
    # l_ref - reference length, i.e. length of axisymmetric body
    l_ref = x_ref_2 - x_ref_1
    p = np.zeros((np.shape(x)[0], np.shape(x)[1]))
    q = np.zeros((np.shape(x)[0], np.shape(x)[1]))

    idx00 = find_nearest(x[-1, :], x_ref_1 - 3 * l_ref)  # p = p1
    idx01 = find_nearest(x[-1, :], x_ref_1 - 1.5 * l_ref)  # p = p2
    idx10 = find_nearest(x[-1, :], x_ref_2 + 1.5 * l_ref)  # p = p2
    idx11 = find_nearest(x[-1, :], x_ref_2 + 3 * l_ref)  # p = p1

    for j in range(0, np.shape(x)[1]):
        for i in range(0, np.shape(x)[0]):
            if j < idx00 or j > idx11:
                p[i, j] = p1[i, j]
                q[i, j] = q1[i, j]
            elif idx00 <= j <= idx01:
                k = (x[i, j] - x[i, idx00]) / (x[i, idx01] - x[i, idx00])
                p[i, j] = p1[i, j] * (1 - k) + p2[i, j] * k
                q[i, j] = q1[i, j] * (1 - k) + q2[i, j] * k
            elif idx01 < j < idx10:
                p[i, j] = p2[i, j]
                q[i, j] = q2[i, j]
            elif idx10 <= j <= idx11:
                k = (x[i, j] - x[i, idx10]) / (x[i, idx11] - x[i, idx10])
                p[i, j] = p1[i, j] * k + p2[i, j] * (1 - k)
                q[i, j] = q1[i, j] * k + q2[i, j] * (1 - k)

    return p, q


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
