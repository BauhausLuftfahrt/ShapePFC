""" Calculate velocity potential at specific coordinates away from the fuselage surface from pre-calculated
singularities. Assumption: alpha_inf = 0

Author:  A. Habermann
"""

import numpy as np

from panel.potential_flow.find_sources import findStreamlineVelocitiesSource, findVelocitiesSource


def calculate_vel_pot_line(surface, j_sources, sigma, Xn, Yn, u_inf, x_coords, y_coords):
    V_inf = 1.0
    Vx = np.zeros(len(x_coords))
    Vy = np.zeros(len(x_coords))
    sigma = sigma[sigma != 0]
    for i in range(0, len(x_coords)):
        Wx, Wy = findStreamlineVelocitiesSource(x_coords[i], y_coords[i], surface[0], surface[1], Xn, Yn, surface[3],
                                                surface[2], j_sources)
        Vx[i] = V_inf + np.matmul(Wx, sigma)
        Vy[i] = np.matmul(Wy, sigma)

    return Vx * float(u_inf), Vy * float(u_inf)
