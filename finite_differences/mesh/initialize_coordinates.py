"""Initialize x- and y-coordinates of transformed grid.

Author:  A. Habermann

 Args:
 
    boundaries: [2, x] array     x- any y-coordinates of grid boundary
    spacing: [1, 2] array        number of points in x- and y-direction
    type: str               Type of mesh. Valid variables: rect, slit, slab
    Xn              [m]     1-D array X-coordinate of geometric profile
    ext_front       [-]     Extension of grid in front of geometry (in percent of max. body length)
    ext_rear       [-]     Extension of grid behind geometry (in percent of max. body length)
    ext_rad       [-]     Radial extension of grid (in percent of max. body height). Must be bigger than 1.

Returns:
    x: array              x-coordinates of transformed coordinates
    y: array              y-coordinates of transformed coordinates

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

import numpy as np


class InitCoordinates:

    def __init__(self, boundaries: list, spacing, type: str, Xn: list, ext_front: float, ext_rear: float,
                 ext_rad: float, boundary_flags: list):
        self.boundaries = boundaries
        self.spacing = spacing
        self.type = type
        self.Xn = Xn
        self.ext_front = ext_front
        self.ext_rear = ext_rear
        self.ext_rad = ext_rad
        self.bound_flags = boundary_flags

    def run(self):

        # n_x = self.spacing[0]
        n_x = len(self.boundaries[2][1])
        if self.type == 'slit':
            # n_y = self.spacing[1] + 1  # for slit: line, connected to slit, needs to be accounted for twice
            n_y = len(self.boundaries[1][1]) + 1
        else:
            # n_y = self.spacing[1]
            n_y = len(self.boundaries[1][1])

        # initialize coordinate matrices and flag matrix
        x = np.zeros((n_y, n_x))
        y = np.zeros((n_y, n_x))
        # flag matrix contains information about the type of node
        # 0: inner point; 1: surface; 2: inlet; 3: outlet; 4: farfield; 5: center axis
        node_flags = np.full((n_y, n_x), 0)

        if self.type == 'rect':
            # populate coordinates matrices with boundary conditions
            x[0, :] = self.boundaries[2][0]
            x[-1, :] = self.boundaries[0][0]
            x[:, 0] = np.transpose(np.flip(self.boundaries[1][0]))
            x[:, -1] = np.transpose(np.flip(self.boundaries[3][0]))

            y[0, :] = self.boundaries[2][1]
            y[-1, :] = self.boundaries[0][1]
            y[:, 0] = np.transpose(np.flip(self.boundaries[1][1]))
            y[:, -1] = np.transpose(np.flip(self.boundaries[3][1]))

            node_flags[-1, :] = self.bound_flags[0]
            node_flags[0, :] = self.bound_flags[2]
            node_flags[:, 0] = np.transpose(np.flip(self.bound_flags[1]))
            node_flags[:, -1] = np.transpose(np.flip(self.bound_flags[3]))

        elif self.type == 'slab':
            # introduce slab to coordinate system. todo: already calculated above, harmonize
            n_x_front = min(int(round(n_x * self.ext_front / (self.ext_front + self.ext_rear + 1), 0)), int(25))
            n_x_rear = 0  # min(int(round(n_x * self.ext_rear / (self.ext_front + self.ext_rear + 1), 0)), int(20))
            n_x_geom = n_x - n_x_rear - n_x_front
            n_y_geom = int(round(0.3 * n_y))

            # populate coordinates matrices with boundary conditions
            x[0, :] = self.boundaries[2][0]  # top
            # bottom including slab
            x[-1, 0:n_x_front] = self.boundaries[0][0][0:n_x_front]
            x[n_y - n_y_geom - 1:-1, n_x_front - 1] = np.flip(self.boundaries[0][0][n_x_front:n_x_front + n_y_geom])
            x[n_y - n_y_geom - 1, n_x_front:n_x_front + n_x_geom] = self.boundaries[0][0][
                                                                    n_x_front + n_y_geom:n_x_front + n_y_geom +
                                                                                         n_x_geom]
            # x[n_y - n_y_geom - 1:-1, n_x_front + n_x_geom] = self.boundaries[0][0][
            #                                                  n_x_front + n_y_geom + n_x_geom:n_x_front + 2 * n_y_geom
            #                                                                                  + n_x_geom]
            # x[-1, n_x_front + n_x_geom:] = self.boundaries[0][0][n_x_front + 2 * n_y_geom + n_x_geom:]
            x[:, 0] = np.transpose(np.flip(self.boundaries[1][0]))  # left
            x[0:-n_y_geom, -1] = np.transpose(np.flip(self.boundaries[3][0]))

            y[0, :] = self.boundaries[2][1]
            # bottom including slab
            y[-1, 0:n_x_front] = self.boundaries[0][1][0:n_x_front]
            y[n_y - n_y_geom - 1:-1, n_x_front - 1] = np.flip(self.boundaries[0][1][n_x_front:n_x_front + n_y_geom])
            y[n_y - n_y_geom - 1, n_x_front:n_x_front + n_x_geom] = self.boundaries[0][1][
                                                                    n_x_front + n_y_geom:n_x_front + n_y_geom +
                                                                                         n_x_geom]
            # y[n_y - n_y_geom - 1:-1, n_x_front + n_x_geom] = self.boundaries[0][1][
            #                                                  n_x_front + n_y_geom + n_x_geom:n_x_front + 2 * n_y_geom + n_x_geom]
            # y[-1, n_x_front + n_x_geom:] = self.boundaries[0][1][n_x_front + 2 * n_y_geom + n_x_geom:]
            y[:, 0] = np.transpose(np.flip(self.boundaries[1][1]))
            y[0:-n_y_geom, -1] = np.transpose(np.flip(self.boundaries[3][1]))

            x[n_y - n_y_geom:n_y, n_x_front:n_x_front + n_x_geom] = np.nan
            y[n_y - n_y_geom:n_y, n_x_front:n_x_front + n_x_geom] = np.nan

            node_flags[0, :] = self.bound_flags[2]  # top
            # bottom including slab
            node_flags[-1, 0:n_x_front] = self.bound_flags[0][0:n_x_front]
            node_flags[n_y - n_y_geom - 1:-1, n_x_front - 1] = np.flip(
                self.bound_flags[0][n_x_front:n_x_front + n_y_geom])
            node_flags[n_y - n_y_geom - 1, n_x_front:n_x_front + n_x_geom] = self.bound_flags[0][
                                                                             n_x_front + n_y_geom:n_x_front + n_y_geom +
                                                                                                  n_x_geom]
            # x[n_y - n_y_geom - 1:-1, n_x_front + n_x_geom] = self.boundaries[0][0][
            #                                                  n_x_front + n_y_geom + n_x_geom:n_x_front + 2 * n_y_geom
            #                                                                                  + n_x_geom]
            # x[-1, n_x_front + n_x_geom:] = self.boundaries[0][0][n_x_front + 2 * n_y_geom + n_x_geom:]
            node_flags[:, 0] = np.transpose(np.flip(self.bound_flags[1]))  # left
            node_flags[0:-n_y_geom, -1] = np.transpose(np.flip(self.bound_flags[3]))

        elif self.type == 'slit' and len(self.Xn) == 2:

            n_y_nacelle = int(round(n_y / 2, 0))

            # populate coordinates matrices with outer boundary conditions
            x[0, :] = self.boundaries[2][0]
            x[-1, :] = self.boundaries[0][0]
            x[0:n_y - n_y_nacelle - 1, 0] = np.transpose(np.flip(self.boundaries[1][0]))[0:n_y - n_y_nacelle - 1]
            x[n_y - n_y_nacelle - 1, 0] = np.transpose(np.flip(self.boundaries[1][0]))[n_y - n_y_nacelle - 2]
            x[n_y - n_y_nacelle:, 0] = np.transpose(np.flip(self.boundaries[1][0]))[n_y - n_y_nacelle - 1:]
            x[0:n_y - n_y_nacelle - 1, -1] = np.transpose(np.flip(self.boundaries[3][0]))[0:n_y - n_y_nacelle - 1]
            x[n_y - n_y_nacelle - 1, -1] = np.transpose(np.flip(self.boundaries[3][0]))[n_y - n_y_nacelle - 2]
            x[n_y - n_y_nacelle:, -1] = np.transpose(np.flip(self.boundaries[3][0]))[n_y - n_y_nacelle - 1:]

            y[0, :] = self.boundaries[2][1]
            y[-1, :] = self.boundaries[0][1]
            y[0:n_y - n_y_nacelle - 1, 0] = np.transpose(np.flip(self.boundaries[1][1]))[0:n_y - n_y_nacelle - 1]
            y[n_y - n_y_nacelle - 1, 0] = np.transpose(np.flip(self.boundaries[1][1]))[n_y - n_y_nacelle - 2]
            y[n_y - n_y_nacelle:, 0] = np.transpose(np.flip(self.boundaries[1][1]))[n_y - n_y_nacelle - 1:]
            y[0:n_y - n_y_nacelle - 1, -1] = np.transpose(np.flip(self.boundaries[3][1]))[0:n_y - n_y_nacelle - 1]
            y[n_y - n_y_nacelle - 1, -1] = np.transpose(np.flip(self.boundaries[3][1]))[n_y - n_y_nacelle - 2]
            y[n_y - n_y_nacelle:, -1] = np.transpose(np.flip(self.boundaries[3][1]))[n_y - n_y_nacelle - 1:]

            idx1 = int(np.where(self.boundaries[0][0] == self.boundaries[5][0][0])[0])
            idx2 = int(np.where(self.boundaries[0][0] == self.boundaries[5][0][-1])[0])

            # populate coordinates matrices with inner boundary conditions
            x[n_y - n_y_nacelle - 2, idx1:idx2 + 1] = self.boundaries[5][0]  # upper
            x[n_y - n_y_nacelle - 1, idx1:idx2 + 1] = self.boundaries[4][0]  # lower
            y[n_y - n_y_nacelle - 2, idx1:idx2 + 1] = self.boundaries[5][1]  # upper
            y[n_y - n_y_nacelle - 1, idx1:idx2 + 1] = self.boundaries[4][1]  # lower

            node_flags[0, :] = self.bound_flags[2]
            node_flags[-1, :] = self.bound_flags[0]
            node_flags[0:n_y - n_y_nacelle - 1, 0] = np.transpose(np.flip(self.bound_flags[1]))[0:n_y - n_y_nacelle - 1]
            node_flags[n_y - n_y_nacelle - 1, 0] = np.transpose(np.flip(self.bound_flags[1]))[n_y - n_y_nacelle - 2]
            node_flags[n_y - n_y_nacelle:, 0] = np.transpose(np.flip(self.bound_flags[1]))[n_y - n_y_nacelle - 1:]
            node_flags[0:n_y - n_y_nacelle - 1, -1] = np.transpose(np.flip(self.bound_flags[3]))[
                                                      0:n_y - n_y_nacelle - 1]
            node_flags[n_y - n_y_nacelle - 1, -1] = np.transpose(np.flip(self.bound_flags[3]))[n_y - n_y_nacelle - 2]
            node_flags[n_y - n_y_nacelle:, -1] = np.transpose(np.flip(self.bound_flags[3]))[n_y - n_y_nacelle - 1:]
            node_flags[n_y - n_y_nacelle - 2, idx1:idx2 + 1] = self.bound_flags[5]
            node_flags[n_y - n_y_nacelle - 1, idx1:idx2 + 1] = self.bound_flags[4]

        elif self.type == 'slit' and len(self.Xn) == 3:

            n_y_nacelle = int(round(1 / 2 * n_y)) - 1

            # populate coordinates matrices with outer boundary conditions
            x[0, :] = self.boundaries[2][0]
            x[-1, :] = self.boundaries[0][0]
            x[0:n_y - n_y_nacelle - 1, 0] = np.transpose(np.flip(self.boundaries[1][0]))[0:n_y - n_y_nacelle - 1]
            x[n_y - n_y_nacelle - 1, 0] = np.transpose(np.flip(self.boundaries[1][0]))[n_y - n_y_nacelle - 2]
            x[n_y - n_y_nacelle:, 0] = np.transpose(np.flip(self.boundaries[1][0]))[n_y - n_y_nacelle - 1:]
            x[0:n_y - n_y_nacelle - 1, -1] = np.transpose(np.flip(self.boundaries[3][0]))[0:n_y - n_y_nacelle - 1]
            x[n_y - n_y_nacelle - 1, -1] = np.transpose(np.flip(self.boundaries[3][0]))[n_y - n_y_nacelle - 2]
            x[n_y - n_y_nacelle:, -1] = np.transpose(np.flip(self.boundaries[3][0]))[n_y - n_y_nacelle - 1:]

            y[0, :] = self.boundaries[2][1]
            y[-1, :] = self.boundaries[0][1]
            y[0:n_y - n_y_nacelle - 1, 0] = np.transpose(np.flip(self.boundaries[1][1]))[0:n_y - n_y_nacelle - 1]
            y[n_y - n_y_nacelle - 1, 0] = np.transpose(np.flip(self.boundaries[1][1]))[n_y - n_y_nacelle - 2]
            y[n_y - n_y_nacelle:, 0] = np.transpose(np.flip(self.boundaries[1][1]))[n_y - n_y_nacelle - 1:]
            y[0:n_y - n_y_nacelle - 1, -1] = np.transpose(np.flip(self.boundaries[3][1]))[0:n_y - n_y_nacelle - 1]
            y[n_y - n_y_nacelle - 1, -1] = np.transpose(np.flip(self.boundaries[3][1]))[n_y - n_y_nacelle - 2]
            y[n_y - n_y_nacelle:, -1] = np.transpose(np.flip(self.boundaries[3][1]))[n_y - n_y_nacelle - 1:]

            idx1 = int(np.where(self.boundaries[0][0] == self.boundaries[5][0][0])[0])
            idx2 = int(np.where(self.boundaries[0][0] == self.boundaries[5][0][-1])[0])

            # populate coordinates matrices with inner boundary conditions
            x[n_y - n_y_nacelle - 2, idx1:idx2 + 1] = self.boundaries[5][0]  # upper
            x[n_y - n_y_nacelle - 1, idx1:idx2 + 1] = self.boundaries[4][0]  # lower
            y[n_y - n_y_nacelle - 2, idx1:idx2 + 1] = self.boundaries[5][1]  # upper
            y[n_y - n_y_nacelle - 1, idx1:idx2 + 1] = self.boundaries[4][1]  # lower

            node_flags[0, :] = self.bound_flags[2]
            node_flags[-1, :] = self.bound_flags[0]
            node_flags[:, 0] = np.append(np.transpose(np.flip(self.bound_flags[1])), [self.bound_flags[1][0]])
            node_flags[:, -1] = np.append(np.transpose(np.flip(self.bound_flags[1])), [self.bound_flags[3][0]])
            node_flags[n_y - n_y_nacelle - 2, idx1:idx2 + 1] = self.bound_flags[5]  # upper
            node_flags[n_y - n_y_nacelle - 1, idx1:idx2 + 1] = self.bound_flags[4]  # lower

        else:
            raise Warning("Mesh type not specified.")

        return x, y, node_flags
