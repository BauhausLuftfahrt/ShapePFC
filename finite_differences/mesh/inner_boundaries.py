"""Generates inner boundary between two subgrids

Author:  A. Habermann

 Args:
        x_subgrid [m]                    x-coordinates of subgrid with adapted boundary (lower subgrid)
        y_subgrid [m]                    y-coordinates of subgrid with adapted boundary (lower subgrid)
        y_bound_top [m]                  y-coordinates of boundary, which will be adapted (upper subgrid)
        x_geom [m]                       x-coordinates of original geometry
        y_geom [m]                       y-coordinates of original geometry

Returns:
    inner_boundary: [2, x] array     x- any y-coordinates of inner boundary

"""

import numpy as np


class InnerBoundary:

    def __init__(self, x_subgrid, y_subgrid, y_bound_top, x_geom, y_geom):
        self.x_subgrid = x_subgrid
        self.y_subgrid = y_subgrid
        self.y_bound_top = y_bound_top
        self.x_geom = x_geom
        self.y_geom = y_geom

    def run(self):
        boundaryx = self.x_subgrid[0,:]
        boundaryy = []

        idx1 = int(np.where(self.x_subgrid[-1,:] == min(self.x_geom[1]))[0])
        idx2 = int(np.where(self.x_subgrid[-1,:] == max(self.x_geom[1]))[0])

        for j in range(0,idx1):
            boundaryy.append(self.y_subgrid[0,:][j])

        for j in range(idx1,idx2+1):
            boundaryy.append(self.y_bound_top[1][j])     # keep upper nacelle geometry

        for j in range(idx2+1,len(self.y_subgrid[0,:])):
            boundaryy.append(self.y_subgrid[0,:][j])

        inner_boundary = np.array([boundaryx, np.array(boundaryy)])

        return inner_boundary
