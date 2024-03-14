"""Generates an initial algebraic grid by linear transfinite interpolation.

Author:  A. Habermann

 Args:
    x_init                array initialized x-coordinates of transformed grid
    y_init                array initialized y-coordinates of transformed grid
    n_x             [-]     Number of x-ccordinates (transformed grid)
    n_y             [-]     Number of y-ccordinates (transformed grid)
    it_max          [-]     Maximum number of iterations
    type: str               Type of grid. Options: (2D) "axi" or "planar"

Returns:
    x                       array final x-coordinates of transformed grid
    y                       array final y-coordinates of transformed grid    

Sources:
    [1] Thompson, Joe F.; Soni, B. K.; Weatherill, N. P.: Handbook of grid generation, CRC Press, Boca Raton (1999), chapter 3.

"""

import numpy as np


class AlgebraicGrid:

    def __init__(self, x_bound, y_bound, type, X, Y):
        self.x = x_bound
        self.y = y_bound
        self.type = type
        self.x_geom = X
        self.y_geom = Y

    def run(self):
        x = np.copy(self.x)
        y = np.copy(self.y)
        if self.type == 'rect' and np.all(y[:, 0] == 0):
            y[:, 0] = np.nan
        xi = np.zeros(np.shape(self.x)[1])
        eta = np.zeros(np.shape(self.x)[0])
        for k in range(0, np.shape(self.x)[1]):
            val = (self.x[0, k] - np.nanmin(self.x[0, :])) / (np.nanmax(self.x[0, :]) - np.nanmin(self.x[0, :]))
            xi[k] = val
        for k in range(0, np.shape(self.y)[0]):
            val = (self.y[k, -1] - np.nanmin(self.y[:, -1])) / (np.nanmax(self.y[:, -1]) - np.nanmin(self.y[:, -1]))
            eta[k] = val

        if self.type == 'slit':  # special treatment of cells over and below slit
            idx_i1 = np.where(self.x[1:-1, 1:-1] != 0)[0][0]
            idx_i2 = np.where(self.x[1:-1, 1:-1] != 0)[0][-1]
            eta_slit = np.zeros(np.shape(self.x)[0])
            # # differentiate between nacelle and fuselage
            # eta_slit[:idx_i1+1] = np.array([1-(self.y[0,0]-self.y[k,0])/(self.y[0,0]-self.y[idx_i1+1,0]) if
            #                                 (len(self.x_geom) == 3) else (-1/(idx_i1+1)*k+1) if (len(self.x_geom) == 2)
            # else 0 for k in range(0,idx_i1+1)])
            # eta_slit[idx_i1+2:np.shape(self.x)[0]] = \
            #     np.array([1-(self.y[idx_i1+2,0]-self.y[k,0])/(self.y[idx_i1+1,0]-self.y[-1,0]) if (len(self.x_geom) == 3)
            #               else (-1/(np.shape(self.x)[0]-2-idx_i2)*k+(idx_i2+1)/(np.shape(self.x)[0]-2-idx_i2)+1)
            #     if (len(self.x_geom) == 2) else 0 for k in range(idx_i1+2,np.shape(self.x)[0])])
            if len(self.x_geom) == 3:  # PFC
                for k in range(0, idx_i1 + 1):
                    eta_slit[k] = 1 - (self.y[0, 0] - self.y[k, 0]) / (self.y[0, 0] - self.y[idx_i1 + 1, 0])
                for k in range(idx_i1 + 2, np.shape(self.x)[0]):
                    eta_slit[k] = 1 - (self.y[idx_i1 + 2, 0] - self.y[k, 0]) / (self.y[idx_i1 + 1, 0] - self.y[-1, 0])
            elif len(self.x_geom) == 2:  # nacelle
                for k in range(0, idx_i1 + 1):
                    eta_slit[k] = -1 / (idx_i1 + 1) * k + 1
                for k in range(idx_i1 + 2, np.shape(self.x)[0]):
                    eta_slit[k] = -1 / (np.shape(self.x)[0] - 2 - idx_i2) * k + (idx_i2 + 1) / (
                                np.shape(self.x)[0] - 2 - idx_i2) + 1

        for j in range(0, np.shape(self.x)[1] - 1, 1):
            for i in range(0, np.shape(self.x)[0] - 1, 1):
                if self.type == 'rect':
                    idx = -1
                elif self.type == 'slab':
                    if np.isnan(self.x[-1, j - 1]):
                        idx = np.where(np.isnan(self.x[:, j - 1]))[0][0] - 1
                    elif np.isnan(self.x[-1, j + 1]):
                        idx = np.where(np.isnan(self.x[:, j + 1]))[0][0] - 1
                    else:
                        try:
                            idx = np.where(np.isnan(self.x[:, j]))[0][0] - 1
                        except:
                            idx = -1
                else:
                    idx = -1

                # if loop required for special handling of slab and slit grid
                if np.isnan(self.x[i, j]) and self.type == 'slab':
                    x[i, j] = np.nan
                    y[i, j] = np.nan
                # grid coordinates above slit
                elif self.type == 'slit' and np.count_nonzero(self.x[:, j]) == 4 and i < np.where(self.x[:, j] != 0)[0][
                    1]:
                    idx = np.where(self.x[:, j] != 0)[0][1]
                    x[i, j] = (1 - eta_slit[i]) * self.x[idx, j] + eta_slit[i] * self.x[0, j] + (1 - xi[j]) * self.x[
                        i, 0] + xi[
                                  j] * self.x[i, -1] - (
                                      xi[j] * eta_slit[i] * self.x[0, -1] + xi[j] * (1 - eta_slit[i]) * self.x[
                                  idx, -1] + eta_slit[
                                          i] * (1 - xi[j]) * self.x[0, 0] + (1 - xi[j]) * (1 - eta_slit[i]) * self.x[
                                          idx, 0])
                    if len(self.x_geom) == 3:  # PFC
                        y[i, j] = (1 - eta_slit[i]) * self.y[idx, j] + eta_slit[i] * self.y[0, j]
                    elif len(self.x_geom) == 2:  # nacelle
                        y[i, j] = (1 - eta_slit[i]) * self.y[idx, j] + eta_slit[i] * self.y[0, j] + (1 - xi[j]) * \
                                  self.y[i, 0] + xi[
                                      j] * self.y[i, -1] - (
                                          xi[j] * eta_slit[i] * self.y[0, -1] + xi[j] * (1 - eta_slit[i]) * self.y[
                                      idx, -1] + eta_slit[
                                              i] * (1 - xi[j]) * self.y[0, 0] + (1 - xi[j]) * (1 - eta_slit[i]) *
                                          self.y[idx, 0])

                # grid coordinates below slit
                elif self.type == 'slit' and np.count_nonzero(self.x[:, j]) == 4 and i > np.where(self.x[:, j] != 0)[0][
                    2]:
                    idx1 = np.shape(self.x)[0] - 1
                    idx = np.where(self.x[:, j] != 0)[0][2]
                    x[i, j] = (1 - eta_slit[i]) * self.x[idx1, j] + eta_slit[i] * self.x[idx, j] + (1 - xi[j]) * self.x[
                        i, 0] + xi[
                                  j] * self.x[i, -1] - (
                                      xi[j] * eta_slit[i] * self.x[idx, -1] + xi[j] * (1 - eta_slit[i]) * self.x[
                                  idx1, -1] + eta_slit[
                                          i] * (1 - xi[j]) * self.x[idx, 0] + (1 - xi[j]) * (1 - eta_slit[i]) * self.x[
                                          idx1, 0])
                    # differentiate between nacelle and PFC configuration
                    if np.all(y[-1, :] == y[-1, 0]):
                        y[i, j] = (1 - eta_slit[i]) * self.y[idx1, j] + eta_slit[i] * self.y[idx, j] + (1 - xi[j]) * \
                                  self.y[i, 0] + xi[
                                      j] * self.y[i, -1] - (
                                          xi[j] * eta_slit[i] * self.y[idx, -1] + xi[j] * (1 - eta_slit[i]) *
                                          self.y[idx1, -1] + eta_slit[
                                              i] * (1 - xi[j]) * self.y[idx, 0] + (1 - xi[j]) * (1 - eta_slit[i]) *
                                          self.y[idx1, 0])
                    else:
                        y[i, j] = (1 - eta_slit[i]) * self.y[idx1, j] + eta_slit[i] * self.y[idx, j]

                elif self.x[i, j] != 0.0:
                    x[i, j] = self.x[i, j]
                    y[i, j] = self.y[i, j]
                else:
                    x[i, j] = (1 - eta[i]) * self.x[idx, j] + eta[i] * self.x[0, j] + (1 - xi[j]) * self.x[i, 0] + xi[
                        j] * self.x[i, -1] - (
                                          xi[j] * eta[i] * self.x[0, -1] + xi[j] * (1 - eta[i]) * self.x[idx, -1] + eta[
                                      i] * (1 - xi[j]) * self.x[0, 0] + (1 - xi[j]) * (1 - eta[i]) * self.x[idx, 0])
                    y[i, j] = (1 - eta[i]) * self.y[idx, j] + eta[i] * self.y[0, j] + (1 - xi[j]) * self.y[i, 0] + xi[
                        j] * self.y[i, -1] - (
                                          xi[j] * eta[i] * self.y[0, -1] + xi[j] * (1 - eta[i]) * self.y[idx, -1] + eta[
                                      i] * (1 - xi[j]) * self.y[0, 0] + (1 - xi[j]) * (1 - eta[i]) * self.y[idx, 0])

        if self.type == 'rect' and np.any(y[:, 0] != y[:, 0]):
            y[:, 0] = 0

        return x, y
