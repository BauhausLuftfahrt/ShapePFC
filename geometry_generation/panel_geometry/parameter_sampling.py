"""Compute new sampling/discretization based on arc length and curvature (fuselage) or on cosine spacing (nacelle)

Author:  Nikolaus Romanow, A. Habermann

 Args:
    X               [m]         X-coordinate of initial nodes
    Y               [m]         Y-coordinate of initial nodes or InterpolatedUnivariateSpline object
    N               [-]         Number of nodes to distribute along body profile
    w               [-]         Weighting factor between arc-length and curvature based parametrisation
    type            [-]         Type of discretization

Returns:
    samples_new     [m]         X-coordinate of new nodes 
    arc_length      [m]         Cumulative arc length function of body contour

Sources:
    [-]     Hernández-Mederos, V., & Estrada-Sarlabous, J. (2003). Sampling points on regular parametric curves with control of their distribution
    [-]     Pagani, L., & Scott, P. J. (2018). Curvature based sampling of curves and surfaces
    [-]     Lu, L., & Zhao, S. (2019). High-quality point sampling for B-spline fitting of parametric curves with feature recognition
    
    [-]     Halsey, N. D., & Hess, J. L. (1978). A Geometry Package for Generation of Imput Data for a Three-dimensional Potential-flow Program
    [-]     Katz, J., & Plotkin, A. (2001). Low-Speed Aerodynamics
    [-]     Lewis, R. I. (2009). Vortex Element Methods for Fluid Dynamic Analysis of Engineering Systems
"""

import numpy as np
from scipy import interpolate, integrate


def paramSampling(X, Y, N, w, sample_type):
    # create y-coordinates if necessary
    if type(Y).__name__ == "InterpolatedUnivariateSpline":
        X_ip = X
        Y_ip = Y(X_ip)
    else:
        # Approximation of body contour
        N_ip = 1000  # number of points for approximation
        Fs = interpolate.UnivariateSpline(X, Y, s=0)
        X_ip = np.linspace(X[0], X[-1], N_ip)
        Y_ip = Fs(X_ip)

    if sample_type == 0:  # fuselage
        spacing = X_ip[1] - X_ip[0]
        # first and second derivative of Y_ip
        dYn = np.gradient(Y_ip, spacing, edge_order=2)
        d2Yn = np.gradient(dYn, spacing, edge_order=2)

        # Arc length parametrisation
        dr_abs = (1 + dYn ** 2) ** 0.5
        L_p = integrate.cumtrapz(dr_abs, X_ip)
        L_p = np.append(0, L_p)

        # Curvature parametrisation
        k_abs = abs(d2Yn) / ((1 + dYn ** 2) ** (3 / 2))
        K_p = integrate.cumtrapz(k_abs, X_ip)  # Curvature parametrisation according to Lu
        # K_p = integrate.cumtrapz(k_abs*dr_abs, X_ip)      # Curvature parametrisation according to Pagani
        # K_p = integrate.cumtrapz(k_abs**2, X_ip)          # Curvature (bending energy) parametrisation according to Hernandez-Mederos
        K_p = np.append(0, K_p)

        # Mixed parametrisation
        P = (1 - w) * (L_p / L_p[-1]) + w * (K_p / K_p[-1])

        # Inverting mixed parametrisation function and scale from [0,1] to [X_ip(start),X_ip(end)]
        P_inv = interpolate.UnivariateSpline(P * (X_ip[-1] - X_ip[0]) + X_ip[0], X_ip, s=0)

        # Get arc length function (for drag computation)
        arc_length = interpolate.UnivariateSpline(X_ip, L_p, s=0)

        samples_old = np.linspace(X[0], X[-1], N)  # equidistant sample points
        samples_new = P_inv(samples_old)  # parametrised sample points

        samples_new[0] = X[0]
        samples_new[-1] = X[-1]

    elif sample_type == 1:  # nacelle bottom
        # Approximation of body contour
        spacing = X_ip[1] - X_ip[0]
        # first derivative of Y_ip
        dYn = np.gradient(Y_ip, spacing, edge_order=2)

        # Arc length parametrisation
        dr_abs = (1 + dYn ** 2) ** 0.5
        L_p = integrate.cumtrapz(dr_abs, X_ip)
        L_p = np.append(0, L_p)

        # Get arc length function (for drag computation)
        arc_length = interpolate.UnivariateSpline(X_ip, L_p, s=0)

        # cosine parametrisation
        c = max(X) - min(X)
        beta = np.linspace(0, np.pi, N)
        samples_new = 0.5 * c * (1 - np.cos(beta))
        samples_new = X[0] + np.flip(samples_new)

    elif sample_type == 2:  # nacelle top
        spacing = X_ip[1] - X_ip[0]
        # first derivative of Y_ip
        dYn = np.gradient(Y_ip, spacing, edge_order=2)

        # Arc length parametrisation
        dr_abs = (1 + dYn ** 2) ** 0.5
        L_p = integrate.cumtrapz(dr_abs, X_ip)
        L_p = np.append(0, L_p)

        # Get arc length function (for drag computation)
        arc_length = interpolate.UnivariateSpline(X_ip, L_p, s=0)

        # cosine parametrisation
        c = max(X) - min(X)
        beta = np.linspace(0, np.pi, N)
        samples_new = 0.5 * c * (1 - np.cos(beta))
        samples_new = X[0] + samples_new

    return samples_new, arc_length


def translate_points(x_node, x_low_old, x_up_old, x_low_new, x_up_new):
    return (x_node - x_low_old) / (x_up_old - x_low_old) * (x_up_new - x_low_new) + x_low_new
