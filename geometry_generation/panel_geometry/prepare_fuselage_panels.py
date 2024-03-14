"""Prepare panels for fuselage geometry.

Author:  A. Habermann

    N:              [-]     Number of nodes for profile discretization
    w:              [-]     Weighting factor between arc-length and curvature based parametrisation
"""

import numpy as np
from scipy import interpolate

from geometry_generation.panel_geometry.parameter_sampling import paramSampling


def sampleFuselageGeometry(X_fuse, Y_fuse, N, w):
    # Sample fuselage geometry
    Fs_f = interpolate.UnivariateSpline(X_fuse, Y_fuse, s=0)
    Xn, arc_length = paramSampling(X_fuse, Y_fuse, N, w, 0)
    Yn = Fs_f(Xn)
    Fm = np.zeros(len(Xn) - 1)

    return Xn, Yn, Fm, arc_length


def sampleFuselageGeometry_refnose(X_fuse, Y_fuse, N, w, X_nose):
    # Sample fuselage geometry
    idx_nose = X_fuse.index(X_nose)
    Fs_f1 = interpolate.UnivariateSpline(X_fuse[:idx_nose], Y_fuse[:idx_nose], s=0)
    Xn1, arc_length1 = paramSampling(X_fuse[:idx_nose], Y_fuse[:idx_nose], 50, w, 0)
    Yn1 = Fs_f1(Xn1)
    Fm1 = np.zeros(len(Xn1) - 1)
    Fs_f2 = interpolate.UnivariateSpline(X_fuse[idx_nose:], Y_fuse[idx_nose:], s=0)
    Xn2, arc_length2 = paramSampling(X_fuse[idx_nose:], Y_fuse[idx_nose:], 50, w, 0)
    Yn2 = Fs_f2(Xn2)
    Fm2 = np.zeros(len(Xn2) - 1)
    Xn = np.concatenate((Xn1, Xn2))
    Yn = np.concatenate((Yn1, Yn2))
    Fm = np.concatenate((Fm1, Fm2))
    arc_length_val1 = arc_length1(Xn1)
    arc_length_val2 = arc_length2(Xn2)
    arc_length_val_tot = np.concatenate((arc_length_val1, arc_length_val2))
    arc_length_fun = interpolate.UnivariateSpline(Xn, arc_length_val_tot, s=0)
    return Xn, Yn, Fm, arc_length_fun


def simplify_and_sample_fuselage_geometry(X_fuse, Y_fuse, N, w, int_loc, ff_inlet_loc):
    # Simplify fuselage geometry
    if ff_inlet_loc > int_loc:
        pass
    else:
        X_fuse_simple, Y_fuse_simple = X_fuse, Y_fuse

    # Sample fuselage geometry
    Xn, Yn, Fm, arc_length = sampleFuselageGeometry(X_fuse_simple, Y_fuse_simple, N, w)

    return Xn, Yn, Fm, arc_length
