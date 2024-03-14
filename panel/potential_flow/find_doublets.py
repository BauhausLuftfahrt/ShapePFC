"""Returns matrices A and B, with the velocity perturbations each panel cause on each other assuming a unitary
doublet strength.

Author:  Nikolaus Romanow, A. Habermann
"""

import numpy as np
import scipy.integrate as integ
from scipy.special import ellipk
from scipy.special import ellipe


def findDoublets(Xm, Ym, Xn, Yn, phi, S, i_pan, j_sing):
    """Computes the effect each panel has on each other (including himself)

    Author:  Nikolaus Romanow

     Args:
        Xn              [m]     1-D array X-coordinate of geometric profile (segment extremities)
        Yn              [m]     1-D array Y-coordinate of geometric profile (segment extremities)
        Xm              [m]     1-D array X-coordinate of segment's mid-point
        Ym              [m]     1-D array Y-coordinate of segment's mid-point
        S               [m]     1-D array Segment sizes
        phi             [rad]   1-D array Segment angle w.r.t symmetry axis
        i_pan           [-]     Indices of panels where velocities are induced
        j_sing          [-]     Indices of panels where doublet singularities are attached to

    Returns:
        A       [-]    2-D array Aij = Normal perturbation at the ith element due to the doublet on the jth element
        B       [-]    2-D array Bij = Tangential perturbation at the ith element due to the doublet on the jth element

    Sources:
        [-] Göde, E., & Haberland, C. (1979). Berechnung des Strömungsfeldes um Triebwerk-Flügel-Konfigurationen
        [-] Katz, J., & Plotkin, A. (2001). Low-Speed Aerodynamics (Second edition)
    """
    # Number of panels
    nseg = len(Xm)  # Number of panels

    # Matrices of influence coefficient
    A = np.zeros([nseg, nseg])  # Normal perturbation at the ith element due to the doublet on the jth element
    B = np.zeros([nseg, nseg])  # Tangential perturbation at the ith element due to the doublet on the jth element

    # Compute doublet matrices
    for i in i_pan:
        for j in j_sing:  # due to doublet distribution on panel j (vortices at b with radius a)
            # al = Ym[j] - (S[j]/2) * np.sin(phi[j])      # y-coordinate left vortex
            al = Yn[j] * 1
            # bl = Xm[j] - (S[j]/2) * np.cos(phi[j])      # x-coordinate left vortex
            bl = Xn[j] * 1
            # ar = Ym[j] + (S[j] / 2) * np.sin(phi[j])
            ar = Yn[j + 1] * 1
            # br = Xm[j] + (S[j] / 2) * np.cos(phi[j])
            br = Xn[j + 1] * 1
            ml = m_ell(al, bl, Xm[i], Ym[i])  # find kernell for elliptic integrals
            mr = m_ell(ar, br, Xm[i], Ym[i])
            Kl = ellipk(ml)  # Elliptic Integral of first kind
            El = ellipe(ml)  # Elliptic Integral of second kind
            Kr = ellipk(mr)
            Er = ellipe(mr)
            Vxl = vx(al, bl, Xm[i], Ym[i], El, Kl)
            Vyl = vy(al, bl, Xm[i], Ym[i], El, Kl)
            Vxr = vx(ar, br, Xm[i], Ym[i], Er, Kr)
            Vyr = vy(ar, br, Xm[i], Ym[i], Er, Kr)
            Vx = -(Vxr - Vxl) * S[j]  # opposite signs for left and right vortex
            Vy = -(Vyr - Vyl) * S[j]  # opposite signs for left and right vortex
            A[i, j] = -np.sin(phi[i]) * Vx + np.cos(phi[i]) * Vy
            B[i, j] = np.cos(phi[i]) * Vx + np.sin(phi[i]) * Vy

    A[A == np.isnan] = 0
    B[B == np.isnan] = 0

    return A, B


def findVelocitiesDoublet(Xs, r_0, Xm, Ym, Xn, Yn, phi, S, i_pan, j_sing):
    """Computes the effect each panel has on a specific point of the domain

    Author:  Nikolaus Romanow

     Args:
        Xs              [m]     1-D array X-coordinate of segment's mid-point
        r_0             [m]     1-D array Y-coordinate of segment's mid-point (local transverse radius)
        Xm              [m]     X-coordinate of point of interest
        Ym              [m]     Y-coordinate of point of interest
        Xn              [m]     1-D array X-coordinate of geometric profile  (segment extremities)
        Yn              [m]     1-D array Y-coordinate of geometric profile  (segment extremities)
        S               [m]     1-D array Segment sizes
        phi             [rad]   1-D array Segment angle w.r.t symmetry axis
        i_pan           [-]     Indices of panels where velocities are induced
        j_sing          [-]     Indices of panels where doublet singularities are attached to

    Returns:
        Wx       [-]   2-D array Wx = X perturbation at the ith point of interest due to the doublet on the jth element
        Wy      [-]    2-D array Wy = Y perturbation at the ith point of interest due to the doublet on the jth element
                        (rectangular coordinates)

    Sources:
        [-] Göde, E., & Haberland, C. (1979). Berechnung des Strömungsfeldes um Triebwerk-Flügel-Konfigurationen
        [-] Katz, J., & Plotkin, A. (2001). Low-Speed Aerodynamics (Second edition)
    """
    # Number of panels
    nseg = len(Xm)

    # Initialize arrays
    Wx = np.zeros([nseg, nseg])
    Wy = np.zeros([nseg, nseg])
    # eps = 1e-12

    for i in i_pan:
        for j in j_sing:
            # al = Ym[j] - (S[j] / 2) * np.sin(phi[j])    # y-coordinate left vortex
            al = Yn[j] * 1
            # bl = Xm[j] - (S[j] / 2) * np.cos(phi[j])    # x-coordinate left vortex
            bl = Xn[j] * 1
            # ar = Ym[j] + (S[j] / 2) * np.sin(phi[j])
            ar = Yn[j + 1] * 1
            # br = Xm[j] + (S[j] / 2) * np.cos(phi[j])
            br = Xn[j + 1] * 1
            ml = m_ell(al, bl, Xs[i], r_0[i])  # find kernell for elliptic integrals
            mr = m_ell(ar, br, Xs[i], r_0[i])
            Kl = ellipk(ml)  # Elliptic Integral of first kind
            El = ellipe(ml)  # Elliptic Integral of second kind
            Kr = ellipk(mr)
            Er = ellipe(mr)
            Vxl = vx(al, bl, Xs[i], r_0[i], El, Kl)
            Vyl = vy(al, bl, Xs[i], r_0[i], El, Kl)
            Vxr = vx(ar, br, Xs[i], r_0[i], Er, Kr)
            Vyr = vy(ar, br, Xs[i], r_0[i], Er, Kr)
            Vx = -(Vxr - Vxl) * S[j]  # opposite signs for left and right vortex
            Vy = -(Vyr - Vyl) * S[j]  # opposite signs for left and right vortex
            Wx[i, j] = Vx
            Wy[i, j] = Vy

    Wx[Wx == np.isnan] = 0
    Wy[Wy == np.isnan] = 0

    return Wx, Wy


# Expression for the kernell of the elliptic integrals
def m_ell(a, b, x, y):  # Kernell k**2 = m of the complete elliptic integral
    return (4 * a * y) / ((y + a) ** 2 + (x - b) ** 2)


# Equations for left and right vortex
def vx(a, b, x, y, E, K):  # Velocity vx to be integrated on each segment
    r1 = ((x - b) ** 2 + (y - a) ** 2) ** 0.5
    r2 = ((x - b) ** 2 + (y + a) ** 2) ** 0.5
    return 2 / r2 * (K - E * (1 + (2 * a * (y - a) / r1 ** 2)))


def vy(a, b, x, y, E, K):  # Velocity vy to be integrated on each segment
    r1 = ((x - b) ** 2 + (y - a) ** 2) ** 0.5
    r2 = ((x - b) ** 2 + (y + a) ** 2) ** 0.5
    return -2 * (x - b) / (r2 * y) * (K - E * (1 + (
                2 * a * y / r1 ** 2)))  # This equation is wrongly stated in Göde 1978, Equ. 16. Correct equ. can be found in Trulin (1968), Equ. (4)
