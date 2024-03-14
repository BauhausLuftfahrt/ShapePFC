"""Returns matrices A and B, with the velocity perturbations each panel cause on each other assuming a unitary vortex
strength.

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, Nikolaus Romanow, A. Habermann
"""

import numpy as np
import scipy.integrate as integ
from scipy.special import ellipk
from scipy.special import ellipe


def findVortices(Xm, Ym, Xn, Yn, phi, S, i_pan, j_sing):
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
        j_sing          [-]     Indices of panels where vortex singularities are attached to

    Returns:
        A       [-]    2-D array Aij = Normal perturbation at the ith element due to the vortex on the jth element
        B       [-]    2-D array Bij = Tangential perturbation at the ith element due to the vortex on the jth element

    Sources:
        [-] Göde, E. (1978). Ein Verfahren zur Berechnung des Strömungsfeldes um Strahltriebwerke
        [-] Göde, E., & Haberland, C. (1979). Berechnung des Strömungsfeldes um Triebwerk-Flügel-Konfigurationen
        [-] Trulin, D. J. (1968). A method for computing the pressure distribution about an axisymmetric nacelle in subsonic flow
    """
    # Number of panels
    nseg = len(Xm)  # Number of panels

    # Matrices of influence coefficient
    A = np.zeros([nseg, nseg])  # Normal perturbation at the ith element due to the vortex on the jth element
    B = np.zeros([nseg, nseg])  # Tangential perturbation at the ith element due to the vortex on the jth element

    # Compute vortex matrices
    for i in i_pan:
        for j in j_sing:
            # smallest distance from segment j to control point i
            r_min = min(((Xm[i] - Xn[j]) ** 2 + (Ym[i] - Yn[j]) ** 2) ** 0.5, (
                    (Xm[i] - Xn[j + 1]) ** 2 + (Ym[i] - Yn[j + 1]) ** 2) ** 0.5)
            # number of sub-segments for integration based on r_min
            n_s = max(int(16 * S[j] / r_min), 2)
            if (j != i):
                s = np.arange(-S[j] / 2, S[j] / 2 + ((S[j] / 2) - (-S[j] / 2)) / (n_s - 1),
                              ((S[j] / 2) - (-S[j] / 2)) / (n_s - 1))  # discretize panel into sub-segments
                a = Ym[j] + s * np.sin(phi[j])  # parametrized y-coordinate
                b = Xm[j] + s * np.cos(phi[j])  # parametrized x-coordinate
                m = m_ell(a, b, Xm[i], Ym[i])  # find kernell for elliptic integrals
                K = ellipk(m)  # Elliptic Integral of first kind
                E = ellipe(m)  # Elliptic Integral of second kind
                # used Simpson integration before, but it was too slow. Change to integ.simps in case trapezoid produces odd/wrong results
                Vx = integ.trapezoid(vx(a, b, Xm[i], Ym[i], E, K),
                                     s)  # Integrate over segment to get perturbation in Vx
                Vy = integ.trapezoid(vy(a, b, Xm[i], Ym[i], E, K),
                                     s)  # Integrate over segment to get perturbation in Vy
                A[i, j] = -np.sin(phi[i]) * Vx + np.cos(phi[i]) * Vy
                B[i, j] = np.cos(phi[i]) * Vx + np.sin(phi[i]) * Vy
            else:  # Special treatment for influence of each panel on itself
                if abs(0.08 * Ym[j]) < S[j] / 2:
                    d = 0.08 * Ym[j]  # d = distance from mid-point where numerical integration cannot be done
                    r_min = abs(d)  # distance of sub-element at the end of segment to midpoint was defined as d
                    n_s = max(int((16 * (S[j] - abs(2 * d)) / 2 / r_min)), 2)
                    sl = np.arange(-S[j] / 2, -abs(d) + (-abs(d) - (-S[j] / 2)) / (n_s - 1),
                                   (-abs(d) - (-S[j] / 2)) / (n_s - 1))
                    sr = np.arange(abs(d), S[j] / 2 + (S[j] / 2 - abs(d)) / (n_s - 1), (S[j] / 2 - abs(d)) / (n_s - 1))
                    al = Ym[j] + sl * np.sin(phi[j])
                    bl = Xm[j] + sl * np.cos(phi[j])
                    ml = m_ell(al, bl, Xm[i], Ym[i])
                    Kl = ellipk(ml)
                    El = ellipe(ml)
                    Vxl = integ.trapezoid(vx(al, bl, Xm[i], Ym[i], El, Kl), sl)
                    Vyl = integ.trapezoid(vy(al, bl, Xm[i], Ym[i], El, Kl), sl)
                    ar = Ym[j] + sr * np.sin(phi[j])
                    br = Xm[j] + sr * np.cos(phi[j])
                    mr = m_ell(ar, br, Xm[i], Ym[i])
                    Kr = ellipk(mr)
                    Er = ellipe(mr)
                    Vxr = integ.trapezoid(vx(ar, br, Xm[i], Ym[i], Er, Kr), sr)
                    Vyr = integ.trapezoid(vy(ar, br, Xm[i], Ym[i], Er, Kr), sr)
                else:
                    d = S[j] / 2
                    Vxl = 0
                    Vyl = 0
                    Vxr = 0
                    Vyr = 0
                # integral from -d to d evaluated via expansion. Version of Vxm and Vym by Göde (197), Equ. A.18
                # and A.19 corrected acc. to Trulin (1968), Equ. 32 and 33, with gamma_1=gamma_2=gamma; S_1=S_2=d/Y_m[j]
                Vxm = 2 * (d / Ym[j]) * (
                            np.sin(phi[j]) ** 2 + np.log(8 / (d / Ym[j])) + (1 / 144) * ((d / Ym[j]) ** 2) *
                            (9 - 23 * (np.sin(phi[j]) ** 2) + 6 * (np.sin(phi[j]) ** 4)
                             + 3 * np.log(8 / (d / Ym[j])) * (-3 + 4 * (np.sin(phi[j]) ** 2))))
                Vym = (d / Ym[j]) * np.sin(2 * phi[j]) * (1 + (1 / 48) * ((d / Ym[j]) ** 2) * (
                            2 * (np.sin(phi[j]) ** 2) - 9 + 6 * np.log(8 / (d / Ym[j]))))
                Vx = 2 * np.pi * np.cos(phi[j]) + Vxl + Vxm + Vxr  # internal flow
                Vy = 2 * np.pi * np.sin(phi[j]) + Vyl + Vym + Vyr  # internal flow
                A[i, j] = -1 * np.sin(phi[i]) * Vx + np.cos(phi[i]) * Vy
                B[i, j] = np.cos(phi[i]) * Vx + np.sin(phi[i]) * Vy

    A[A == np.isnan] = 0
    B[B == np.isnan] = 0

    return A, B


def findVelocitiesVortex(Xs, r_0, Xm, Ym, Xn, Yn, phi, S, i_pan, j_sing):
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
        j_sing          [-]     Indices of panels where vortex singularities are attached to

    Returns:
        Wx       [-]   2-D array Wx = X perturbation at the ith point of interest due to the vortex on the jth element
        Wy      [-]    2-D array Wy = Y perturbation at the ith point of interest due to the vortex on the jth element
                        (rectangular coordinates)

    Sources:
        [-] Göde, E. (1978). Ein Verfahren zur Berechnung des Strömungsfeldes um Strahltriebwerke
        [-] Göde, E., & Haberland, C. (1979). Berechnung des Strömungsfeldes um Triebwerk-Flügel-Konfigurationen
        [-] Trulin, D. J. (1968). A method for computing the pressure distribution about an axisymmetric nacelle in subsonic flow
    """
    # Number of panels
    nseg = len(Xm)

    # Initialize arrays
    Wx = np.zeros([nseg, nseg])
    Wy = np.zeros([nseg, nseg])
    eps = 1e-12

    for i in i_pan:
        for j in j_sing:
            r_min = min(((Xs[i] - Xn[j]) ** 2 + (r_0[i] - Yn[j]) ** 2) ** 0.5, (
                    (Xs[i] - Xn[j + 1]) ** 2 + (r_0[i] - Yn[j + 1]) ** 2) ** 0.5)
            n_s = max(int(16 * S[j] / r_min), 2)
            if abs(Xs[i] - Xm[j]) > eps or abs(r_0[i] - Ym[j]) > eps:
                s = np.arange(-S[j] / 2, S[j] / 2 + ((S[j] / 2) - (-S[j] / 2)) / (n_s - 1),
                              ((S[j] / 2) - (-S[j] / 2)) / (n_s - 1))
                a = Ym[j] + s * np.sin(phi[j])
                b = Xm[j] + s * np.cos(phi[j])
                m = m_ell(a, b, Xs[i], r_0[i])
                K = ellipk(m)
                E = ellipe(m)
                Vx = integ.trapezoid(vx(a, b, Xs[i], r_0[i], E, K), s)
                Vy = integ.trapezoid(vy(a, b, Xs[i], r_0[i], E, K), s)
                Wx[i, j] = Vx
                Wy[i, j] = Vy
            else:  # effect of element at own midpoint -> different procedure
                if abs(0.08 * Ym[j]) < S[j] / 2:
                    d = 0.08 * Ym[j]
                    r_min = abs(d)
                    n_s = max(int((16 * (S[j] - abs(2 * d)) / 2 / r_min)), 2)
                    sl = np.arange(-S[j] / 2, -abs(d) + (-abs(d) - (-S[j] / 2)) / (n_s - 1),
                                   (-abs(d) - (-S[j] / 2)) / (n_s - 1))
                    sr = np.arange(abs(d), S[j] / 2 + (S[j] / 2 - abs(d)) / (n_s - 1), (S[j] / 2 - abs(d)) / (n_s - 1))
                    al = Ym[j] + sl * np.sin(phi[j])
                    bl = Xm[j] + sl * np.cos(phi[j])
                    ml = m_ell(al, bl, Xs[i], r_0[i])
                    Kl = ellipk(ml)
                    El = ellipe(ml)
                    Vxl = integ.trapezoid(vx(al, bl, Xs[i], r_0[i], El, Kl), sl)
                    Vyl = integ.trapezoid(vy(al, bl, Xs[i], r_0[i], El, Kl), sl)
                    ar = Ym[j] + sr * np.sin(phi[j])
                    br = Xm[j] + sr * np.cos(phi[j])
                    mr = m_ell(ar, br, Xs[i], r_0[i])
                    Kr = ellipk(mr)
                    Er = ellipe(mr)
                    Vxr = integ.trapezoid(vx(ar, br, Xs[i], r_0[i], Er, Kr), sr)
                    Vyr = integ.trapezoid(vy(ar, br, Xs[i], r_0[i], Er, Kr), sr)
                else:
                    d = S[j] / 2
                    Vxl = 0
                    Vyl = 0
                    Vxr = 0
                    Vyr = 0
                # integral from -d to d evaluated via expansion
                Vxm = 2 * (d / Ym[j]) * (
                            np.sin(phi[j]) ** 2 + np.log(8 / (d / Ym[j])) + (1 / 144) * ((d / Ym[j]) ** 2) *
                            (9 - 23 * (np.sin(phi[j]) ** 2) + 6 * (np.sin(phi[j]) ** 4)
                             + 3 * np.log(8 / (d / Ym[j])) * (-3 + 4 * (np.sin(phi[j]) ** 2))))
                Vym = (d / Ym[j]) * np.sin(2 * phi[j]) * (1 + (1 / 48) * ((d / Ym[j]) ** 2) * (
                            2 * (np.sin(phi[j]) ** 2) - 9 + 6 * np.log(8 / (d / Ym[j]))))
                Vx = -2 * np.pi * np.cos(phi[j]) + Vxl + Vxm + Vxr  # external flow
                Vy = -2 * np.pi * np.sin(phi[j]) + Vyl + Vym + Vyr  # external flow
                Wx[i, j] = Vx
                Wy[i, j] = Vy

    Wx[Wx == np.isnan] = 0
    Wy[Wy == np.isnan] = 0

    return Wx, Wy


def findStreamlineVelocitiesVortex(Xg, Yg, Xm, Ym, Xn, Yn, phi, S, j_sing):
    """Computes the effect each panel has on a specific point of the domain

    Author:  A. Habermann

     Args:
        Xg              [m]     1-D array X-coordinate of point of interest
        Yg              [m]     1-D array Y-coordinate of point of interest
        Xm              [m]     X-coordinate of panel mid-points
        Ym              [m]     Y-coordinate of panel mid-points (local transverse radius)
        Xn              [m]     1-D array X-coordinate of geometric profile  (panel end points)
        Yn              [m]     1-D array Y-coordinate of geometric profile  (panel end points)
        S               [m]     1-D array Segment length
        phi             [rad]   1-D array Segment angle w.r.t symmetry axis
        j_sing          [-]     Indices of panels where vortex singularities are attached to

    Returns:
        Wx       [-]   2-D array Wx = X perturbation at the ith point of interest due to the vortex on the jth element
        Wy      [-]    2-D array Wy = Y perturbation at the ith point of interest due to the vortex on the jth element
                        (rectangular coordinates)

    Sources:
        [-] Göde, E. (1978). Ein Verfahren zur Berechnung des Strömungsfeldes um Strahltriebwerke
        [-] Göde, E., & Haberland, C. (1979). Berechnung des Strömungsfeldes um Triebwerk-Flügel-Konfigurationen
        [-] Trulin, D. J. (1968). A method for computing the pressure distribution about an axisymmetric nacelle in subsonic flow
    """
    # Number of panels
    nseg = len(j_sing)
    # Number of panels
    Wx = np.zeros([1, nseg])
    Wy = np.zeros([1, nseg])
    eps = 1e-10

    if isinstance(Xg, float):
        Xg = [float(Xg)]
    if isinstance(Yg, float):
        Yg = [float(Yg)]

    i = 0

    for j in j_sing:
        r_min = min(((Xg[i] - Xn[j]) ** 2 + (Yg[i] - Yn[j]) ** 2) ** 0.5, (
                (Xg[i] - Xn[j + 1]) ** 2 + (Yg[i] - Yn[j + 1]) ** 2) ** 0.5)
        n_s = min(max(int(16 * S[j] / r_min), 2), 1000)
        if ((Xg[i] - Xm[j]) ** 2 + (Yg[i] - Ym[j]) ** 2) ** 0.5 > eps:
            s = np.arange(-S[j] / 2, S[j] / 2 + ((S[j] / 2) - (-S[j] / 2)) / (n_s - 1),
                          ((S[j] / 2) - (-S[j] / 2)) / (n_s - 1))
            a = Ym[j] + s * np.sin(phi[j])
            b = Xm[j] + s * np.cos(phi[j])
            m = m_ell(a, b, Xg[i], Yg[i])
            K = ellipk(m)
            E = ellipe(m)
            Vx = integ.trapezoid(vx(a, b, Xg[i], Yg[i], E, K), s)
            Vy = integ.trapezoid(vy(a, b, Xg[i], Yg[i], E, K), s)
            Wx[i, j - j_sing[0]] = Vx
            Wy[i, j - j_sing[0]] = Vy
        else:  # effect of element at own midpoint -> different procedure
            if abs(0.08 * Ym[j]) < S[j] / 2:
                d = 0.08 * Ym[j]
                r_min = abs(d)
                n_s = max(int((16 * (S[j] - abs(2 * d)) / 2 / r_min)), 2)
                sl = np.arange(-S[j] / 2, -abs(d) + (-abs(d) - (-S[j] / 2)) / (n_s - 1),
                               (-abs(d) - (-S[j] / 2)) / (n_s - 1))
                sr = np.arange(abs(d), S[j] / 2 + (S[j] / 2 - abs(d)) / (n_s - 1), (S[j] / 2 - abs(d)) / (n_s - 1))
                al = Ym[j] + sl * np.sin(phi[j])
                bl = Xm[j] + sl * np.cos(phi[j])
                ml = m_ell(al, bl, Xg[i], Yg[i])
                Kl = ellipk(ml)
                El = ellipe(ml)
                Vxl = integ.trapezoid(vx(al, bl, Xg[i], Yg[i], El, Kl), sl)
                Vyl = integ.trapezoid(vy(al, bl, Xg[i], Yg[i], El, Kl), sl)
                ar = Ym[j] + sr * np.sin(phi[j])
                br = Xm[j] + sr * np.cos(phi[j])
                mr = m_ell(ar, br, Xg[i], Yg[i])
                Kr = ellipk(mr)
                Er = ellipe(mr)
                Vxr = integ.trapezoid(vx(ar, br, Xg[i], Yg[i], Er, Kr), sr)
                Vyr = integ.trapezoid(vy(ar, br, Xg[i], Yg[i], Er, Kr), sr)
            else:
                d = S[j] / 2
                Vxl = 0
                Vyl = 0
                Vxr = 0
                Vyr = 0
            # integral from -d to d evaluated via expansion
            Vxm = 2 * (d / Ym[j]) * (np.sin(phi[j]) ** 2 + np.log(8 / (d / Ym[j])) + (1 / 144) * ((d / Ym[j]) ** 2) *
                                     (9 - 23 * (np.sin(phi[j]) ** 2) + 6 * (np.sin(phi[j]) ** 4)
                                      + 3 * np.log(8 / (d / Ym[j])) * (-3 + 4 * (np.sin(phi[j]) ** 2))))
            Vym = (d / Ym[j]) * np.sin(2 * phi[j]) * (1 + (1 / 48) * ((d / Ym[j]) ** 2) * (
                        2 * (np.sin(phi[j]) ** 2) - 9 + 6 * np.log(8 / (d / Ym[j]))))
            Vx = -2 * np.pi * np.cos(phi[j]) + Vxl + Vxm + Vxr  # external flow
            Vy = -2 * np.pi * np.sin(phi[j]) + Vyl + Vym + Vyr  # external flow
            Wx[i, j - j_sing[0]] = Vx
            Wy[i, j - j_sing[0]] = Vy

            if (np.isnan(Wx[i, j])):
                Wx[i, j - j_sing[0]] = 0
            if (np.isnan(Wy[i, j])):
                Wy[i, j - j_sing[0]] = 0

    return Wx, Wy


# Expression for the kernell of the elliptic integrals
def m_ell(a, b, x, y):  # Kernell k**2 = m of the complete elliptic integral
    return (4 * a * y) / ((y + a) ** 2 + (x - b) ** 2)


# Equations to be integrated over each segment for the influence of the panel
def vx(a, b, x, y, E, K):  # Velocity vx to be integrated on each segment
    r1 = ((x - b) ** 2 + (y - a) ** 2) ** 0.5
    r2 = ((x - b) ** 2 + (y + a) ** 2) ** 0.5
    return 2 / r2 * (K - E * (1 + (2 * a * (y - a) / r1 ** 2)))


def vy(a, b, x, y, E, K):  # Velocity vy to be integrated on each segment
    r1 = ((x - b) ** 2 + (y - a) ** 2) ** 0.5
    r2 = ((x - b) ** 2 + (y + a) ** 2) ** 0.5
    return -2 * (x - b) / (r2 * y) * (K - E * (1 + (
                2 * a * y / r1 ** 2)))  # This equation is wrongly stated in Göde 1978, Equ. 16. Correct equ. can be found in Trulin (1968), Equ. (4)
