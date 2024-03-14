"""Compute jet boundary and entrainment velocity

Author:  Nikolaus Romanow

 Args:
    s               [m]         Jet coordinate in X-direction (starting at nacelle TE)
    d               [m]         Nozzle diameter
    d_eq            [m]         Equivalent nozzle diameter
    r_i             [m]         Fuselage body contour (aft cone)

Returns:
    Y_jet           [m]         Y-coordinates of jet boundary
    V_z             [N]         Dimensionless entrainment velocity (divided by nozzle exit velocity)
    
Sources:
    [-]     Liem, K. (1962). Strömungsvorgänge beim freien Hubstrahler
    [-]     Seibold, W. (1963). Untersuchungen über die von Hubstrahlen an Senkrechtstartern erzeugten Sekundärkräfte
    [-]     Snel, H. (1972). A method for the calculation of the flow field induced by a free jet
    
    (original width rules have been modifed to area rules to account for the fuselage aft cone)
"""

import numpy as np
from scipy import interpolate


def jetLiem_WidthRule(s, d, r_i):  # jet model of Liem (1962) - Width Rule (original method)
    sr_Liem = 4.6 * d  # potential core length
    b = d + 0.352 * s  # jet width
    Y_jet = b / 2 + r_i  # jet boundary
    V_z = np.zeros(len(s))  # entrainment velocity (dimensionless)
    counter = 0
    for i in range(0, len(s)):
        if s[i] >= sr_Liem:
            V_z[i] = 0.0686 / (1 + 0.352 * (s[i] / d))
            counter = counter + 1
            if counter == 1:
                i_sr = i
    V_z[0] = 0.0316  # (see Seibold S. 266)
    f = interpolate.UnivariateSpline(np.append(s[0], s[i_sr:-1]), np.append(V_z[0], V_z[i_sr:-1]), k=2, s=0, ext=0)
    V_z[1:i_sr] = f(s[1:i_sr])
    return Y_jet, V_z, sr_Liem


def jetLiem_AreaRule(s, d_eq, r_i):  # jet model of Liem (1962) - Area Rule (modified method)
    sr_Liem = 4.6 * d_eq  # potential core length
    A_Ae = (1 + 0.352 * (s / d_eq)) ** 2  # law of jet broadening
    Y_jet = np.sqrt((d_eq / 2) ** 2 * A_Ae + r_i ** 2)  # jet boundary
    Q_Q0 = np.zeros(len(s))  # Liem Fig. 6
    dQQ0_dsdeq = np.zeros(len(s))
    V_z = np.zeros(len(s))
    counter = 0
    for i in range(0, len(s)):
        if s[i] >= sr_Liem:
            Q_Q0[i] = 0.78 * (1 + 0.352 * (s[i] / d_eq))
            dQQ0_dsdeq[i] = 0.78 * 0.352
            counter = counter + 1
            if counter == 1:
                i_sr = i
    Q_Q0[0] = 1
    f_QQ0 = interpolate.UnivariateSpline(np.append(s[0], s[i_sr:-1]), np.append(Q_Q0[0], Q_Q0[i_sr:-1]), k=3, s=0,
                                         ext=0)
    Q_Q0[1:i_sr] = f_QQ0(s[1:i_sr])
    # dQQ0_dsdeq = np.diff(Q_Q0)/np.diff(s/d_eq)
    # dQQ0_dsdeq = np.append(dQQ0_dsdeq, dQQ0_dsdeq[-1])             # variant 1
    dQQ0_dsdeq[0] = 4 * 0.0316
    f_dQQ0dsdeq = interpolate.UnivariateSpline(np.append(s[0], s[i_sr:-1]),
                                               np.append(dQQ0_dsdeq[0], dQQ0_dsdeq[i_sr:-1]), k=2, s=0, ext=0)
    dQQ0_dsdeq[1:i_sr] = f_dQQ0dsdeq(s[1:i_sr])  # variant 2 (more accurate)
    V_z = (1 / (4 * np.sqrt(A_Ae))) * dQQ0_dsdeq  # entrainment velocity (dimensionless)
    return Y_jet, V_z, sr_Liem, Q_Q0


def jetSeibold_WidthRule(s, d, r_i):  # jet model of Seibold (1962) - Width Rule (original method)
    s0_Seibold = 3 * d  # (6*r0)
    sr_Seibold = 6 * d  # potential core length (12*r0)
    r_F = np.zeros(len(s))
    V_z = np.zeros(len(s))  # entrainment velocity (dimensionless)
    counter_0 = 0
    counter_r = 0
    for i in range(0, len(s)):
        if s[i] <= s0_Seibold:
            # r_F[i] =                           # profile by abramovic
            # V_z[i] =
            counter_0 = counter_0 + 1
            if counter_0 == 1:
                i_s0 = i
        if s[i] >= sr_Seibold:
            r_F[i] = 0.272 * s[i]
            V_z[i] = 0.0883 / (0.272 * (s[i] / (d / 2)))
            counter_r = counter_r + 1
            if counter_r == 1:
                i_sr = i
    f_rF = interpolate.UnivariateSpline(np.append(s[0:i_s0 + 1], s[i_sr:-1]), np.append(V_z[0:i_s0 + 1], V_z[i_sr:-1]),
                                        k=2, s=0, ext=0)
    f_Vz = interpolate.UnivariateSpline(np.append(s[0:i_s0 + 1], s[i_sr:-1]), np.append(V_z[0:i_s0 + 1], V_z[i_sr:-1]),
                                        k=2, s=0, ext=0)
    r_F[i_s0 + 1:i_sr] = f_rF(s[i_s0 + 1:i_sr])
    Y_jet = r_F + r_i
    V_z[i_s0 + 1:i_sr] = f_Vz(s[i_s0 + 1:i_sr])
    return Y_jet, V_z, sr_Seibold


def jetSeibold_AreaRule(s, d_eq, r_i):  # jet model of Seibold (1962) - Area Rule (modified method)
    sr_Seibold = 6 * d_eq  # potential core length (12*r0)
    A_Ae = np.zeros(len(s))  # law of jet broadening
    d12QFQ0_dyr0 = np.zeros(len(s))  # Seibold Fig. 25
    counter_0 = 0
    counter_r = 0
    for i in range(0, len(s)):
        if s[i] >= sr_Seibold:
            A_Ae[i] = ((0.272 * s[i]) / (d_eq / 2)) ** 2
            d12QFQ0_dyr0[i] = 0.0883
            counter_r = counter_r + 1
            if counter_r == 1:
                i_sr = i
    A_Ae[0] = 1
    f_AAe = interpolate.UnivariateSpline(np.append(s[0], s[i_sr:-1]), np.append(A_Ae[0], A_Ae[i_sr:-1]), k=3, s=0,
                                         ext=0)
    A_Ae[1:i_sr] = f_AAe(s[1:i_sr])
    d12QFQ0_dyr0[0] = 0.0316
    f_dQdy = interpolate.UnivariateSpline(np.append(s[0], s[i_sr:-1]),
                                          np.append(d12QFQ0_dyr0[0], d12QFQ0_dyr0[i_sr:-1]), k=2, s=0, ext=0)
    d12QFQ0_dyr0[1:i_sr] = f_dQdy(s[1:i_sr])
    Y_jet = np.sqrt((d_eq / 2) ** 2 * (A_Ae) + r_i ** 2)  # jet boundary
    V_z = (1 / (np.sqrt(A_Ae))) * d12QFQ0_dyr0  # entrainment velocity (dimensionless)
    return Y_jet, V_z, sr_Seibold


def jetSnel_AreaRule(s, d_eq, r_i):  # jet model of Snel (1972) - Area Rule (original paper)
    sr_Snel = 6.2 * d_eq  # potential core length
    F_1c = 0.196
    F_2c = 0.093
    # alpha = np.log(F_2c)/np.log(F_1c)                                                                                   # alpha = 1.4574721303835003...
    alpha = np.log(1 / F_2c) / (np.log(1 / F_2c) - np.log(
        1 + 0.128 * 6.2 + (1.176 / 3) * 10 ** (-3) * 6.2 ** 3 + (0.616 / 4) * 10 ** (
            -3) * 6.2 ** 4))  # alpha = 1.4604664028601442 (for continuous A/Ae at s = sr_Snel)
    A_Ae = np.zeros(len(s))  # law of jet broadening
    E = np.zeros(len(s))  # entrainment function
    for i in range(0, len(s)):
        if s[i] < sr_Snel:
            # original equation from paper -> does not match orifice and jump between core and fully developed region
            # A_Ae[i] = (0.128*(s[i]/d_eq) + (1.176/3) * 10**(-3) * (s[i]/d_eq)**3 + (0.616/4) * 10**(-3) * (s[i]/d_eq)**4)**(alpha/(alpha-1))
            # '1' added inside brackets -> realistic at orifice and no jump
            A_Ae[i] = (1 + 0.128 * (s[i] / d_eq) + (1.176 / 3) * 10 ** (-3) * (s[i] / d_eq) ** 3 + (0.616 / 4) * 10 ** (
                -3) * (s[i] / d_eq) ** 4) ** (alpha / (alpha - 1))

            E[i] = 0.128 + 1.176 * 10 ** (-3) * (s[i] / d_eq) ** 2 + 0.616 * 10 ** (-3) * (s[i] / d_eq) ** 3
        elif s[i] >= sr_Snel:
            A_Ae[i] = (F_2c / F_1c ** 2) * ((F_1c / F_2c) + 0.32 * ((s[i] / d_eq) - sr_Snel / d_eq)) ** 2
            E[i] = 0.32
    Y_jet = np.sqrt((d_eq / 2) ** 2 * (A_Ae) + r_i ** 2)  # jet boundary
    V_z = (1 / (4 * np.sqrt(A_Ae))) * E  # entrainment velocity (dimensionless)
    return Y_jet, V_z, sr_Snel
