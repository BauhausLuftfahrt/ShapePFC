"""Calculate the tubulent boundary layer characteristics acc. to Green.

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, A. Habermann
"""

import numpy as np
from scipy.optimize import fsolve


def greenCharacteristics(thetapl, Hpl, r_0, phi, U_e, nu, Xs, flags, M_e, r_f, mu, rho_e, C_e, Ip, Ik, kappa, phi_d):
    """Transforms planar boundary layer definitions into axisymmetric thick boundary layer definitions and compute
    equilibrium conditions

    Author:  Carlos E. Ribeiro Santa Cruz Mendoza

    Sources:
        [1] Nakayama, A.; Patel, V. C. & Landweber, L.: Flow interaction near the tail of
            a body of resolution: Part 2: Iterative solution for row within and exterior to
            boundary layer and wake. Journal of Fluids Engineering, Transactions of the
            ASME 98 (1976), 538-546.
        [2] Green,  J.  E.;  Weeks,  D.  J.  &  Brooman,  J.  W.  F.:   Prediction  of  turbulent
        boundary layers and wakes in compressible flow by a lag-entrainment method. ARC R&M 3791 (1973)
        [12] Patel, V. C.: On the equations of a thick axisymmetric turbulent boundary layer.
            Tech. rep., Iowa Institute of Hydraulic Research 1973.
        [13] Patel, V. C.: a Simple Integral Method for the Calculation of Thick Axisymmetric
            Turbulent Boundary Layers. Aeronaut. Quart. 25 (1974).
    """
    # alpha = Beta*thetapl/(2*r_0)
    alpha = 0.5 * np.cos(phi) * (((Hpl ** 2) * (Hpl + 1)) / ((Hpl - 1) * (Hpl + 3))) * (thetapl / r_0)
    Delta_star = (Hpl + alpha * (Hpl + 1)) * r_0 * thetapl
    Theta = (1 + 2 * alpha) * r_0 * thetapl
    H = Delta_star / Theta
    Hpl_i = (Hpl + 1) / (1 + r_f * (M_e ** 2) / 5) - 1  # H with upper bar in Green 1977
    Hpl_star = 3.15 + 1.72 / (Hpl_i - 1) - 0.01 * (Hpl_i - 1) ** 2  # H_1 in Green 1977
    delta = (Hpl_star + Hpl) * thetapl
    Q = U_e * ((r_0 + 0.5 * delta * np.cos(phi)) * delta - Delta_star)
    Re_theta = rho_e * thetapl * U_e / mu
    FR = 1 + 0.056 * M_e ** 2
    FC = (1 + 0.2 * M_e ** 2) ** 0.5
    c_f0 = (0.01013 / (np.log10(FR * Re_theta) - 1.02) - 0.00075) / FC
    H_o = 1 / (1 - 6.55 * (0.5 * c_f0 * (1 + 0.04 * M_e ** 2)) ** 0.5)
    c_f = (0.9 / ((Hpl_i / H_o) - 0.4) - 0.5) * c_f0
    n = 2 / (Hpl - 1)  # power-law parameter for velocity profile
    theta_eq = (1.25 / Hpl) * (c_f / 2 - (((Hpl_i - 1) / (6.432 * Hpl_i)) ** 2) / (1 + 0.04 * M_e ** 2))
    Ce_eq = (1 / ((r_0 + delta) * (1 + kappa * delta))) * (
                Hpl_star * (c_f * r_0 / 2 - r_0 * (Hpl + 1) * theta_eq + Ip + Ik) + \
                (thetapl / r_0) * ((Hpl_star + Hpl) ** 2) * (
                            np.cos(phi) * (c_f * r_0 / 2 - r_0 * (Hpl + 1.5) * theta_eq + Ip + Ik) + \
                            thetapl * np.sin(phi) * (0.5 * kappa * r_0 - np.cos(phi))))
    C_taueq = (0.024 * Ce_eq + 1.2 * Ce_eq ** 2 + 0.32 * c_f0) * (1 + 0.1 * M_e ** 2)
    C_tau = (0.024 * C_e + 1.2 * C_e ** 2 + 0.32 * c_f0) * (1 + 0.1 * M_e ** 2)
    F = (0.02 * C_e + C_e ** 2 + 0.8 * c_f0 / 3) / (0.01 + C_e)
    dHH = -((Hpl_i - 1) ** 2) / (1.72 + 0.02 * (Hpl_i - 1) ** 3)
    return H, Theta, Delta_star, delta, c_f, Q, n, dHH, Hpl_star, F, C_taueq, C_tau, theta_eq, Hpl_i


def greenPlanar(Hpl_i, phi, r_0, Theta, r_f, M_e, Xs, i):
    """Solves equation (f1) iteratively to obtain planar characteristics from axisymmetric

    Author:  Carlos E. Ribeiro Santa Cruz Mendoza

    Sources:
        [1] Nakayama, A.; Patel, V. C. & Landweber, L.: Flow interaction near the tail of
            a body of resolution: Part 2: Iterative solution for row within and exterior to
            boundary layer and wake. Journal of Fluids Engineering, Transactions of the
            ASME 98 (1976), 538-546.
        [2] Green,  J.  E.;  Weeks,  D.  J.  &  Brooman,  J.  W.  F.:   Prediction  of  turbulent
            boundary layers and wakes in compressible flow by a lag-entrainment method. ARC R&M 3791 (1973)
        [12] Patel, V. C.: On the equations of a thick axisymmetric turbulent boundary layer.
            Tech. rep., Iowa Institute of Hydraulic Research 1973.
        [13] Patel, V. C.: a Simple Integral Method for the Calculation of Thick Axisymmetric
            Turbulent Boundary Layers. Aeronaut. Quart. 25 (1974).
    """
    Hpl = (Hpl_i + 1) * (1 + r_f * M_e ** 2 / 5) - 1
    dataf1 = (phi, r_0, Theta, Hpl)
    [thetapl, infodict2, ier2, mesg2] = fsolve(f1, 1e-3, args=dataf1, xtol=1e-10, factor=0.1, maxfev=1000,
                                               full_output=True)
    return Hpl, thetapl


def f1(thetapl, *data):
    phi, r_0, Theta, Hpl = data
    B = np.cos(phi) * (((Hpl ** 2) * (Hpl + 1)) / ((Hpl - 1) * (Hpl + 3)))
    f1 = (B * (thetapl ** 2) + r_0 * thetapl - Theta)
    return f1
