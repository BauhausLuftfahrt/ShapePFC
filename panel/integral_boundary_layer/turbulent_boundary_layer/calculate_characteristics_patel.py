"""Calculate the tubulent boundary layer characteristics acc. to Patel.

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, A. Habermann
"""

import numpy as np
from scipy.optimize import fsolve


def frictionLaw(u, thetapl, nu, Hpl):
    """friction law of Thompson (1965) fitted by Patel """
    ReBar_theta = u * thetapl / nu
    c = np.log(ReBar_theta)
    a = 0.019521 - 0.386768 * c + 0.028345 * (c ** 2) - 0.000701 * (c ** 3)
    b = 0.191511 - 0.834891 * c + 0.062588 * (c ** 2) - 0.001953 * (c ** 3)
    return np.exp(a * Hpl + b)


def patelCharacteristics(thetapl, Hpl, r_0, phi, U_e, nu, Xs, flags, M_e, r_f, mu, rho_e):
    """Transforms planar boundary layer definitions into axisymmetric thick boundary layer definitions

    Author:  Carlos E. Ribeiro Santa Cruz Mendoza

    Sources:
        [1] Nakayama, A.; Patel, V. C. & Landweber, L.: Flow interaction near the tail of
            a body of resolution: Part 2: Iterative solution for row within and exterior to
            boundary layer and wake. Journal of Fluids Engineering, Transactions of the
            ASME 98 (1976), 538-546.
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
    Hpl_star = 3.3 + 1.535 * (Hpl - 0.7) ** (-2.715)
    delta = (Hpl_star + Hpl) * thetapl
    Q = U_e * ((r_0 + 0.5 * delta * np.cos(phi)) * delta - Delta_star)
    C_e = np.exp(-3.512 - 0.617 * np.log(Hpl_star - 3))
    c_f = frictionLaw(U_e, thetapl, nu, Hpl)
    n = 2 / (Hpl - 1)  # power-law parameter for velocity profile
    return H, Theta, Delta_star, delta, C_e, c_f, Q, n


def patelPlanar(thetapl, eps, Hpl, phi, r_0, Theta, Q, U_e, Xs, i):
    """Solves system of equations (f1,f2) iteratively to obtain planar characteristics from axisymmetric

    Author:  Carlos E. Ribeiro Santa Cruz Mendoza

    Sources:
        [1] Nakayama, A.; Patel, V. C. & Landweber, L.: Flow interaction near the tail of
            a body of resolution: Part 2: Iterative solution for row within and exterior to
            boundary layer and wake. Journal of Fluids Engineering, Transactions of the
            ASME 98 (1976), 538-546.
        [12] Patel, V. C.: On the equations of a thick axisymmetric turbulent boundary layer.
            Tech. rep., Iowa Institute of Hydraulic Research 1973.
        [13] Patel, V. C.: a Simple Integral Method for the Calculation of Thick Axisymmetric
            Turbulent Boundary Layers. Aeronaut. Quart. 25 (1974).
    """
    # Solve nonlinear system to obtain planar variables at [i+1] as a function of axisymmetric properties

    thetapl_1 = float(thetapl)
    thetapl, Hpl = fsolve(equations, (thetapl_1, 1.1), args=(phi, r_0, Theta, Q, U_e), xtol=1e-10, factor=0.1,
                          maxfev=1000)

    return Hpl, thetapl


def equations(vars, *data):
    phi, r_0, Theta, Q, U_e = data
    thetapl, Hpl = vars
    B = np.cos(phi) * (((Hpl ** 2) * (Hpl + 1)) / ((Hpl - 1) * (Hpl + 3)))
    a = (B * (thetapl ** 2) + r_0 * thetapl - Theta)
    Hpl_star = 3.3 + 1.535 * np.power((Hpl - 0.7), (-2.715))
    b = np.cos(phi) * ((Hpl_star + Hpl) ** 2) + (2 * r_0 / thetapl) * Hpl_star - 2 * Q / (U_e * thetapl ** 2) - B * (
                Hpl + 1)
    return (a, b)


def f1(thetapl, *data):
    phi, r_0, Theta, Hpl = data
    B = np.cos(phi) * (((Hpl ** 2) * (Hpl + 1)) / ((Hpl - 1) * (Hpl + 3)))
    f1 = (B * (thetapl ** 2) + r_0 * thetapl - Theta)
    return f1


def f2(Hpl, *data):
    phi, r_0, Q, U_e, thetapl = data
    B = np.cos(phi) * (((Hpl ** 2) * (Hpl + 1)) / ((Hpl - 1) * (Hpl + 3)))
    Hpl_star = 3.3 + 1.535 * np.power((Hpl - 0.7), (-2.715))
    f2 = np.cos(phi) * ((Hpl_star + Hpl) ** 2) + (2 * r_0 / thetapl) * Hpl_star - 2 * Q / (U_e * thetapl ** 2) - B * (
                Hpl + 1)
    return f2
