"""Solves the Boundary Layer development of external flow around axisymmetric body

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, A. Habermann

 Args:
    thetapl_in      [m]     Planar momentum thickness at the first turbulent point
    Hpl_in          [-]     Planar shape factor at the first turbulent point
    Vx_e            [-]     Dimensionless X-component of the edge velocity (rectangular coordinates, divided by u_inf)
    Vy_e            [-]     Dimensionless Y-component of the edge velocity (rectangular coordinates, divided by u_inf)
    u_e             [-]     1-D array Dimensionless edge velocity (divided by u_inf)
    p_e             [Pa]    1-D array Static pressure at the edge of the boundary layer
    rho_e           [kg/m^3]    1-D array Density at the edge of the boundary layer
    M_e             [-]     1-D array Mach number at the edge of the boundary layer
    Xs              [m]     1-D array X-coordinate of discretized nodes
    r_0             [m]     1-D array Y-coordinate of discretized nodes (local transverse radius)
    S               [m]     1-D array Segment sizes
    phi             [rad]   1-D array Segment angle w.r.t symmetry axis
    eps             [-]     Relative Tolerance for convergence check
    end             [-]     index of last calculation point
    counter         [-]     Number of viscid/inviscid iterations
    r_f             [-]     Temperature recovery factor
    Air_prop        [-]     Tuple air properties
    M_inf           [-]     Freestream Mach number
    phi_d           [rad]   1-D array Flow angle w.r.t symmetry axis
    filled up to transition point:
    Theta:          [m^2]     1-D array Momentum deficit area
    H               [-]     1-D array Shape factor
    delta           [m]     1-D array Boundary layer thickness
    C_f             [-]     1-D array Friction coefficient
    n               [-]     1-D array Exponent of velocity profile power-law
    delta_starPhys  [m]     1-D array Displacement thickness
    p_s             [Pa]    1-D array Static pressure at body's surface
    theta           [m]     1-D array Momentum thickness
    Delta_star      [m^2]     1-D array Displacement area
    C_e             [-]     1-D array Entrainment coefficient

Returns:
    Theta:          [m^2]     1-D array Momentum deficit area
    H               [-]     1-D array Shape factor
    delta           [m]     1-D array Boundary layer thickness
    C_f             [-]     1-D array Friction coefficient
    n               [-]     1-D array Exponent of velocity profile power-law
    delta_starPhys  [m]     1-D array Displacement thickness
    p_s             [Pa]    1-D array Static pressure at body's surface
    theta           [m]     1-D array Momentum thickness
    Delta_star      [m^2]   1-D array Displacement area

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

# Built-in/Generic Imports
import numpy as np
from scipy.optimize import fsolve
from panel.integral_boundary_layer.turbulent_boundary_layer.calculate_characteristics_patel import patelCharacteristics, \
    patelPlanar
from panel.integral_boundary_layer.turbulent_boundary_layer.turbulent_separation import turbulent_separation


def turbulentPatel(thetapl, Hpl, laminarSolution, potentialSolution, surface, n, flags, eps, end, counter, r_f, atmos,
                   M_inf, phi_d):
    # Initialize Variables
    u_inf = atmos.ext_props['u']
    mu = atmos.ext_props['mue']  # Dynamic viscosity [Pa.s]
    nu = atmos.ext_props['nue']  # Kinematic viscosity [m²/s]
    Xs = surface[0]
    r_0 = surface[1]
    phi = surface[3]
    Vx_e = potentialSolution[0] * u_inf
    Vy_e = potentialSolution[1] * u_inf
    u_e = potentialSolution[2] * u_inf
    p_e = potentialSolution[3]
    rho_e = potentialSolution[4]
    M_e = potentialSolution[5]
    Theta = laminarSolution[0]
    H = laminarSolution[1]
    delta = laminarSolution[2]
    k = laminarSolution[4]
    delta_star = laminarSolution[5]
    theta = laminarSolution[6]
    Delta_star = laminarSolution[7]
    delta_starPhys = laminarSolution[8]
    C_f = laminarSolution[11]

    # Initialize volume flow and pressure on the body surface
    Q = np.zeros(len(Xs))
    p_s = p_e

    def calc_ip_ik(Vx_e, Vy_e, delta, dx, kappa, r_0, phi_d, n, i):
        dVdx = (Vy_e[i + 1] - Vy_e[i]) / dx
        dDdx = (delta[i + 1] - delta[i]) / dx
        dVddx = dVdx / delta[i] - Vy_e[i] * (1 / delta[i] ** 2) * dDdx  # derivative of Vy_e w.r.t. delta (chain rule)
        p_0 = pressureGrad(kappa, n[i + 1], Vx_e[i + 1], Vy_e[i + 1], dVddx, delta[i + 1])
        # p_s[i] = p_0*rho_e[i]+p_e[i]
        p_s[i] = pressureGrad(kappa, n[i], Vx_e[i], Vy_e[i], dVddx, delta[i]) * rho_e[i] + p_e[
            i]  # pressure on surface of body
        dVddx = dVdx / delta[i] - Vy_e[i] * (1 / delta[i] ** 2) * dDdx
        dpdx = (p_0 - pressureGrad(kappa, n[i], Vx_e[i], Vy_e[i], dVddx, delta[i])) / dx
        ip = I_p(dpdx, pressureGrad(kappa, n[i], Vx_e[i], Vy_e[i], dVddx, delta[i]), delta[i], r_0[i], phi_d[i],
                 Vy_e[i], dVdx, Vx_e[i], dDdx)
        ik = I_k(kappa, delta[i], n[i], r_0[i], phi_d[i], Vx_e[i], Vy_e[i])
        return [ip, ik]

    # Compute Turbulent Boundary Layer
    # phi = np.array([np.arcsin(np.sin(phi[i])) for i in range(k,end)])
    dx = np.array([Xs[i + 1] - Xs[i] for i in range(k, end)])
    drdx = np.array([np.sin(phi[i]) for i in range(k, end)])
    dUdx = np.array([(u_e[i + 1] - u_e[i]) / dx[i - k] for i in range(k, end)])
    dphi = np.array([(np.arcsin(np.sin(phi[i + 1])) - np.arcsin(np.sin(phi[i]))) for i in range(k, end)])
    kappa_init = np.array([-(dphi[i] / dx[i]) for i in range(0, end - k)])  # longitudinal curvature at [i]
    kappa_d = np.array(
        [-((np.arcsin(np.sin(phi_d[i + 1])) - np.arcsin(np.sin(phi_d[i]))) / dx[i - k]) for i in range(k, end)])
    # average between body's and BL edge (flow) curvature
    kappa = np.array([(0.5 * kappa_init[i] + 0.5 * kappa_d[i]) for i in range(0, end - k)])
    # Compute curvature and pressure integrals
    ip_ik = [
        calc_ip_ik(Vx_e, Vy_e, delta, dx[i - k], kappa[i - k], r_0, phi_d, n, i) if (counter > 0 and i > k) else [0, 0]
        for i in range(k, end)]
    Ip = np.array([ip_ik[i][0] for i in range(0, len(ip_ik))])
    Ik = np.array([ip_ik[i][1] for i in range(0, len(ip_ik))])

    # Compute Turbulent Boundary Layer
    for i in range(k, end):
        # find relationship between planar and axisymmetric parameters
        H[i], Theta[i], Delta_star[i], delta[i], C_e, c_f, Q[i], n[i] = patelCharacteristics(thetapl, Hpl, r_0[i],
                                                                                             phi[i], u_e[i],
                                                                                             nu, Xs[i], flags, M_e[i],
                                                                                             r_f, mu, rho_e[i])
        # find physical displacement thickness
        inp = (Delta_star[i], phi[i], r_0[i])
        delta_starPhys[i] = fsolve(physicalThick, 1e-3, args=inp, xtol=1e-12)
        delta_star[i] = Delta_star[i] / r_0[i]
        theta[i] = Theta[i] / r_0[i]
        C_f[i] = c_f

        # Integrate Momentum and Entrainment equations by simple Euler Integration to obtain variables at [i+1]
        f_0 = momentumIntegralPatel(c_f, u_e[i], r_0[i], dUdx[i - k], Theta[i], Delta_star[i], Ik[i - k], Ip[i - k],
                                    M_e[i])
        g_0 = psiIntegralPatel(C_e, u_e[i], r_0[i], delta[i], kappa[i - k])
        Theta[i + 1] = Theta[i] + dx[i - k] * f_0
        Q[i + 1] = Q[i] + g_0 * dx[i - k]

        # Predictor-Corrector loop to increase integrals accuracy
        theta_pred = 0
        ct = 0
        while abs(1 - theta_pred / Theta[i + 1]) > eps:
            theta_pred = Theta[i + 1]

            # find planar and axisymmetric properties
            Hpl, thetapl = patelPlanar(thetapl, eps, Hpl, phi[i + 1], r_0[i + 1], Theta[i + 1], Q[i + 1], u_e[i + 1],
                                       Xs, i)
            H[i + 1], Theta[i + 1], Delta_star[i + 1], delta[i + 1], C_e, c_f, Q[i + 1], n[
                i + 1] = patelCharacteristics(thetapl,
                                              Hpl,
                                              r_0[i + 1],
                                              phi[i + 1],
                                              u_e[i + 1],
                                              nu, Xs[i], flags, M_e[i + 1], r_f, mu, rho_e[i + 1])
            # Compute curvature and pressure integrals with new parameters
            if counter > 0:
                dVdx = (Vy_e[i + 1] - Vy_e[i]) / dx[i - k]
                dDdx = (delta[i + 1] - delta[i]) / dx[i - k]
                dVddx = dVdx / delta[i] - Vy_e[i] * (1 / delta[i] ** 2) * dDdx  #
                p_n0 = pressureGrad(kappa[i - k], n[i], Vx_e[i], Vy_e[i], dVddx, delta[i])
                dVddx = dVdx / delta[i + 1] - Vy_e[i + 1] * (1 / delta[i + 1] ** 2) * dDdx
                p_0 = pressureGrad(kappa[i - k], n[i + 1], Vx_e[i + 1], Vy_e[i + 1], dVddx, delta[i + 1])
                dpdx = (p_0 - p_n0) / dx[i - k]
                Ip[i - k] = I_p(dpdx, p_0, delta[i + 1], r_0[i + 1], phi_d[i + 1], Vy_e[i + 1], dVdx, Vx_e[i + 1], dDdx)
                Ik[i - k] = I_k(kappa[i - k], delta[i + 1], n[i + 1], r_0[i + 1], phi_d[i + 1], Vx_e[i + 1],
                                Vy_e[i + 1])
            else:
                Ip[i - k] = 0
                Ik[i - k] = 0

            C_f[i + 1] = c_f
            # Correct momentum and stream integrals
            f_1 = momentumIntegralPatel(c_f, u_e[i + 1], r_0[i + 1], dUdx[i - k],
                                        Theta[i + 1], Delta_star[i + 1], Ik[i - k], Ip[i - k], M_e[i + 1])
            g_1 = psiIntegralPatel(C_e, u_e[i + 1], r_0[i + 1], delta[i + 1], kappa[i - k])

            # stabilize loop with weight factor
            weightfactor = 0.75
            Theta[i + 1] = corrector(Theta[i], dx[i - k], f_0, f_1, weightfactor)
            Q[i + 1] = corrector(Q[i], dx[i - k], g_0, g_1, weightfactor)

            # Solve nonlinear system again to obtain corrected planar variables at [i+1]
            Hpl, thetapl = patelPlanar(thetapl, eps, Hpl, phi[i + 1], r_0[i + 1], Theta[i + 1], Q[i + 1], u_e[i + 1],
                                       Xs, i)
            ct = ct + 1
            if ct > 100:
                break

    # detect point of turbulent transition and re-calculate characteristics for zone of separation
    Theta_new, H_new, delta_new, delta_starPhys_new, p_s_new, theta_new, Delta_star_new = \
        turbulent_separation(Theta, H, delta, delta_starPhys, p_s, theta, Delta_star, Xs, k, u_e)

    print("Boundary Layer computation finished succesfully")
    return Theta_new, H_new, delta_new, C_f, n, delta_starPhys_new, p_s, theta_new, Delta_star_new


def findFirstThickness(thetapl, *inp):
    phi, Hpl, theta_0, r_0 = inp
    return (1 + 2 * (0.5 * np.cos(phi) * (((Hpl ** 2) * (Hpl + 1)) / ((Hpl - 1) * (Hpl + 3))) * (
                thetapl / r_0))) * thetapl - theta_0


def momentumIntegralPatel(c_f, u_e, r_0, dUdx, Theta, Delta_star, Ik, Ip, M_e):
    return 0.5 * c_f * r_0 + Ik + Ip - (2 * Theta + Delta_star - Theta * M_e ** 2) * dUdx / u_e
    # return 0.5 * c_f*r_0 + Ik + Ip - (2*Theta + Delta_star) * dUdx / u_e


def psiIntegralPatel(c_e, u, r, delta, kappa):
    return c_e * u * (r + delta) * (1 + (kappa) * delta)


def corrector(y, h, f_0, f_1, omega):
    return y + h * (omega * f_0 + (1 - omega) * f_1)


def pressureGrad(kappa, n, U, V, dVdx, delta):
    p_0 = (n / (2 * n + 1)) * U * (delta ** 2) * dVdx + 0.5 * (V ** 2) \
          - (kappa * delta) * (n / (3 * n + 1)) * U * (delta ** 2) * dVdx \
          - (kappa * delta) * (2 / (n + 2)) * U ** 2
    return p_0


def I_p(dpdx, p_0, delta, r_0, phi, V, dVdx, U, dDdx):
    I_p = (1 / U ** 2) * (dpdx * (delta ** 2) * ((2 / 3) * (r_0 / delta) + 0.25 * np.cos(phi)) +
                          p_0 * delta * dDdx * ((2 / 3) * (r_0 / delta) + 0.5 * np.cos(phi)) -
                          V * dVdx * (delta ** 2) * ((r_0 / delta) + 0.5 * np.cos(phi)))
    return I_p


def I_k(kappa, delta, n, r_0, phi, U, V):
    I_k = (1 / U ** 2) * (kappa * delta) * ((n / (2 * n + 1)) * r_0 + (n / (3 * n + 1)) * np.cos(phi) * delta) * U * V
    return -I_k


def physicalThick(disp, *inp):
    Delta_star, phi, r_0 = inp
    return Delta_star - disp * (r_0 + 0.5 * disp * np.cos(phi))
