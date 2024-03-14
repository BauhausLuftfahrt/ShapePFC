"""Solves the Laminar Boundary Layer, obtain transition point and compute initial values for turbulent region

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, A. Habermann

 Args:
    Air_prop        [-]     1-D array air properties
    M_inf           [-]     Free stream Mach number
    gamma           [-]     Specific heat ratio
    tr              [-]     Transition option (user input or apply model)
    Vx_e            [-]     Dimensionless X-component of the edge velocity (rectangular coordinates, divided by u_inf)
    Vy_e            [-]     Dimensionless Y-component of the edge velocity (rectangular coordinates, divided by u_inf)
    u_e             [-]     Dimensionless edge velocity (divided by u_inf)
    p_e             [Pa]    Static pressure at the edge of the boundary layer
    rho_e           [kg/m^3]    Density at the edge of the boundary layer
    M_e             [-]     Mach number at the edge of the boundary layer
    Xs              [m]     1-D array X-coordinate of discretized nodes
    r_0             [m]     1-D array Y-coordinate of discretized nodes (local transverse radius)
    S               [m]     1-D array Segment sizes
    phi             [rad]   1-D array Segment angle w.r.t symmetry axis

Returns: 
    Xtr             [-]     x/L position of transition (node position)
    thetapl_in      [-]     Planar momentum thickness at the first turbulent point
    Hpl_in          [-]     Planar shape factor at the first turbulent point
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

Sources:
    [6] Rott, N. & Crabtree, L. F.: Simplifed Laminar Boundary-Layer Calculations
        for Bodies of Revolution and Yawed Wings. Journal of the Aeronautical Sciences
        19 (1952), 553-565.
    [7] Cebeci, T. & Bradshaw, P.: Physical and Computational Aspects of Convective
        Heat Transfer. Springer Berlin Heidelberg 1984, ISBN 978-3-662-02413-3.
    [8] Kays, W. & Crawford, M.: Convective Heat and Mass Transfer. McGraw-Hill
        series in mechanical engineering, McGraw-Hill 1993, ISBN 9780070337213.
    [9] Preston, J. H.: The minimum Reynolds number for a turbulent boundary layer
        and the selection of a transition device. Journal of Fluid Mechanics 3 (1958),
        373-384, ISSN 14697645.
    [10] Crabtree, L. F.: Prediction of transition in the boundary layer on an aerofoil.
        The Journal of the Royal Aeronautical Society 62 (1958), 525-528.
    [11] Cebeci, T.; Mosinskis, G. & Smith, A.: Calculation of Viscous Drag and Tur-
        bulent Boundary-Layer Separation on Two-Dimensional and Axisymetric Bod-
        ies in Incompressible Flows. Defense Technical Information Center 1970.
"""

# Built-in/Generic Imports
import numpy as np
from scipy.optimize import fsolve
from scipy import interpolate
import warnings

# Own modules
from panel.integral_boundary_layer.laminar_boundary_layer.calculate_laminar_characteristics import \
    laminarCharacteristics


def calc_laminar_parameters(atmos, delta, tr, potentialSolution, surface):
    u_inf = atmos.ext_props['u']
    u_e = potentialSolution[2] * u_inf
    Xs = surface[0]
    r_0 = surface[1]
    phi = surface[3]
    # Load Crabtree's curve
    lambda_crab = np.array([-0.013395069178525584, -0.01276659242272962, -0.01214250447109424, -0.011511394432801924,
                            -0.010877651112013257, -0.009924402848333902, -0.008967643541326079, -0.007684357204770632,
                            -0.005749772330784171, -0.0038116764134692414, -0.000892243885847202, 0.0026837537441985505,
                            0.008221546834026403, 0.0150628147595484, 0.02288454153454537, 0.0300576030546077,
                            0.03527413567988062, 0.04081543981303695, 0.04733808055650038, 0.05288201797215304,
                            0.05875072689568911, 0.06461943581922516, 0.07048814474276123, 0.07635773142712941,
                            0.08222731811149757, 0.08613949814024424])
    Re_crab = np.array([2173.118574516408, 2082.770652066578, 1976.2938743265925, 1895.623265050855, 1824.6299689492103,
                        1727.817337970836, 1643.9077912245857, 1559.9850780658535, 1469.584489966096,
                        1392.0869860984624,
                        1321.0015251094453, 1262.792815527589, 1214.1824206449346, 1155.8420469382602,
                        1100.6879450521717,
                        1061.6890312812016, 1032.4464291592146, 996.7391185086842, 967.4438507367702, 941.4138532603329,
                        908.9191472553514, 876.4244412503701, 843.9297352453889, 814.6608002984387, 785.3918653514884,
                        762.6534709954906])
    Crabtree = interpolate.interp1d(lambda_crab, Re_crab, fill_value="extrapolate")
    # Load Michel-e9 Curve
    Re_Lm = np.array(
        [5.910672986, 5.99935298, 6.096149806, 6.218464155, 6.336354831, 6.435004259, 6.538766972, 6.640631845,
         6.751338813, 6.830101508, 6.932442123, 7.021065845, 7.118403492, 7.241107044, 7.369886709, 7.496925206,
         7.625858936, 7.731581226, 7.873087217, 8.0440174, 8.241098546, 8.419693249, 8.597451983, 8.764477642,
         8.93963914, 9.128110636])
    Re_tm = np.array(
        [2.83064311, 2.867549739, 2.911372293, 2.965436715, 3.024043319, 3.06686262, 3.110221476, 3.157820365,
         3.198786857, 3.237257738, 3.272077173, 3.315023895, 3.347046711, 3.394221077, 3.438106827, 3.486564624,
         3.519791243, 3.553792092, 3.583461736, 3.637444299, 3.687682632, 3.72638697, 3.765578345, 3.801137616,
         3.836950336, 3.864208389])
    Michel = interpolate.interp1d(Re_Lm, Re_tm, fill_value="extrapolate")

    # Initializations and definitions
    lambda_1_ = np.zeros(len(Xs))
    Theta_lam = np.zeros(len(Xs))
    H_lam = np.zeros(len(Xs))
    Delta_star_lam = np.zeros(len(Xs))
    delta_star_lam = np.zeros(len(Xs))
    theta_lam = np.zeros(len(Xs))
    delta_starPhys_lam = np.zeros(len(Xs))
    C_f_lam = np.zeros(len(Xs))
    delta_lam = np.copy(delta)

    k = 0
    # Compute point of transition to turbulent flow (Xtr) and momentum thickness at transition (theta)
    if tr == 1:  # Computation of transition point according to Preston 1958
        transition = 'Preston criterion'
        for i in range(1, len(Xs)):
            theta_lam[i], Re_theta, dudx, lambda_1_[i], Xtr, C_f_lam[i], H_lam[i], delta_lam[
                i], Re_x = laminarCharacteristics(atmos,
                                                  potentialSolution,
                                                  surface, i)
            if Re_theta > 320:
                k = i
                break
    elif tr == 2:  # Transition according to michel-e^9 method as in Parsons (1972)
        transition = 'Michel-e9 criterion'
        for i in range(1, int(len(Xs))):
            theta_lam[i], Re_theta, dudx, lambda_1_[i], Xtr, C_f_lam[i], H_lam[i], delta_lam[
                i], Re_x = laminarCharacteristics(atmos,
                                                  potentialSolution,
                                                  surface, i)
            if np.log10(Re_x) > 7.75:
                k = i
            else:
                if np.log10(Re_theta) > Michel(np.log10(Re_x)):
                    k = i
            if k != 0:
                break
    elif tr == 3:  # Transition according to Crabtree (1958), momentum thickness according to Rott-Crabtree (1952)
        transition = 'Crabtree criterion'
        for i in range(1, int(len(Xs))):
            theta_lam[i], Re_theta, dudx, lambda_1_[i], Xtr, C_f_lam[i], H_lam[i], delta_lam[
                i], Re_x = laminarCharacteristics(atmos,
                                                  potentialSolution,
                                                  surface, i)
            if -lambda_1_[i] > 0.09:
                k = i
            else:
                if Re_theta > Crabtree(-lambda_1_[i]):
                    k = i
            if k != 0:
                break
    else:  # Transition point as user input
        transition = 'User input'
        if tr == 0:
            k = 1
            theta_lam[1], Re_theta, dudx, lambda_1_[1], Xtr_dummy, C_f_lam[1], H_lam[1], delta_lam[
                1], Re_x = laminarCharacteristics(
                atmos, potentialSolution, surface, 1)
            Xtr = Xs[1]
        else:
            Xtr = tr
            i = 1
            if Xs[0] > min(Xs):
                while Xtr < Xs[i] / Xs[0]:
                    k = i
                    theta_lam[i], Re_theta, dudx, lambda_1_[i], Xtr_dummy, C_f_lam[i], H_lam[i], delta_lam[
                        i], Re_x = laminarCharacteristics(
                        atmos, potentialSolution, surface, i)
                    i = i + 1
                Xtr = Xs[k]
            else:
                while Xtr > Xs[i] / Xs[-1]:
                    k = i
                    theta_lam[i], Re_theta, dudx, lambda_1_[i], Xtr_dummy, C_f_lam[i], H_lam[i], delta_lam[
                        i], Re_x = laminarCharacteristics(
                        atmos, potentialSolution, surface, i)
                    i = i + 1
                Xtr = Xs[k]

    # Compute Shape Factor immediately downstream from transition
    # Ludwieg & Tillman 1949 Skin friction law to start iteration
    ct = 0
    Hpl_in = 0
    H_new = 1.4  # Initial guess, based on equilibrium BL in flat plates
    c_f = 0.246 * (10 ** (-0.678 * H_new)) * Re_theta ** (-0.268)  # Initial guess based on Ludwieg & Tillman
    eps = 1e-10  # Tolerance for shape factor convergence
    # Loop to obtain shape factor at beginning of turbulent region
    # Based on the several works by Nash cited above
    while abs(1 - Hpl_in / H_new) > eps:
        Hpl_in = H_new
        PI = -(Hpl_in / (c_f / 2)) * (theta_lam[k] / u_e[k]) * dudx
        G = 6.1 * (PI + 1.81) ** 0.5 - 1.7
        K = 1.5 * G + (2110) / (G ** 2 + 200) - 14.8
        c_f = ((5.75 * np.log10(Re_theta * Hpl_in) + K) ** (-2)) * 2
        H_new = (1 - (G / ((c_f / 2) ** (-0.5)))) ** (-1)
        ct = ct + 1
        if ct > 1000:
            break

    Hpl_in = H_new
    # Replace momentum thickness with momentum deficit area and compute other variables in laminar region
    # (axisymmetric momentum thickness times local transverse radius)
    for i in range(0, k + 1):
        Theta_lam[i] = theta_lam[i] * r_0[i]
        Delta_star_lam[i] = Theta_lam[i] * H_lam[i]
        delta_star_lam[i] = Delta_star_lam[i] / r_0[i]
        inp = (Delta_star_lam[i], phi[i], r_0[i])
        delta_starPhys_lam[i] = fsolve(physicalThick, 1e-3, args=inp, xtol=1e-12)

    # The laminar calculations yield the axisymmetric momentum thickness, but we need the planar definition as input
    # Thus we solve the system:
    inp = (phi[k], H_lam[k], theta_lam[k], r_0[k])
    [thetapl_in] = fsolve(findFirstThickness, 1e-5, args=inp, xtol=1e-12, maxfev=10000)

    return thetapl_in, Hpl_in


def findFirstThickness(thetapl, *inp):
    phi, Hpl, theta_0, r_0 = inp
    return (1 + 2 * (0.5 * np.cos(phi) * (((Hpl ** 2) * (Hpl + 1)) / ((Hpl - 1) * (Hpl + 3))) * (
            thetapl / r_0))) * thetapl - theta_0


def physicalThick(disp, *inp):
    Delta_star, phi, r_0 = inp
    return Delta_star - disp * (r_0 + 0.5 * disp * np.cos(phi))
