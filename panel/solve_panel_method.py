"""Solves the Boundary Layer development of external flow around axisymmetric body (along one meridional section)

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, Nikolaus Romanow, A. Habermnn

Hypothesis: Geometry is fully axisymmetric (loaded as points on the x-y plane)
            Angle of Attack is zero (free-stream parallel to axis of symmetry)

 Args:
    geom:           [m,m]   2D array x,y-coordinates of axisymmetric profile loaded from .txt file
    M_inf           [-]     Freestream Mach number
    Alt:            [m]     Altitude from sea level
    N:              [-]     Number of nodes for profile discretization
    w:              [-]     Weighting factor between arc-length and curvature based parametrisation
    FPR             [-]     Fuselage fan pressure ratio
    etapol_fan      [-]  Polytropic efficiency of fuselage fan
    AR_ell          [-]     Aspect ratio of elliptic fuselage cross-section
    flags           [-]     1-D array Calculation options
    counter         [-]     Counter of interactions between viscid and inviscid flow
    eps             [-]     Relative Tolerance for convergence of linear system and integration loops of turbulent calculations
    eps_2           [-]     Relative tolerance for convergence of Inviscid & Viscid interaction
    Xn_old          [m]     Sampling of previously calculated meridional section (sampling must be equal for elliptic body)
    
Returns:
    Xn              [m]     Sampling of meridional section
    file            [-]     File name
    
    Saved in .csv files
        Theta:          [m^2]     1-D array Momentum deficit area
        H               [-]     1-D array Shape factor
        delta           [m]     1-D array Boundary layer thickness
        C_f             [-]     1-D array Friction coefficient
        n               [-]     1-D array Exponent of velocity profile power-law
        delta_starPhys  [m]     1-D array Displacement thickness
        p_s             [Pa]    1-D array Static pressure at body's surface
        theta           [m]     1-D array Momentum thickness
        Delta_star      [m^2]     1-D array Displacement area
        Cp              [-]     1-D array Pressure coefficient 
        dS              [m]     Cumulative arc length of body contour
        
        Vx_e            [-]   1-D array Dimensionless X-component of the edge velocity (rectangular coordinates, divided by u_inf)
        Vy_e            [-]   1-D array Dimensionless Y-component of the edge velocity (rectangular coordinates, divided by u_inf)
        u_e             [-]   1-D array Dimensionless edge velocity (divided by u_inf)
        p_e             [Pa]  1-D array Static pressure at the edge of the boundary layer
        rho_e           [kg/m^3] 1-D array Density at the edge of the boundary layer
        M_e             [-]      1-D array Mach number at the edge of the boundary layer

Sources:
    General algorithm:
    [1] Nakayama, A.; Patel, V. C. & Landweber, L.: Flow interaction near the tail of
        a body of resolution: Part 2: Iterative solution for row within and exterior to
        boundary layer and wake. Journal of Fluids Engineering, Transactions of the
        ASME 98 (1976), 538-546.
    [2] Green,  J.  E.;  Weeks,  D.  J.  &  Brooman,  J.  W.  F.:   Prediction  of  turbulent
        boundary layers and wakes in compressible flow by a lag-entrainment method. ARC R&M 3791 (1973)
    Further literature in Literature file and in specific python files
"""

# Built-in/Generic Imports
import os
import glob
import numpy as np
from scipy import interpolate
from bhlpythontoolbox.atmosphere.atmosphere import Atmosphere

# Own modules
from geometry_generation.panel_geometry.generate_panel_geometries import bodyGeometry
from panel.potential_flow.calculate_potential_flow import potentialFlow
from panel.integral_boundary_layer.laminar_boundary_layer.laminar_region import laminarRegion
from panel.integral_boundary_layer.laminar_boundary_layer.calculate_transition_parameters import calc_laminar_parameters
from panel.integral_boundary_layer.turbulent_boundary_layer.turbulent_region_patel import turbulentPatel
from panel.integral_boundary_layer.turbulent_boundary_layer.turbulent_region_green import turbulentGreen
from panel.integral_boundary_layer.fan.fan_parameters import rotorAveraged
from panel.integral_boundary_layer.fan.fan_parameters import thermoSys


class CalculateBL:

    def __init__(self, AR_ell, Xn_old):
        self.AR_ell = AR_ell
        self.Xn_old = Xn_old

    def calculateBoundaryLayer(self, Alt, M_inf, N, w, FPR, etapol_fan, geom, flags, counter, eps, eps_2, filename,
                               sing_type=0, X_fuse=None, Y_fuse=None):

        # Atmospheric and Free-stream defitions
        atmos = Atmosphere(Alt, ext_props_required=['rho', 'sos', 'mue', 'nue'],
                           backend='bhl')  # Standard atmospheric properties
        rho_inf = atmos.ext_props['rho']  # Density [kg/m³]
        c = atmos.ext_props['sos']  # Speed of sound [m/s]
        T_inf = atmos.temperature  # Static temperature
        p_inf = atmos.pressure  # Static pressure on free-stream [Pa]
        u_inf = M_inf * c  # Free-stream velocity [m/s]
        atmos.ext_props['u'] = u_inf
        mu = atmos.ext_props['mue']  # Dynamic viscosity [Pa.s]
        nu = atmos.ext_props['nue']  # Kinematic viscosity [m²/s]
        gamma = 1.401  # Specific heat ratio [-]
        R = 287.058  # Specific gas constant [J kg^-1 K^-1]
        r_f = 1  # Temperature recovery factor [-]
        p_t = p_inf * (1 + 0.5 * (gamma - 1) * M_inf ** 2) ** (gamma / (gamma - 1))
        T_t_inf = T_inf * (1 + 0.5 * (gamma - 1) * M_inf ** 2)  # Total temperature [K]
        rho_t_inf = rho_inf * (1 + 0.5 * (gamma - 1) * M_inf ** 2) ** (1 / (gamma - 1))  # Total density [kg/m³]
        atmos.ext_props['mach'] = M_inf
        atmos.ext_props['gamma'] = gamma
        atmos.ext_props['u'] = u_inf
        atmos.ext_props['p_t'] = p_t
        atmos.ext_props['T_t'] = T_t_inf
        atmos.ext_props['rho_t'] = rho_t_inf

        # Obtain Geometric properties and experimental data, if available
        geometryData = bodyGeometry(flags, geom, N, w, self.AR_ell, self.Xn_old, u_inf, nu, M_inf, 0, X_fuse=X_fuse,
                                    Y_fuse=Y_fuse)
        Xn = geometryData[0]  # x-coordinates of profile points (edges of segments in the discretization)
        Yn = geometryData[1]  # y-coordinates of profile points (edges of segments in the discretization)
        Fm = geometryData[2]  # entrainment velocites
        N = geometryData[
            3]  # Number of nodes (can differ from input, depending on how many points are available in profile)
        L = L_ref = geometryData[4]  # Length of body
        arc_length = geometryData[5]
        file = geometryData[6]  # Name of folder where files will be saved
        cp_exp = geometryData[14]
        H_exp = geometryData[11]
        Re_L = u_inf * L_ref / nu
        atmos.ext_props['re_l'] = Re_L

        if flags[0] == 10:
            if Xn[1] != []:  # flip bottom nacelle (code works if nodes sampled in positive X-direction)
                Xn[1] = np.flip(Xn[1])
                Yn[1] = np.flip(Yn[1])

            Xn_1 = Xn * 1
            Yn_1 = Yn * 1

        print("Free-stream velocity: %5.2f [m/s]" % u_inf)
        print("Kinematic viscosity: %10.3e [m²/s]" % nu)
        print("Free-stream static pressure: %5.2f [Pa]" % p_inf)
        print("Free-stream temperature: %5.2f [K]" % T_inf)
        print("Free-stream density: %5.4f [kg/m³]" % rho_inf)
        print("Speed of Sound: %5.4f [m/s]" % c)
        print("Reynolds Number (Length): %10.3e [-]" % Re_L)
        print('---------------------------')
        print("Results will be written in folder studies/panel_studies/results/%s/" % filename, (file))

        # Create path to save results
        if os.path.exists('results/%s/' % filename + file):
            rem = glob.glob('results/%s/' % filename + file + '/*')
            for f in rem:
                os.remove(f)
        else:
            os.makedirs('results/%s/' % filename + file)

        if flags[0] == 10 or flags[0] == 11 or (flags[0] == 0 and len(Xn) > 1):
            pathBody = 'results/%s/' % filename + file + '/' + file + '.txt'
            np.savetxt(pathBody, np.column_stack((np.transpose(np.concatenate(Xn)), np.transpose(np.concatenate(Yn)))),
                       fmt='%s')
        elif flags[0] == 0 and len(Xn) == 1:
            pathBody = 'results/%s/' % filename + file + '/' + file + '.txt'
            np.savetxt(pathBody, np.column_stack((np.transpose(Xn), np.transpose(Yn))), fmt='%s')

        # Save input data
        pathInput = 'results/%s/' % filename + file + '/' + file + 'Input' + '.txt'
        pathGeom = 'results/%s/' % filename + file + '/' + file + 'Geom' + '.txt'

        if len(N) < 6:
            N.extend([0] * (6 - len(N)))

        np.savetxt(pathInput, np.transpose(np.array(
            [[Alt], [M_inf], [N[0]], [N[1]], [N[2]], [N[3]], [N[4]], [N[5]], [w], [self.AR_ell], [eps], [eps_2], [c[0]],
             [u_inf[0]], [nu[0]], [Re_L[0]], [flags[0]], [flags[1]], [flags[2]], [flags[3]], [flags[4]], [flags[5]]])),
                   delimiter=',',
                   header='Alt, M_inf, N1, N2, N3, N4, N5, N6, AspectRatio, WeightParam, eps, eps_2, c, u_inf, nu, Re_L, geometry, pg, Xtr, ItOnOff, ComprCor, Sampling',
                   fmt='%s')

        if flags[0] != 0:
            geom = file
        save = open(pathGeom, "w+")
        save.write(geom)
        save.close()

        # Set thickness to zero for first potential computation
        delta_total = [np.array(0) for _ in range(len(Xn))]
        delta_starPhys_total = [[] for _ in range(len(Xn))]
        delta_starPhys_BC_total = [[] for _ in range(len(Xn))]
        ue_BC_total = [[] for _ in range(len(Xn))]
        for sub in range(len(Xn)):
            if Xn[sub] != []:
                n_sub = np.shape([Xn[sub]])[1]
                delta_starPhys_total[sub] = np.zeros(n_sub)
                delta_starPhys_BC_total[sub] = np.zeros(
                    n_sub - 1)  # displacement thickness, input for transpiration technique
                ue_BC_total[sub] = np.zeros(
                    n_sub - 1)  # velocity at boundary layer edge, input for transpiration technique

        bl_charac = []
        bl_charac.append(delta_starPhys_total)
        bl_charac.append(delta_total)
        bl_charac.append(delta_starPhys_BC_total)
        bl_charac.append(ue_BC_total)

        # Solve potential flow (and correct for compressibility)
        # for the first iteration, the variable surface contains the body's geometrical data
        potentialSolution_total, surface_total, sigma, j_s, j_v = potentialFlow(Xn, Yn, Fm, flags, file, counter, atmos,
                                                                                filename, bl_charac, sing_type)

        # separate solutions for the different subbodies
        Vx_e_total = [[] for _ in range(len(Xn))]
        Vy_e_total = [[] for _ in range(len(Xn))]
        u_e_total = [[] for _ in range(len(Xn))]
        p_e_total = [[] for _ in range(len(Xn))]
        rho_e_total = [[] for _ in range(len(Xn))]
        M_e_total = [[] for _ in range(len(Xn))]
        Xs_total = [[] for _ in range(len(Xn))]
        r_0_total = [[] for _ in range(len(Xn))]
        S_total = [[] for _ in range(len(Xn))]
        phi_total = [[] for _ in range(len(Xn))]
        phi_d_total = [[] for _ in range(len(Xn))]
        dS_total = [[] for _ in range(len(Xn))]
        Cp_total = [[] for _ in range(len(Xn))]
        for sub in range(len(Xn)):
            if Xn[sub] != []:
                Vx_e_total[sub] = potentialSolution_total[sub][0] * u_inf
                Vy_e_total[sub] = potentialSolution_total[sub][1] * u_inf
                u_e_total[sub] = potentialSolution_total[sub][2] * u_inf
                p_e_total[sub] = potentialSolution_total[sub][3]
                rho_e_total[sub] = potentialSolution_total[sub][4]
                M_e_total[sub] = potentialSolution_total[sub][5]
                Xs_total[sub] = surface_total[sub][0]
                r_0_total[sub] = surface_total[sub][1]
                S_total[sub] = surface_total[sub][2]
                phi_total[sub] = surface_total[sub][3]
                if arc_length[sub] != []:  # only for solid bodies
                    dS_total[sub] = arc_length[sub](Xs_total[sub])
                    phi_d_total[sub] = surface_total[sub][
                                           3] * 1  # flow curvature cannot yet be assessed, so we use the body curvature
                    ue_BC_total[sub] = u_e_total[sub] * 1  # arc length values for each segment
                # Pressure coefficient for compressible flow
                Cp_total[sub] = (p_e_total[sub] - p_inf) / (0.5 * rho_inf * u_inf ** 2)

        boundary_layer = {}
        # stop calculation here, if only potential flow is required
        if flags[7] == 1:
            # get input values for boundary layer computation of fuselage
            # Todo: boundary layer computation of all other bodies (incl. nacelles)
            # if flags[0] > 0:
            delta = delta_total[0] * 1
            delta_starPhys = delta_starPhys_total[0] * 1
            potentialSolution = potentialSolution_total[0]
            surface = surface_total[0]
            Vx_e = Vx_e_total[0] * 1
            Vy_e = Vy_e_total[0] * 1
            u_e = u_e_total[0] * 1
            p_e = p_e_total[0] * 1
            rho_e = rho_e_total[0] * 1
            M_e = M_e_total[0] * 1
            Xs = Xs_total[0] * 1
            r_0 = r_0_total[0] * 1
            S = S_total[0] * 1
            phi = phi_total[0] * 1
            phi_d = phi_d_total[0] * 1
            dS = dS_total[0] * 1
            Cp = Cp_total[0] * 1
            delta_starPhys_BC = delta_starPhys_BC_total[0] * 1
            ue_BC = ue_BC_total[0] * 1

            # Initialize Variables
            delta = np.zeros(len(Xs))
            H = np.zeros(len(Xs))
            n = np.zeros(len(Xs))

            # Since it's a forward marching scheme we compute up to len(Xs) - 1
            end = int(len(Xs) - 1)
            # Compute characteristics in laminar region and immediately downstream of the transition point
            laminarSolution = laminarRegion(atmos, delta, flags[2], potentialSolution, surface)
            if M_inf > 0.3:
                # Preston
                thetapl_in_1, Hpl_in_1 = calc_laminar_parameters(atmos, delta, 1, potentialSolution, surface)
                # Michel
                thetapl_in_2, Hpl_in_2 = calc_laminar_parameters(atmos, delta, 2, potentialSolution, surface)
                weight_trans = 0
                thetapl_in = weight_trans * thetapl_in_1 + (1 - weight_trans) * thetapl_in_2
                Hpl_in = weight_trans * Hpl_in_1 + (1 - weight_trans) * Hpl_in_2
            else:
                thetapl_in, Hpl_in = calc_laminar_parameters(atmos, delta, 1, potentialSolution,
                                                             surface)

            if flags[1] == 0:
                Theta, H, delta, C_f, n, delta_starPhys, p_s, theta, Delta_star = turbulentPatel(thetapl_in, Hpl_in,
                                                                                                 laminarSolution,
                                                                                                 potentialSolution,
                                                                                                 surface, n,
                                                                                                 flags,
                                                                                                 eps,
                                                                                                 end, counter, r_f,
                                                                                                 atmos, M_inf,
                                                                                                 phi_d)
            else:
                Theta, H, delta, C_f, n, delta_starPhys, p_s, theta, Delta_star = turbulentGreen(thetapl_in, Hpl_in,
                                                                                                 laminarSolution,
                                                                                                 potentialSolution,
                                                                                                 surface, n,
                                                                                                 flags,
                                                                                                 eps, end, counter,
                                                                                                 r_f, atmos,
                                                                                                 M_inf,
                                                                                                 phi_d)

            # get input values for displacement effect of fuselage
            # (displacement effect of nacelle not included yet)
            delta_total[0] = delta * 1
            delta_starPhys_total[0] = delta_starPhys * 1
            delta_starPhys_BC_total[0] = delta_starPhys * 1

            if flags[0] == 10:
                # include rotor and stator from first iteration onwards
                Xn = Xn_1 * 1
                Yn = Yn_1 * 1

            if flags[0] == 10 and Xn[4] != [] and Xn[5] != []:
                # rotor inlet
                x_rot = min(Xn[4])
                d_rot = max(Yn[4]) - min(Yn[4])
                u_rot, mdot_rot, Tt_rot, pt_rot, T_rot, p_rot, rho_rot, Ma_rot = rotorAveraged(x_rot, d_rot, u_inf,
                                                                                               p_inf, M_inf,
                                                                                               delta_total[0],
                                                                                               u_e_total[0],
                                                                                               p_e_total[0],
                                                                                               rho_e_total[0],
                                                                                               Xs_total[0],
                                                                                               r_0_total[0],
                                                                                               phi_total[0], n, p_s,
                                                                                               end, gamma, R, c)
                # stator outlet
                A_stat = np.pi * (max(Yn[5]) ** 2 - min(Yn[5]) ** 2)
                mdot_stat = mdot_rot * 1
                pt_stat = pt_rot * FPR
                Tt_stat = Tt_rot * FPR ** ((gamma - 1) / (gamma * etapol_fan))
                init = T_rot, p_rot, rho_rot, u_rot, Ma_rot
                thermoSolution = thermoSys(mdot_stat, Tt_stat, pt_stat, A_stat, gamma, R, init)
                u_stat = thermoSolution[3]

                # specification of throughflow velocities
                Fm[4] = np.ones(len(Xn_1[4]) - 1) * u_rot / u_inf
                Fm[5] = np.ones(len(Xn_1[5]) - 1) * u_stat / u_inf

            fileit = file + 'BL' + str(counter) + '.txt'
            path = 'results/%s/' % filename + file + '/' + fileit
            np.savetxt(path, np.transpose([Theta, H, delta, C_f, n, delta_starPhys, p_s, theta, Delta_star, Cp, dS]),
                       delimiter=',', fmt='%s')  # dummy (dS gespeichert)

            if (flags[3] == 1 or flags[3] == 2):
                # Start iteration between viscid and inviscid flow
                delta_starPhys_1 = delta_starPhys * 0
                while np.allclose(delta_starPhys[0:end - 1], delta_starPhys_1[0:end - 1], rtol=eps_2) is False:
                    delta_starPhys_1 = np.copy(delta_starPhys)
                    counter = counter + 1
                    print("Viscid and Inviscid Flow interaction: starting iteration number %2d" % counter)
                    # Interpolate displacement thickness to element edges (currently on middle of element)
                    dp = interpolate.interp1d(Xs[0:end - 1], delta_starPhys[0:end - 1], fill_value="extrapolate")
                    delta_starPhys_total[0] = dp(Xn[0])
                    delta_starPhys_BC_total[0] = dp(Xs)
                    dp2 = interpolate.interp1d(Xs[0:end - 1], delta[0:end - 1], fill_value="extrapolate")
                    delta_total[0] = dp2(Xs)
                    bl_charac = []
                    bl_charac.append(delta_starPhys_total)
                    bl_charac.append(delta_total)
                    bl_charac.append(delta_starPhys_BC_total)
                    bl_charac.append(ue_BC_total)
                    # Compute potential flow field at edge of boundary layer considering displacement thickness
                    potentialSolution_total, newSurface_total, sigma, j_s, j_v = potentialFlow(Xn, Yn, Fm, flags, file,
                                                                                               counter, atmos, filename,
                                                                                               bl_charac)
                    # separate solutions for the different subbodies
                    Cp_e_total = [[] for _ in range(len(Xn))]
                    for sub in range(len(Xn)):
                        if Xn[sub] != []:
                            Vx_e_total[sub] = potentialSolution_total[sub][0] * u_inf
                            Vy_e_total[sub] = potentialSolution_total[sub][1] * u_inf
                            u_e_total[sub] = potentialSolution_total[sub][2] * u_inf
                            p_e_total[sub] = potentialSolution_total[sub][3]
                            rho_e_total[sub] = potentialSolution_total[sub][4]
                            M_e_total[sub] = potentialSolution_total[sub][5]
                            if arc_length[sub] != []:  # only for solid bodies
                                phi_d_total[sub] = (np.arctan2(Vy_e_total[sub], Vx_e_total[sub]))
                                ue_BC_total[sub] = u_e_total[sub] * 1
                            # Pressure coefficient for compressible flow
                            Cp_e_total[sub] = (p_e_total[sub] - p_inf) / (0.5 * rho_inf * u_inf ** 2)

                    # get input values for boundary layer computation of fuselage
                    # (boundary layer computation of nacelle not included yet)
                    delta = delta_total[0] * 1
                    delta_starPhys = delta_starPhys_total[0] * 1
                    potentialSolution = potentialSolution_total[0]
                    Vx_e = Vx_e_total[0] * 1
                    Vy_e = Vy_e_total[0] * 1
                    u_e = u_e_total[0] * 1
                    p_e = p_e_total[0] * 1
                    rho_e = rho_e_total[0] * 1
                    M_e = M_e_total[0] * 1
                    # Xs = Xs_total[0] * 1
                    # r_0 = r_0_total[0] * 1
                    # S = S_total[0] * 1
                    # phi = phi_total[0] * 1
                    phi_d = phi_d_total[0] * 1
                    dS = dS_total[0] * 1
                    Cp_e = Cp_e_total[0] * 1
                    delta_starPhys_BC = delta_starPhys_BC_total[0] * 1
                    ue_BC = ue_BC_total[0] * 1

                    # Recompute Boundary Layer
                    laminarSolution = laminarRegion(atmos, delta, flags[2], potentialSolution, surface)
                    del_d = delta * 1
                    if M_inf > 0.3:
                        # Preston
                        thetapl_in_1, Hpl_in_1 = calc_laminar_parameters(atmos, del_d, 1, potentialSolution, surface)
                        # Michel
                        thetapl_in_2, Hpl_in_2 = calc_laminar_parameters(atmos, del_d, 2, potentialSolution, surface)
                        weight_trans = 0
                        thetapl_in = weight_trans * thetapl_in_1 + (1 - weight_trans) * thetapl_in_2
                        Hpl_in = weight_trans * Hpl_in_1 + (1 - weight_trans) * Hpl_in_2
                    else:
                        thetapl_in, Hpl_in = calc_laminar_parameters(atmos, del_d, 1, potentialSolution,
                                                                     surface)

                    if flags[1] == 0:
                        Theta, H, delta, C_f, n, delta_starPhys, p_s, theta, Delta_star = turbulentPatel(thetapl_in,
                                                                                                         Hpl_in,
                                                                                                         laminarSolution,
                                                                                                         potentialSolution,
                                                                                                         surface, n,
                                                                                                         flags,
                                                                                                         eps,
                                                                                                         end, counter,
                                                                                                         r_f,
                                                                                                         atmos, M_inf,
                                                                                                         phi_d)
                    else:
                        Theta, H, delta, C_f, n, delta_starPhys, p_s, theta, Delta_star = turbulentGreen(thetapl_in,
                                                                                                         Hpl_in,
                                                                                                         laminarSolution,
                                                                                                         potentialSolution,
                                                                                                         surface, n,
                                                                                                         flags,
                                                                                                         eps, end,
                                                                                                         counter,
                                                                                                         r_f, atmos,
                                                                                                         M_inf,
                                                                                                         phi_d)

                    # get input values for displacement effect of fuselage
                    # (displacement effect of nacelle not included yet)
                    delta_total[0] = delta * 1
                    delta_starPhys_total[0] = delta_starPhys * 1
                    delta_starPhys_BC_total[0] = delta_starPhys * 1

                    if flags[0] == 10 and Xn[4] != [] and Xn[5] != []:
                        # rotor inlet
                        x_rot = min(Xn[4])
                        d_rot = max(Yn[4]) - min(Yn[4])
                        u_rot, mdot_rot, Tt_rot, pt_rot, T_rot, p_rot, rho_rot, Ma_rot = rotorAveraged(x_rot, d_rot,
                                                                                                       u_inf,
                                                                                                       p_inf, M_inf,
                                                                                                       delta_total[0],
                                                                                                       u_e_total[0],
                                                                                                       p_e_total[0],
                                                                                                       rho_e_total[0],
                                                                                                       Xs_total[0],
                                                                                                       r_0_total[0],
                                                                                                       phi_total[0], n,
                                                                                                       p_s,
                                                                                                       end, gamma, R, c)
                        # stator outlet
                        A_stat = np.pi * (max(Yn[5]) ** 2 - min(Yn[5]) ** 2)
                        mdot_stat = mdot_rot * 1
                        pt_stat = pt_rot * FPR
                        Tt_stat = Tt_rot * FPR ** ((gamma - 1) / (gamma * etapol_fan))
                        init = T_rot, p_rot, rho_rot, u_rot, Ma_rot
                        thermoSolution = thermoSys(mdot_stat, Tt_stat, pt_stat, A_stat, gamma, R, init)
                        u_stat = thermoSolution[3]

                        # specification of throughflow velocities
                        Fm[4] = np.ones(len(Xn_1[4]) - 1) * u_rot / u_inf
                        Fm[5] = np.ones(len(Xn_1[5]) - 1) * u_stat / u_inf

                    fileit = file + 'BL' + str(counter) + '.txt'
                    path = 'results/%s/' % filename + file + '/' + fileit
                    pathC = 'results/%s/' % filename + file + '/' + 'counter' + '.txt'
                    np.savetxt(path,
                               np.transpose([Theta, H, delta, C_f, n, delta_starPhys, p_s, theta, Delta_star, Cp_e]),
                               delimiter=',', fmt='%s')
                    np.savetxt(pathC, [counter], delimiter=',', fmt='%s')

            boundary_layer = {'Theta': Theta, 'H': H, 'delta': delta, 'delta_starPhys': delta_starPhys,
                              'Delta_star': Delta_star, 'C_f': C_f, 'theta': theta, 'n': n, 'p_s': p_s}

        return Xn, Yn, file, sigma, j_s, j_v, potentialSolution_total, surface_total, atmos, boundary_layer
