"""Computes the potential solution of external flow around axisymmetric body and corrects for compressibility

Author:  A. Habermann, Carlos E. Ribeiro Santa Cruz Mendoza, Nikolaus Romanow

 Args:
    Xn              [m]     1-D array X-coordinate of geometric profile
    Yn              [m]     1-D array Y-coordinate of geometric profile
    flags           [-]     1-D array Calculation options
    delta_starPhys  [-]     1-D array Displacement thickness
    delta           [-]     1-D array Boundary layer thickness
    file            [-]     Name to save results
    counter         [-]     Number of viscid/inviscid iterations
    gamma           [-]     Specific heat ratio
    Air_prop        [-]     Tuple air properties
    M_inf           [-]     Free stream Mach number

    Fm              [-]     1-D array Dimensionless throughflow velocity (divided by u_inf)
    delta_starPhys_BC   [-] 1-D array Displacement thickness at control point (input for transpiration technique)
    ue_BC               [-] 1-D array Dimensionless edge velocity (divided by u_inf) (input for transpiration technique)

    if flags[0] == 10:      Xn, Yn, delta_starPhys, delta, Fm, delta_starPhys_BC, ue_BC are lists containing one array for every subbody
    sing_type               Type of singularity to be used for lifting bodies. 1 = vortices, 2 = doublets, 3 = vortices+doublets

Returns:
    Vx_e            [-]     Dimensionless X-component of the edge velocity (rectangular coordinates, divided by u_inf)
    Vy_e            [-]     Dimensionless Y-component of the edge velocity (rectangular coordinates, divided by u_inf)
    u_e             [-]     1-D array Dimensionless edge velocity (divided by u_inf)
    p_e             [Pa]    1-D array Static pressure at the edge of the boundary layer
    rho_e           [kg/m^3]   1-D array Density at the edge of the boundary layer
    M_e             [-]     1-D array Mach number at the edge of the boundary layer
    Xs              [m]     1-D array X-coordinate of discretized nodes
    r_0             [m]     1-D array Y-coordinate of discretized nodes (local transverse radius)
    S               [m]     1-D array Segment sizes
    phi             [rad]   1-D array Segment angle w.r.t symmetry axis

    if flags[0] == 10:      all return variables are lists containing one array for every subbody

Sources:
    [3] Hess, J. L. & Smith, A. M.: Calculation of potential flow about arbitrary bodies.
        Progress in Aerospace Sciences 8 (1967), 1-138, ISSN 03760421
    [4] von Karman, T.: Compressibility effects in aerodynamics. Journal of the
        Aeronautical Sciences 8 (1941), 337-356.
    [5] Myring, D. F.: Theoretical Study of Body Drag in Subcritical Axisymmetric
        Flow. 27 (1976), 186-194.

    [-] Haberland, C., Göde, E., & Sauer, G. (1980). Calculation of the Flow Field around Engine-Wing-Configurations. 
        In 12th Congress of the International Council of the Aeronautical Sciences (ICAS 1980), Munich.
    [-] Haberland, C., & Sauer, G. (1986). On the Computation of Wing Lift Interference Caused by High Bypass Engines. 
        In 15th Congress of the International Council of the Aeronautical Sciences (ICAS 1986), London.
"""

# Built-in/Generic Imports
import numpy as np
import math as math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

# Own modules
from panel.potential_flow.find_sources import findSources
from panel.potential_flow.find_sources import findVelocitiesSource
from panel.potential_flow.find_doublets import findDoublets
from panel.potential_flow.find_doublets import findVelocitiesDoublet
from panel.potential_flow.find_vortices import findVortices
from panel.potential_flow.find_vortices import findVelocitiesVortex
from panel.potential_flow.transpiration_velocity import transpirationVelocity


def potentialFlow(Xn, Yn, Fm, flags, file, counter, atmos, filename, bl_characteristics, sing_type=0):
    # Free-stream (Flow with angle of attack not implemented, but described in original article) # Todo: Implement handling of AoA
    V_inf = 1  # dummy free stream velocity (magnitude is not important for the incompressible part)
    rho_inf = atmos.ext_props['rho']  # Density [kg/m³]
    c_inf = atmos.ext_props['sos']  # Speed of sound [m/s]
    T_inf = atmos.temperature  # Static temperature
    p_inf = atmos.pressure  # Static pressure on free-stream [Pa]
    M_inf = atmos.ext_props['mach']
    u_inf = atmos.ext_props['u']  # Free-stream velocity [m/s]
    gamma = atmos.ext_props['gamma']
    p_t_inf = atmos.ext_props['p_t']
    rho_t_inf = atmos.ext_props['rho_t']

    type = len(Xn) * [0]
    body = len(Xn) * [0]
    if flags[0] == 0 or flags[0] == 13:
        for i in range(0, len(Xn)):
            if abs(min(Yn[i])) < 0.001:  # check if streamlined body, then set flag to use source singularities
                type[i] = 0
                body[i] = 'streamline'
            else:
                type[i] = sing_type  # for lifting bodies, choose defined singularity type
                body[i] = 'lifting'
    elif flags[0] == 10:
        type[0] = 0  # sources on fuselage
        type[1] = 3  # doublets/vortices on nacelle bottom
        type[2] = 1  # vortices on nacelle top
        type[3] = 2  # doublets on jet boundary
        type[4] = 2  # doublets on rotor disk
        type[5] = 2  # doublets on stator disk
        type[6] = 2  # doublets on Kutta panels
    elif flags[0] == 11:
        type[0] = 0  # sources on fuselage
        type[1] = 4  # vortices/sources on nacelle bottom
        type[2] = 4  # vortices/sources on nacelle top
    elif flags[0] == 12:  # FD case, nacelle only
        type[0] = 4  # vortices/sources on nacelle bottom
        type[1] = 4  # vortices/sources on nacelle top
        body[0] = 'lifting'
        body[1] = 'lifting'
    elif flags[0] == 14:  # FD case, streamlined body only
        type[0] = sing_type
        body[0] = 'streamline'
    else:
        type[0] = sing_type
        body[0] = 'streamline'

    Yn_d = [[] for _ in range(len(Xn))]
    nn = [0 for _ in range(len(Xn))]  # Number of boundary points
    nseg = [0 for _ in range(len(Xn))]  # Number of segments (panels)
    Xm = [[] for _ in range(len(Xn))]  # Panel control point X-coordinate
    Ym = [[] for _ in range(len(Xn))]  # Panel control point Y-coordinate
    S = [[] for _ in range(len(Xn))]  # Panel length
    phi = [[] for _ in range(len(Xn))]  # panel orientation
    Xs = [[] for _ in range(len(Xn))]
    r_0 = [[] for _ in range(len(Xn))]
    Vx_e = [[] for _ in range(len(Xn))]
    Vy_e = [[] for _ in range(len(Xn))]
    u_ei = [[] for _ in range(len(Xn))]
    delta_starPhys = [[] for _ in range(len(Xn))]
    delta = [[] for _ in range(len(Xn))]
    delta_starPhys_BC = [[] for _ in range(len(Xn))]
    ue_BC = [[] for _ in range(len(Xn))]

    for sub in range(len(Xn)):

        nn[sub] = np.shape([Xn[sub]])[1]  # Number of boundary points
        nseg[sub] = nn[sub] - 1  # Number of segments (panels)

        if bl_characteristics is None:
            delta_starPhys[sub] = np.zeros(nn[sub])
            delta[sub] = np.zeros(nn[sub])
            delta_starPhys_BC[sub] = np.zeros(nseg[sub])  # displacement thickness, input for transpiration technique
            ue_BC[sub] = np.zeros(nseg[sub])  # velocity at boundary layer edge, input for transpiration technique
        else:
            delta_starPhys[sub] = bl_characteristics[0][sub]
            delta[sub] = bl_characteristics[1][sub]
            delta_starPhys_BC[sub] = bl_characteristics[2][sub]
            ue_BC[sub] = bl_characteristics[3][sub]

        if Xn[sub] != []:
            if flags[3] == 2:
                # Transpiration technique (no displaced body)
                Yn_d[sub] = Yn[sub] * 1
            else:
                # Inviscid flow is displaced due to the Boundary Layer
                Yn_d[sub] = Yn[sub] + delta_starPhys[sub]

            # Initialize and fill variables
            Xm[sub] = np.array([0.5 * (Xn[sub][i] + Xn[sub][i + 1]) for i in
                                range(0, nseg[sub])])  # control point on the middle of segment
            Ym[sub] = np.array([0.5 * (Yn_d[sub][i] + Yn_d[sub][i + 1]) for i in range(0, nseg[sub])])
            S[sub] = np.array(
                [((Xn[sub][i + 1] - Xn[sub][i]) ** 2 + (Yn_d[sub][i + 1] - Yn_d[sub][i]) ** 2) ** 0.5 for i in
                 range(0, nseg[sub])])  # Length of the panel
            phi[sub] = np.array([math.atan2(Yn_d[sub][i + 1] - Yn_d[sub][i], Xn[sub][i + 1] - Xn[sub][i]) for i in
                                 range(0, nseg[sub])])  # Angle of panel
            if (flags[0] == 0 and body[sub] == 'lifting') or (flags[0] == 10 and (sub == 1 or sub == 2)) or (
                    flags[0] == 11 and (sub == 1 or sub == 2)):
                phi[sub] = map(lambda x: x - np.pi, phi[sub])
            phi[sub] = np.array(list(phi[sub]))
            phi[sub][phi[sub] < 0] += 2 * np.pi

            if flags[3] == 2 and counter > 0:
                # Transpiration technique (only for solid bodies)
                ue_input = ue_BC[sub] / u_inf
                Fm[sub] = transpirationVelocity(Xm[sub], Ym[sub], ue_input, delta_starPhys_BC[sub], delta[sub],
                                                phi[sub])
            else:
                Fm[sub] = np.zeros(len(Xm[sub]))

            # Input points of interest where to evaluate the velocity (usually edge of Boundary Layer)
            if (flags[0] == 1):
                # Evaluate points as given in Nakayama et. al (1976) to fvmValidation potential flow around Ellipse
                # ONLY FOR POTENTIAL FLOW VALIDATION, NOT FOR BOUNDARY LAYER CALCULATION
                Xs[sub] = np.linspace(0, 2 * max(Xm[sub]), 2 * len(Xm[sub]) - 1)
                Xs[0:len(Xm[sub])] = [list(Xm[sub])]
                q = max(Ym[sub])  # maximum section of the ellipse
                r_0[sub] = q * np.ones(2 * len(Xm[sub]) - 1)  # Stream will contour the ellipse up to max
                r_0[0:int(len(Xm) / 2) - 1] = list(Ym[0:int(len(Xm[sub]) / 2) - 1])  # and then follow the same height
            else:
                # Evaluate velocity field points on the edge of boundary layer (at y = delta)
                Xs[sub] = Xm[sub] * 1
                if counter > 0:
                    r_0[sub] = Ym[sub] + delta[sub]
                else:
                    r_0[sub] = Ym[sub] * 1

            # Initialize velocity arrays on points of interest
            Vx_e[sub] = np.zeros(nseg[sub])
            Vy_e[sub] = np.zeros(nseg[sub])
            u_ei[sub] = np.zeros(nseg[sub])

    # concatenate arrays for coefficient matrices
    Xn_tot = np.concatenate(Xn)
    Ynd_tot = np.concatenate(Yn_d)
    Xm_tot = np.concatenate(Xm)
    Ym_tot = np.concatenate(Ym)
    S_tot = np.concatenate(S)
    phi_tot = np.concatenate(phi)
    Fm_tot = np.concatenate(Fm)

    A = np.zeros([2 * sum(nseg), 2 * sum(nseg)])  # Matrix of influence coefficient
    b = np.zeros(2 * sum(nseg))  # RHS of linear system (known velocity component)

    i_pan_total = range(0, sum(nseg))  # indices of panels where velocity is induced
    i_pan_bodies = len(Xn) * [0]

    if flags[0] < 10 or flags[0] == 12 or flags[0] == 14 or flags[0] == 15:
        i_n = []
        i_t = []
        for i in range(0, len(Xn)):
            if i == 0:
                i_pan_bodies[i] = range(0, nseg[i])
                nseg_old = nseg[0]
            else:
                i_pan_bodies[i] = range(nseg_old, nseg_old + nseg[i])
                nseg_old += nseg[i]

            if type[i] == 0 or type[i] == 2 or type[i] == 3:
                i_n.extend(i_pan_bodies[i])
            if type[i] == 1 or type[i] == 3:
                i_t.extend(i_pan_bodies[i])
            if type[i] == 4:
                i_n.extend(i_pan_bodies[i])
                A_S = np.zeros([2 * sum(nseg), 2 * sum(nseg)])  # Matrix of influence coefficient
                B_S = np.zeros([2 * sum(nseg), 2 * sum(nseg)])  # Matrix of influence coefficient
                A_V = np.zeros([2 * sum(nseg), 2 * sum(nseg)])  # Matrix of influence coefficient
                B_V = np.zeros([2 * sum(nseg), 2 * sum(nseg)])  # Matrix of influence coefficient

    elif flags[0] == 10:
        i_pan_bodies[0] = range(0, nseg[0])  # indices fuselage
        i_pan_bodies[1] = range(nseg[0], nseg[0] + nseg[1])  # indices nacelle bottom
        i_pan_bodies[2] = range(nseg[0] + nseg[1], nseg[0] + nseg[1] + nseg[2])  # indices nacelle top
        i_pan_bodies[3] = range(nseg[0] + nseg[1] + nseg[2], nseg[0] + nseg[1] + nseg[2] + nseg[3])  # indices jet
        i_pan_bodies[4] = range(nseg[0] + nseg[1] + nseg[2] + nseg[3],
                                nseg[0] + nseg[1] + nseg[2] + nseg[3] + nseg[4])  # indices rotor
        i_pan_bodies[5] = range(nseg[0] + nseg[1] + nseg[2] + nseg[3] + nseg[4],
                                nseg[0] + nseg[1] + nseg[2] + nseg[3] + nseg[4] + nseg[5])  # indices stator
        i_pan_bodies[6] = range(nseg[0] + nseg[1] + nseg[2] + nseg[3] + nseg[4] + nseg[5],
                                nseg[0] + nseg[1] + nseg[2] + nseg[3] + nseg[4] + nseg[5] + nseg[6])  # indices rotor
        i_n = [*i_pan_bodies[0], *i_pan_bodies[1], *i_pan_bodies[3], *i_pan_bodies[4], *i_pan_bodies[5],
               *i_pan_bodies[6]]  # indices of panels where normal boundary conditions are fulfilled
        i_t = [*i_pan_bodies[1],
               *i_pan_bodies[2]]  # indices of panels where tangential boundary conditions are fulfilled

    elif flags[0] == 11 or flags[0] == 13:
        i_pan_bodies[0] = range(0, nseg[0])  # indices fuselage
        i_pan_bodies[1] = range(nseg[0], nseg[0] + nseg[1])  # indices nacelle bottom
        i_pan_bodies[2] = range(nseg[0] + nseg[1], nseg[0] + nseg[1] + nseg[2])  # indices nacelle top
        i_n = [*i_pan_bodies[0], *i_pan_bodies[1],
               *i_pan_bodies[2]]  # indices of panels where normal boundary conditions are fulfilled
        i_t = []  # indices of panels where tangential boundary conditions are fulfilled
        A_S = np.zeros([2 * sum(nseg), 2 * sum(nseg)])  # Matrix of influence coefficient
        B_S = np.zeros([2 * sum(nseg), 2 * sum(nseg)])  # Matrix of influence coefficient
        A_V = np.zeros([2 * sum(nseg), 2 * sum(nseg)])  # Matrix of influence coefficient
        B_V = np.zeros([2 * sum(nseg), 2 * sum(nseg)])  # Matrix of influence coefficient

    j_sing = [[] for _ in range(len(Xn))]
    j_sources = []
    j_vortices = []
    j_doublets = []
    S_sigma = 2 * len(S_tot) * [0]

    for sub in range(len(Xn)):
        if Xn[sub] != []:
            if sub == 0:
                j_sing[sub] = range(0, nseg[0])
            else:
                j_sing[sub] = range(max(j_sing[sub - 1]) + 1, max(j_sing[sub - 1]) + 1 + nseg[sub])

            if type[sub] == 0:
                # Obtain coefficients due to sources
                A_subQ, B_subQ = findSources(Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot, i_pan_total, j_sing[sub])
                # The following line might have been required for something, but it is not useful for single
                # streamlined body with sources
                # np.fill_diagonal(A_subQ, 2*np.pi)
                for j in j_sing[sub]:
                    for i in i_n:
                        A[i, j] = A_subQ[i, j]
                        if flags[0] == 11:
                            A_S[i, j] = A_subQ[i, j]
                            B_S[i, j] = B_subQ[i, j]
                    for i in i_t:
                        A[i + sum(nseg), j] = B_subQ[i, j]
                    # No tangential boundary condition for panels without vortex (also RHS is set 0)
                    i = j
                    A[i + sum(nseg), j + sum(nseg)] = 1
                    # Obtain RHS
                    b[j] = V_inf * np.sin(phi_tot[j]) + Fm_tot[j]  # Fm_tot for transpiration velocity
                    # b[j+sum(nseg)] = 0
                Xn_tot = np.delete(Xn_tot, j_sing[sub][
                    -1])  # Xn, Yn has one element more than Xm, Ym -> delete one for consistency at next subbody
                Ynd_tot = np.delete(Ynd_tot, j_sing[sub][-1])
                idx = list(i_pan_bodies[sub])
                j_sources.append(idx)
                S_sigma[min(j_sing[sub]):max(j_sing[sub])] = S[sub]

            elif type[sub] == 1:
                # Obtain coefficients due to vortices
                A_subV, B_subV = findVortices(Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot, i_pan_total, j_sing[sub])
                for j in j_sing[sub]:
                    for i in i_n:
                        A[i, j + sum(nseg)] = A_subV[i, j]
                    for i in i_t:
                        A[i + sum(nseg), j + sum(nseg)] = B_subV[i, j]
                    # No normal boundary condition for panels without source/dipol (also RHS is set 0)
                    i = j
                    A[i, j] = 1  # Obtain RHS
                    b[j + sum(nseg)] = - V_inf * np.cos(phi_tot[j])
                Xn_tot = np.delete(Xn_tot, j_sing[sub][-1])
                Ynd_tot = np.delete(Ynd_tot, j_sing[sub][-1])
                idx = list(i_pan_bodies[sub])
                idx = [x + sum(nseg) for x in idx]
                j_vortices.append(idx)
                B_V = B_subV
                S_sigma[min(j_sing[sub]) + sum(nseg):max(j_sing[sub]) + sum(nseg)] = S[sub]

            elif type[sub] == 2:
                # Obtain coefficients due to doublets
                A_subD, B_subD = findDoublets(Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot, i_pan_total, j_sing[sub])
                for j in j_sing[sub]:
                    for i in i_n:
                        A[i, j] = A_subD[i, j]
                    for i in i_t:
                        A[i + sum(nseg), j] = B_subD[i, j]
                    # No tangential boundary condition for panels without vortex (also RHS is set 0)
                    i = j
                    A[i + sum(nseg), j + sum(nseg)] = 1
                    # Obtain RHS
                    b[j] = V_inf * np.sin(phi_tot[j])
                    # b[j+sum(nseg)] = 0
                Xn_tot = np.delete(Xn_tot, j_sing[sub][-1])
                Ynd_tot = np.delete(Ynd_tot, j_sing[sub][-1])
                idx = list(i_pan_bodies[sub])
                j_doublets.append(idx)
                S_sigma[min(j_sing[sub]):max(j_sing[sub])] = S[sub]

            elif type[sub] == 3:
                # Obtain coefficients due to doublets/vortices
                A_subD, B_subD = findDoublets(Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot, i_pan_total, j_sing[sub])
                A_subV, B_subV = findVortices(Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot, i_pan_total, j_sing[sub])
                for j in j_sing[sub]:
                    for i in i_n:
                        A[i, j] = A_subD[i, j]
                        A[i, j + sum(nseg)] = A_subV[i, j]
                    for i in i_t:
                        A[i + sum(nseg), j] = B_subD[i, j]
                        A[i + sum(nseg), j + sum(nseg)] = B_subV[i, j]
                    # Obtain RHS
                    b[j] = V_inf * np.sin(phi_tot[j]) + Fm_tot[
                        j]  # Fm_tot for transpiration velocity (not implemented for nacelle yet)
                    b[j + sum(nseg)] = - V_inf * np.cos(phi_tot[j])
                Xn_tot = np.delete(Xn_tot, j_sing[sub][-1])
                Ynd_tot = np.delete(Ynd_tot, j_sing[sub][-1])
                idx = list(i_pan_bodies[sub])
                idx = [x + sum(nseg) for x in idx]
                j_doublets.append(list(i_pan_bodies[sub]))
                j_vortices.append(idx)
                S_sigma[min(j_sing[sub]):max(j_sing[sub])] = S[sub]
                S_sigma[min(j_sing[sub]) + sum(nseg):max(j_sing[sub]) + sum(nseg)] = S[sub]

            if type[sub] == 4:
                # Obtain coefficients due to sources
                A_subQ, B_subQ = findSources(Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot, i_pan_total, j_sing[sub])
                np.fill_diagonal(A_subQ, 2 * np.pi)
                np.fill_diagonal(B_subQ, 2 * np.pi)
                # Obtain coefficients due to vortices
                A_subV, B_subV = findVortices(Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot, i_pan_total, j_sing[sub])
                for j in j_sing[sub]:
                    for i in i_n:
                        A[i, j] = A_subQ[i, j]
                        A_S[i, j] = A_subQ[i, j]
                        B_S[i, j] = B_subQ[i, j]
                        A_V[i, j] = A_subV[i, j]
                        B_V[i, j] = B_subV[i, j]
                    # Obtain RHS
                    b[j] = V_inf * np.sin(phi_tot[j])
                    i = j
                    A[i + sum(nseg), j + sum(nseg)] = 1
                    A_S[i, j] = 0
                    A_V[i, j] = 0
                    B_S[i, j] = 0
                    B_V[i, j] = 0
                Xn_tot = np.delete(Xn_tot, j_sing[sub][-1])
                Ynd_tot = np.delete(Ynd_tot, j_sing[sub][-1])
                j_sources.append(list(i_pan_bodies[sub]))
                S_sigma[min(j_sing[sub]):max(j_sing[sub])] = S[sub]
                S_sigma[min(j_sing[sub]) + sum(nseg):max(j_sing[sub]) + sum(nseg)] = S[sub]

    # Kutta condition for vortex method (sum of vortex strengths of first and last panels equals zero)
    if flags[0] == 0 and type[0] == 1 and body[0] == 'lifting':
        A[sum(nseg) + sum(nseg) - 1, :] = 0
        A[sum(nseg) + sum(nseg) - 1, sum(nseg)] = sum(B_V[0, :] + B_V[sum(nseg) - 1, :])
        A[sum(nseg) + sum(nseg) - 1, sum(nseg) + sum(nseg) - 1] = 1
        b[sum(nseg) + sum(nseg) - 1] = 0

    # Setup A matrix for combined Source/Vortex method with constant vortex strength panels
    # Following the approach by Hess and Smith: https://www.youtube.com/watch?v=bc_pkKGEypU ; https://www.youtube.com/watch?v=V77QTAgZuqw
    if (flags[0] == 0 or flags[0] == 12) and type[0] == 4 and body[0] == 'lifting':
        newAV = np.zeros((len(A[0]), 1))
        A = np.hstack((A, newAV))
        for i in range(sum(nseg)):
            A[i, sum(nseg)] = -sum(A_V[i, :])  # constant vortex strength contribution
        newAH = np.zeros((1, len(A[0] + 1)))
        A = np.vstack((A, newAH))
        for j in range(sum(nseg)):  # source contribution of Kutta condition
            A[sum(nseg), j] = B_S[0, j] + B_S[sum(nseg) - 1, j]
        A[sum(nseg), sum(nseg)] = -(
            sum(B_V[0, :] + B_V[sum(nseg) - 1, :])) + 1  # vortex contribution of Kutta condition
        b = np.append(b, 0)
        b[sum(nseg)] = -V_inf * (np.cos(phi_tot[0]) + np.cos(phi_tot[-1]))
        b = np.delete(b, slice(sum(nseg) + 1, 2 * (sum(nseg)) + 1), 0)
        A = np.delete(A, slice(sum(nseg) + 1, 2 * (sum(nseg)) + 1), 0)
        A = np.delete(A, slice(sum(nseg) + 1, 2 * (sum(nseg)) + 1), 1)
        j_vortices.append([len(A[0]) - 1])
    elif flags[0] == 11 or flags[0] == 13 or (
            flags[0] == 0 and len(Xn) == 3 and body[1] == 'lifting' and body[0] == 'streamline'):
        newAV = np.zeros((len(A[0]), 1))
        A = np.hstack((A, newAV))
        for i in range(sum(nseg)):
            A[i, sum(nseg)] = -sum(A_V[i, nseg[0]:sum(nseg) - 1])  # constant vortex strength contribution
        newAH = np.zeros((1, len(A[0] + 1)))
        A = np.vstack((A, newAH))
        for j in range(sum(nseg)):  # source contribution of Kutta condition
            A[sum(nseg), j] = B_S[nseg[0], j] + B_S[sum(nseg) - 1, j]  #
        A[sum(nseg), sum(nseg)] = -(
            sum(B_V[nseg[0], :] + B_V[sum(nseg) - 1, :])) + 1  # vortex contribution of Kutta condition
        b = np.append(b, 0)
        b[sum(nseg)] = -V_inf * (np.cos(phi[1][0]) + np.cos(phi[2][-1]))
        b = np.delete(b, slice(sum(nseg) + 1, 2 * (sum(nseg)) + 1), 0)
        A = np.delete(A, slice(sum(nseg) + 1, 2 * (sum(nseg)) + 1), 0)
        A = np.delete(A, slice(sum(nseg) + 1, 2 * (sum(nseg)) + 1), 1)
        j_vortices.append([len(A[0]) - 1])

    # Compute singularity strengths that yields zero normal/tangential velocity to surface
    j_sources = [y for x in j_sources for y in x]
    j_vortices = [y for x in j_vortices for y in x]
    j_doublets = [y for x in j_doublets for y in x]
    S_sigma = np.array(S_sigma)
    sigma = np.linalg.solve(A, b)  # Compute all singularity strength values
    Sigma = sigma[j_sources]  # source strength values
    My = sigma[j_doublets]  # doublet strength values
    Gamma = sigma[j_vortices]  # vortex strength values

    if Sigma != []:
        sumSigma = sum(Sigma * S_sigma[j_sources])  # Check sum of source panel strengths (MUST BE CLOSE TO ZERO)
        print("Potential Flow - sum of sources (must be close to zero): ",
              sumSigma)  # Print sum of all source strengths
        if np.abs(sumSigma) > 0.2:
            # if np.abs(sumSigma) > 0.2 and np.abs(sumMy) > 0.2 and np.abs(sumGamma) > 0.2:
            print("Potential flow computation is generating/removing mass, please check geometry/mesh.")
            print(
                "(points too close to one another with alternating slopes may cause bad behaviour, see Troubleshooting file)")
    if My != []:
        sumMy = sum(My * S_sigma[j_doublets])  # Check sum of source panel strengths
        print("Potential Flow - sum of doublets: ", sumMy)
    if Gamma != []:
        sumGamma = sum(Gamma * S_sigma[j_vortices])  # Check sum of vortex panel strengths
        print("Potential Flow - sum of vortices: ", sumGamma)

    Xn_tot = np.concatenate(Xn)  # calculate new since last entities have been deleted for every subbody
    Ynd_tot = np.concatenate(Yn_d)  # calculate new since last entities have been deleted for every subbody
    Xs_tot = np.concatenate(Xs)
    r0_tot = np.concatenate(r_0)

    # Obtain velocity due to free-stream and singularities on selected points
    if flags[0] == 11 or (
            (flags[0] == 0 or flags[0] == 13) and len(Xn) == 3 and body[1] == 'lifting' and body[0] == 'streamline'):
        Wx = np.zeros((sum(nseg), sum(nseg) + nseg[1] + nseg[2]))
        Wy = np.zeros((sum(nseg), sum(nseg) + nseg[1] + nseg[2]))
    else:
        Wx = np.zeros((sum(nseg), 2 * sum(nseg)))
        Wy = np.zeros((sum(nseg), 2 * sum(nseg)))

    # Velocities induced by panel j on selected point i
    # (if panel j has no singularity, this is already taken into account by sigma[j]=0)

    for sub in range(len(Xn)):
        if Xn[sub] != []:
            if type[sub] == 0:
                # Velocities induced by sources
                Wx_subQ, Wy_subQ = findVelocitiesSource(Xs_tot, r0_tot, Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot,
                                                        i_pan_total, j_sing[sub])
                for j in j_sing[sub]:
                    for i in i_pan_total:
                        Wx[i, j] = Wx_subQ[i, j]
                        Wy[i, j] = Wy_subQ[i, j]
                Xn_tot = np.delete(Xn_tot, j_sing[sub][
                    -1])  # Xn, Yn ist one element more than Xm, Ym -> delete one for consistency at next subbody
                Ynd_tot = np.delete(Ynd_tot, j_sing[sub][-1])

            elif type[sub] == 1:
                # Velocities induced by vortices
                Wx_subV, Wy_subV = findVelocitiesVortex(Xs_tot, r0_tot, Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot,
                                                        i_pan_total, j_sing[sub])
                for j in j_sing[sub]:
                    for i in i_pan_total:
                        Wx[i, j + sum(nseg)] = Wx_subV[i, j]
                        Wy[i, j + sum(nseg)] = Wy_subV[i, j]
                Xn_tot = np.delete(Xn_tot, j_sing[sub][-1])
                Ynd_tot = np.delete(Ynd_tot, j_sing[sub][-1])

            elif type[sub] == 2:
                # Velocities induced by doublets
                Wx_subD, Wy_subD = findVelocitiesDoublet(Xs_tot, r0_tot, Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot,
                                                         S_tot,
                                                         i_pan_total, j_sing[sub])
                for j in j_sing[sub]:
                    for i in i_pan_total:
                        Wx[i, j] = Wx_subD[i, j]
                        Wy[i, j] = Wy_subD[i, j]
                Xn_tot = np.delete(Xn_tot, j_sing[sub][-1])
                Ynd_tot = np.delete(Ynd_tot, j_sing[sub][-1])

            elif type[sub] == 3:
                # Velocities induced by doublets/vortices
                Wx_subD, Wy_subD = findVelocitiesDoublet(Xs_tot, r0_tot, Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot,
                                                         S_tot,
                                                         i_pan_total, j_sing[sub])
                Wx_subV, Wy_subV = findVelocitiesVortex(Xs_tot, r0_tot, Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot,
                                                        i_pan_total, j_sing[sub])
                for j in j_sing[sub]:
                    for i in i_pan_total:
                        Wx[i, j] = Wx_subD[i, j]
                        Wy[i, j] = Wy_subD[i, j]
                        Wx[i, j + sum(nseg)] = Wx_subV[i, j]
                        Wy[i, j + sum(nseg)] = Wy_subV[i, j]
                Xn_tot = np.delete(Xn_tot, j_sing[sub][-1])
                Ynd_tot = np.delete(Ynd_tot, j_sing[sub][-1])

            if type[sub] == 4:
                # Velocities induced by sources/vortices
                Wx_subQ, Wy_subQ = findVelocitiesSource(Xs_tot, r0_tot, Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot,
                                                        i_pan_total, j_sing[sub])
                Wx_subV, Wy_subV = findVelocitiesVortex(Xs_tot, r0_tot, Xm_tot, Ym_tot, Xn_tot, Ynd_tot, phi_tot, S_tot,
                                                        i_pan_total, j_sing[sub])
                for j in j_sing[sub]:
                    for i in i_pan_total:
                        Wx[i, j] = Wx_subQ[i, j]
                        Wy[i, j] = Wy_subQ[i, j]
                        if flags[0] == 0 and len(Xn) == 2:
                            Wx[i, j + sum(nseg)] = Wx_subV[i, j]
                            Wy[i, j + sum(nseg)] = Wy_subV[i, j]
                        if flags[0] == 11 or (flags[0] == 0 and len(Xn) == 3):
                            Wx[i, j + nseg[1] + nseg[2]] = Wx_subV[i, j]
                            Wy[i, j + nseg[1] + nseg[2]] = Wy_subV[i, j]
                Xn_tot = np.delete(Xn_tot, j_sing[sub][-1])
                Ynd_tot = np.delete(Ynd_tot, j_sing[sub][-1])

    if (flags[0] == 0 or flags[0] == 12) and type[0] == 4:
        sigma = np.concatenate((Sigma, np.full(len(Sigma), -Gamma[0])))
    elif flags[0] == 11 or (
            (flags[0] == 0 or flags[0] == 13) and len(Xn) == 3 and body[1] == 'lifting' and body[0] == 'streamline'):
        sigma = np.concatenate((Sigma, np.full(nseg[1] + nseg[2], -Gamma[0])))

    Vx_e = V_inf + np.matmul(Wx, sigma)  # x-component in rectangular coordinate system
    Vy_e = np.matmul(Wy, sigma)  # y-component in rectangular coordinate system

    k = 0
    xcomp = [[] for _ in range(len(Xn))]
    ycomp = [[] for _ in range(len(Xn))]
    for sub in range(len(Xn)):
        if ((flags[0] == 10 or flags[0] == 11) and sub == 2) or (
                flags[0] == 0 and body[sub] == 'lifting' and sub == 1 and len(Xn) == 2) or (
                flags[0] == 0 and body[sub] == 'lifting' and sub == 2 and len(Xn) == 3):
            u_ei[sub] = np.cos(phi[sub] + np.pi) * Vx_e[k:k + len(Xn[sub]) - 1] + np.sin(phi[sub] + np.pi) * Vy_e[
                                                                                                             k:k + len(
                                                                                                                 Xn[
                                                                                                                     sub]) - 1]  # transform from cartesian to curvilinear coordinate system
        else:
            u_ei[sub] = -(np.cos(phi[sub] + np.pi) * Vx_e[k:k + len(Xn[sub]) - 1] + np.sin(phi[sub] + np.pi) * Vy_e[
                                                                                                               k:k + len(
                                                                                                                   Xn[
                                                                                                                       sub]) - 1])  # transform from cartesian to curvilinear coordinate system
        xcomp[sub] = Vx_e[k:k + len(Xn[sub]) - 1] / u_ei[sub]
        ycomp[sub] = Vy_e[k:k + len(Xn[sub]) - 1] / u_ei[sub]
        k += len(Xn[sub]) - 1
    p_e = np.zeros(sum(nseg))
    rho_e = rho_inf * np.ones(sum(nseg))
    M_e = np.zeros(sum(nseg))
    xcomp = np.asarray([y for x in xcomp for y in x])
    ycomp = np.asarray([y for x in ycomp for y in x])
    u_ei = np.asarray([y for x in u_ei for y in x])

    if flags[0] > 0 and flags[0] < 10:
        # Plot potential flow results at streamline defined in validation literature, only for validation
        if flags[0] == 1:
            U_num = np.loadtxt("validation_data/g1Spheroid/SpheroidUeStream.txt", delimiter=',')
            plt.rcParams['mathtext.fontset'] = 'stix'
            plt.rcParams['font.family'] = 'STIXGeneral'
            plt.rc('font', size=14)
            plt.rcParams['lines.linewidth'] = 1.5
            plt.rcParams['axes.formatter.limits'] = '-2, 2'
            plt.rcParams['axes.formatter.use_mathtext'] = 'True'
            plt.rcParams['axes.formatter.useoffset'] = 'False'
            plt.rcParams['axes.labelsize'] = 22
            # BHL colors: orange, dark blue, green, light blue, brown, bright blue, yellow
            clr = ['#D7801C', '#0A3A5A', '#65883E', '#2882BB', '#CA4F1B', '#6AB9EC', '#F5D40E']
            lnst = ['solid', 'dashed', 'dashdot', 'dotted', 'dashed', 'dashed']
            dsh = ['', [6, 1], [6, 2, 2, 2], [2, 2], [6, 2, 2, 2, 2, 2], [6, 4]]
            figU, axU = plt.subplots(constrained_layout=True)
            axU.plot(U_num[:, 0], U_num[:, 1], '--', color='black', linewidth=2)
            axU.plot(Xm[0] / Xn[0][-1], u_ei, color=clr[0], linestyle=lnst[0], dashes=dsh[0])
            axU.set(xlabel='$x/L$', ylabel='$u_e / u_\infty$')
            ax2 = axU.twinx()
            ax2.set_ylabel('$r_0/L$')
            ax2.plot(Xn[0] / Xn[0][-1], Yn_d[0] / Xn[0][-1], linestyle='solid', color='black', linewidth=2,
                     label='Body')
            ax2.fill_between(Xn[0] / Xn[0][-1], 0, Yn_d[0] / Xn[0][-1], facecolor='none', edgecolor='gray',
                             hatch="////")
            ax2.set_ylim([0, 0.6])
            # axU.set_ylim([0.75, 1.1])
            plt.savefig('results/Spheroid_6_1/g1/U_e.png')
            # plt.show()

        p_e = np.zeros(len(Xs[0]))
        rho_e = rho_inf * np.ones(len(Xs[0]))
        M_e = np.zeros(len(Xs[0]))
        xcomp = Vx_e / u_ei
        ycomp = Vy_e / u_ei

    Cp_crit = (2 / (gamma * M_inf ** 2)) * (
                ((1 + ((gamma - 1) / 2 * M_inf ** 2)) / (1 + (gamma - 1) / 2)) ** (gamma / (gamma - 1)) - 1)

    # Compressibility corrections
    if flags[4] == 0:  # No correction (Incompressible)
        u_e = u_ei * u_inf  # for incompressible
        p_e = p_t_inf * (1 + 0.5 * (gamma - 1) * (u_e / c_inf) ** 2) ** (-gamma / (gamma - 1))  # for incompressible
        M_e = u_e / c_inf  # Edge Mach number
        u_e = u_e / u_inf  # dimensionless Edge velocity
        Cp_e = (p_e - p_inf) / (0.5 * rho_inf * u_inf ** 2)
    elif flags[4] == 1:  # Karman-Tsien correction as implemented in Xfoil
        u_ei *= u_inf
        p_ei = p_t_inf * (1 + 0.5 * (gamma - 1) * (u_ei / c_inf) ** 2) ** (-gamma / (gamma - 1))
        Cp_i = (p_ei - p_inf) / (0.5 * rho_inf * u_inf ** 2)
        if np.any(Cp_i < Cp_crit):
            warnings.warn("Freestream Mach number bigger than critical Mach number. Compressibility correction will "
                          "not be correct.")
        beta = np.sqrt(1 - M_inf ** 2)
        f = M_inf ** 2 / (1 + beta) ** 2
        Cp_e = np.array([i / (beta + f * (1 + beta) * i / 2) for i in Cp_i])
        u_e = u_ei * (1 - f) / (1 - f * (u_ei / u_inf) ** 2)
        # constant stagnation enthalpy
        h_0 = u_inf ** 2 / ((gamma - 1) * M_inf ** 2) * (1 + (gamma - 1) / 2 * M_inf ** 2)
        # local speed of sound
        c_loc = np.sqrt((gamma - 1) * (h_0 - u_e ** 2 / 2))
        M_e = u_e / c_loc
        p_e = (0.5 * rho_inf * u_inf ** 2) * Cp_e + p_inf
        for i in range(0, len(Xn_tot)):
            if M_e[i] > 1.5:
                raise Exception(f"Sonic region detected on body at location x={Xn_tot[i]}. "
                                f"PM calculation terminated.", Xn_tot[i])
            else:
                pass
        u_e /= u_inf  # dimensionless Edge velocity
    elif flags[4] == 3:  # Dietrich Correction (NOT FUNCTIONING PROPERLY)
        u_e, p_e, rho_e = compDietrichCorrection(gamma, M_inf, u_inf, u_ei * u_inf, c_inf, p_t_inf, p_e, rho_inf,
                                                 rho_t_inf, p_inf, T_inf)
        Vx_e = u_e * xcomp / u_inf
        Vy_e = u_e * ycomp / u_inf
        M_e = u_e / c_inf  # Edge Mach number
        u_e = u_e / u_inf  # Edge velocity magnitude
        Cp_e = (p_e - p_inf) / (0.5 * rho_inf * u_inf ** 2)
    elif flags[4] == 2:  # Myrian Correction
        u_e = u_inf * (u_ei - (u_ei * M_inf) ** 2) / (1 - (u_ei * M_inf) ** 2)  # for ESDU
        p_e = p_t_inf * (1 + 0.5 * (gamma - 1) * (u_e / c_inf) ** 2) ** (-gamma / (gamma - 1))  # for incompressible
        M_e = u_e / c_inf  # Edge Mach number
        rho_e = rho_t_inf * (1 + 0.5 * (gamma - 1) * M_e ** 2) ** (-1 / (gamma - 1))
        Vx_e = u_e * xcomp / u_inf
        Vy_e = u_e * ycomp / u_inf
        u_e = u_e / u_inf  # dimensionless Edge velocity
        Cp_e = (p_e - p_inf) / (0.5 * rho_inf * u_inf ** 2)
    elif flags[4] == 4:  # Laitone Correction
        u_e = u_ei * u_inf
        p_e = p_t_inf * (1 + 0.5 * (gamma - 1) * (u_e / c_inf) ** 2) ** (-gamma / (gamma - 1))
        Cp_i = (p_e - p_inf) / (0.5 * rho_inf * u_inf ** 2)
        if np.any(Cp_i < Cp_crit):
            warnings.warn("Freestream Mach number bigger than critical Mach number. Compressibility correction will "
                          "not be correct.")
        M_i = u_e / c_inf
        Cp_e = np.zeros(len(Xn_tot))
        for i in range(0, len(Xn_tot)):
            if M_i[i] < 1:
                Cp_e[i] = Cp_i[i] / (
                            (1 - M_inf ** 2) ** 0.5 + (M_inf ** 2) / ((1 - M_inf ** 2) ** 0.5) * (Cp_i[i] / 2) * (
                            1 + ((gamma - 1) / 2) * M_inf ** 2))
            else:
                Cp_e[i] = Cp_i[i] / (
                    np.abs(1 - M_i[i] ** 2)) ** 0.5  # Prandtl-Glauert transformation, valid for 0.7<Ma_freestream<1.3
        p_e = (0.5 * rho_inf * u_inf ** 2) * Cp_e + p_inf
        u_e = c_inf * ((2 / (gamma - 1)) * ((p_e / p_t_inf) ** (-(gamma - 1) / gamma) - 1)) ** 0.5  # for Karman-Tsien
        Vx_e = u_e * xcomp / u_inf
        Vy_e = u_e * ycomp / u_inf
        M_e = u_e / c_inf
        rho_e = rho_t_inf * (1 + 0.5 * (gamma - 1) * M_e ** 2) ** (-1 / (gamma - 1))
        u_e = u_e / u_inf  # dimensionless Edge velocity

    fileit = file + 'Pot' + str(counter) + '.txt'
    path = 'results/%s/' % filename + file + '/' + fileit

    if flags[0] == 1 and counter == 0:
        line = np.array([1111111111, 0, 0, 0, 0, 0, 0])
        result = line
        result = np.vstack((result, np.transpose(
            [Vx_e[i_pan_bodies[sub]], Vy_e[i_pan_bodies[sub]], u_e[i_pan_bodies[sub]], Xs_tot[i_pan_bodies[sub]],
             r0_tot[i_pan_bodies[sub]], Xm_tot[i_pan_bodies[sub]], Ym_tot[i_pan_bodies[sub]]])))
        result = np.vstack((result, line))
    else:
        line = np.array([1111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        result = line
        for sub in range(len(Xn)):
            result = np.vstack((result, np.transpose(
                [Vx_e[i_pan_bodies[sub]], Vy_e[i_pan_bodies[sub]], u_e[i_pan_bodies[sub]], Xs_tot[i_pan_bodies[sub]],
                 r0_tot[i_pan_bodies[sub]], S_tot[i_pan_bodies[sub]],
                 phi_tot[i_pan_bodies[sub]], p_e[i_pan_bodies[sub]], rho_e[i_pan_bodies[sub]], M_e[i_pan_bodies[sub]],
                 Xm_tot[i_pan_bodies[sub]], Ym_tot[i_pan_bodies[sub]]])))
            result = np.vstack((result, line))

    # np.savetxt(path, result, delimiter=',', fmt='%.12f')

    # if len(Xn) > 1:
    # separate solutions for the different bodies
    potentialSolution = [[] for _ in range(len(Xn))]
    surface = [[] for _ in range(len(Xn))]
    for sub in range(len(Xn)):
        if Xn[sub] != []:
            potentialSolution[sub] = Vx_e[i_pan_bodies[sub]], Vy_e[i_pan_bodies[sub]], u_e[i_pan_bodies[sub]], p_e[
                i_pan_bodies[sub]], rho_e[i_pan_bodies[sub]], M_e[
                                         i_pan_bodies[sub]], Cp_e[i_pan_bodies[sub]]
            surface[sub] = Xs_tot[i_pan_bodies[sub]], r0_tot[i_pan_bodies[sub]], S_tot[i_pan_bodies[sub]], phi_tot[
                i_pan_bodies[sub]], Xn[sub], Yn[sub]
    # else:
    #     potential_flow = Vx_e, Vy_e, u_e, p_e, rho_e, M_e
    #     surface = Xs, r_0, S, phi

    print("Potential Flow computation finished succesfully")

    return potentialSolution, surface, sigma, j_sources, j_vortices


def compDietrichCorrection(gamma, M_inf, u_inf, u_ei, c, p_t, p_e, rho_inf, rho_t, p_inf, T_inf):
    v_cstar = c * (2 / (gamma + 1)) ** 0.5
    ratio = np.zeros(len(u_ei))  # density ratio rho_inf/rho_t
    u_eibar = np.zeros(len(u_ei))
    rho_e = np.zeros(len(u_ei))
    rho_ei = rho_t * (1 + 0.5 * (gamma - 1) * (u_ei / c) ** 2) ** (-1 / (gamma - 1))
    U_c = np.zeros(len(u_ei))
    ratio2 = np.zeros(len(u_ei))  # density ratio rho_cbar/rho_t
    for i in range(0, len(u_ei)):
        ratio[i] = (1 + ((gamma - 1) / 2) * (u_inf / c) ** 2) ** (-1 / (gamma - 1))
        u_eibar[i] = (ratio[i]) * u_inf * (rho_inf / rho_t) + (1 - ratio[i]) * u_ei[
            i]  # *(u_ei[i]/u_inf-1)/((u_ei[i]/u_inf)-M_inf**2)
        if u_eibar[i] > v_cstar:
            inp = (1, 1, gamma, rho_t)
            ratio2[i] = fsolve(avCompressibleDensityRatio, 0.5, args=inp, xtol=1e-10, factor=0.1, maxfev=1000)
            U_c[i] = u_ei[i] * ((rho_ei[i] / (ratio2[i] * rho_t)) ** (u_ei[i] / u_eibar[i]))
        else:
            inp = (u_eibar[i], v_cstar, gamma, rho_t)
            ratio2[i] = fsolve(avCompressibleDensityRatio, 0.5, args=inp, xtol=1e-10, factor=0.1, maxfev=1000)
            U_c[i] = u_ei[i] * ((rho_ei[i] / (ratio2[i] * rho_t)) ** (u_ei[i] / u_eibar[i]))
        # U_c[i] = u_ei[i] * (rho_t/rho_inf) ** (u_ei[i] / u_eibar[i]) # alternative by nasa paper
        p_e[i] = p_t * (1 + 0.5 * (gamma - 1) * (U_c[i] / c) ** 2) ** (-gamma / (gamma - 1))
        rho_e[i] = rho_t * (1 + 0.5 * (gamma - 1) * (U_c[i] / c) ** 2) ** (-1 / (gamma - 1))
    return U_c, p_e, rho_e


def avCompressibleDensityRatio(ratio2, *inp):
    u_eibar, vc, gamma, rho_t = inp
    f1 = u_eibar / vc - (ratio2) * (((gamma + 1) / (gamma - 1)) * (1 - (ratio2) ** (gamma - 1))) ** 0.5
    return f1
