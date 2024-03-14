"""Solves the fuselage boundary layer of the PFC using the panel method.

Author:  Anais Habermann

Hypothesis: Geometry is fully axisymmetric (loaded as points on the x-y plane)
            Angle of Attack is zero (free-stream parallel to axis of symmetry)

 Args:
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


import numpy as np
from scipy import interpolate

# Own modules
from panel.solve_potential_flow_pfc_hybrid import PotentialFlow
from panel.integral_boundary_layer.laminar_boundary_layer.laminar_region import laminarRegion
from panel.integral_boundary_layer.laminar_boundary_layer.calculate_transition_parameters import calc_laminar_parameters
from panel.integral_boundary_layer.turbulent_boundary_layer.turbulent_region_patel import turbulentPatel
from panel.integral_boundary_layer.turbulent_boundary_layer.turbulent_region_green import turbulentGreen


class BoundaryLayerCalculation:

    def __init__(self, Xn, Yn, Fm_fuse, arc_length, atmos, flags, counter=0, potentialSolution=None, surface=None,
                 sigma=None, j_s=None, j_v=None, eps=0.0, eps2=0.0, max_it=20):
        self.j_v = j_v
        self.sigma = sigma
        self.j_s = j_s
        self.eps_2 = eps2
        self.eps = eps
        self.arc_length = arc_length
        self.surface = surface
        self.Xn = Xn
        self.Yn = Yn
        self.Fm_fuse = Fm_fuse
        self.atmos = atmos
        self.flags = flags
        self.counter = counter
        self.potentialSolution = potentialSolution
        self.max_it = max_it

    def calculateIBL(self):
        counter = self.counter
        # initialize potential flow solution, if it is not provided
        if self.potentialSolution is None:
            fuselage_panel_pot_init = PotentialFlow(self.Xn, self.Yn, self.Fm_fuse, self.atmos, self.flags,
                                                    self.counter)
            pot_init, surface, sigma, j_s, j_v = fuselage_panel_pot_init.calculate_potential_flow()
        else:
            pot_init = self.potentialSolution
            surface = self.surface
            sigma = self.sigma
            j_s = self.j_s
            j_v = self.j_v

        if self.surface:
            surface = self.surface

        # Set thickness to zero for first iteration
        delta_total = [np.array(0) for _ in range(len(self.Xn))]
        delta_starPhys_total = [[] for _ in range(len(self.Xn))]
        delta_starPhys_BC_total = [[] for _ in range(len(self.Xn))]
        ue_BC_total = [[] for _ in range(len(self.Xn))]
        dS_total = [[] for _ in range(len(self.Xn))]
        for sub in range(len(self.Xn)):
            if self.Xn[sub] != []:
                n_sub = np.shape([self.Xn[sub]])[1]
                delta_starPhys_total[sub] = np.zeros(n_sub)
                delta_starPhys_BC_total[sub] = np.zeros(
                    n_sub - 1)  # displacement thickness, input for transpiration technique
                ue_BC_total[sub] = np.zeros(
                    n_sub - 1)  # velocity at boundary layer edge, input for transpiration technique

        # get input values for boundary layer computation of fuselage

        potentialSolution = pot_init[0]

        u_inf = self.atmos.ext_props['u']
        M_inf = self.atmos.ext_props['mach']
        Xs = surface[0][0]
        phi_d = surface[0][3] * 1  # flow curvature cannot yet be assessed, so we use the body curvature
        r_f = 1  # Temperature recovery factor [-]

        dS_total = self.arc_length[0](Xs)

        # Initialize Variables
        delta = np.zeros(len(Xs))
        n = np.zeros(len(Xs))

        # Since it's a forward marching scheme we compute up to len(Xs) - 1
        end = int(len(Xs) - 1)
        # Compute characteristics in laminar region and immediately downstream of the transition point
        laminarSolution = laminarRegion(self.atmos, delta, self.flags[2], potentialSolution, surface[0])
        transition = laminarSolution[10]
        Xtr = laminarSolution[3]
        L = max(self.Xn[0]) - min(self.Xn[0])
        print("Laminar to turbulent transition according to", transition)
        print("Point of transition x_tr/L:", Xtr / L)
        del_d = delta * 1
        # Compute characteristics in laminar region up to a "natural" transition point
        # (forced transitions, input by user for example, would have unrealistic characteristics for the start of a turbulent flow)
        if M_inf > 0.3:
            thetapl_in, Hpl_in = calc_laminar_parameters(self.atmos, del_d, 2, potentialSolution, surface[0])
        else:
            thetapl_in, Hpl_in = calc_laminar_parameters(self.atmos, del_d, 1, potentialSolution, surface[0])

        # Compute characteristics in turbulent region according to Patel or Green
        if self.flags[1] == 0:
            Theta, H, delta, C_f, n, delta_starPhys, p_s, theta, Delta_star = turbulentPatel(thetapl_in, Hpl_in,
                                                                                             laminarSolution,
                                                                                             potentialSolution,
                                                                                             surface[0],
                                                                                             n, self.flags, self.eps,
                                                                                             end, self.counter, r_f,
                                                                                             self.atmos, M_inf, phi_d)
        else:
            Theta, H, delta, C_f, n, delta_starPhys, p_s, theta, Delta_star = turbulentGreen(thetapl_in, Hpl_in,
                                                                                             laminarSolution,
                                                                                             potentialSolution,
                                                                                             surface[0],
                                                                                             n, self.flags, self.eps,
                                                                                             end, self.counter, r_f,
                                                                                             self.atmos, M_inf, phi_d)

        # get input values for displacement effect of fuselage
        delta_total[0] = delta * 1
        delta_starPhys_total[0] = delta_starPhys * 1
        delta_starPhys_BC_total[0] = delta_starPhys * 1

        if self.flags[3] == 1 or self.flags[3] == 2:
            # Start iteration between viscid and inviscid flow
            delta_starPhys_1 = delta_starPhys * 0
            conv = True
            while np.allclose(delta_starPhys[0:end - 1], delta_starPhys_1[0:end - 1], rtol=self.eps_2) is False and \
                    counter <= self.max_it:
                if counter == 20:
                    conv = False
                    break
                delta_starPhys_1 = np.copy(delta_starPhys)
                counter += 1
                print("Viscid and Inviscid Flow interaction: starting iteration number %2d" % counter)
                # Interpolate displacement thickness to element edges (currently on middle of element)
                dp = interpolate.interp1d(Xs[0:end - 1], delta_starPhys[0:end - 1], fill_value="extrapolate")
                delta_starPhys_total[0] = dp(self.Xn[0])
                delta_starPhys_BC_total[0] = dp(Xs)
                dp2 = interpolate.interp1d(Xs[0:end - 1], delta[0:end - 1], fill_value="extrapolate")
                delta_total[0] = dp2(Xs)
                boundary_layer = []
                boundary_layer.append(delta_starPhys_total)
                boundary_layer.append(delta_total)
                boundary_layer.append(delta_starPhys_BC_total)
                boundary_layer.append(ue_BC_total)
                # Compute potential flow field at edge of boundary layer considering displacement thickness
                fuselage_panel_pot_init = PotentialFlow(self.Xn, self.Yn, self.Fm_fuse, self.atmos, self.flags,
                                                        counter, boundary_layer)
                potentialSolution_total, newSurface_total, sigma, j_s, j_v = fuselage_panel_pot_init.calculate_potential_flow()
                # separate solutions for the different subbodies
                Vx_e_total = [[] for _ in range(len(self.Xn))]
                Vy_e_total = [[] for _ in range(len(self.Xn))]
                u_e_total = [[] for _ in range(len(self.Xn))]
                p_e_total = [[] for _ in range(len(self.Xn))]
                rho_e_total = [[] for _ in range(len(self.Xn))]
                M_e_total = [[] for _ in range(len(self.Xn))]
                phi_d_total = [[] for _ in range(len(self.Xn))]
                # separate solutions for the different subbodies
                Cp_e_total = [[] for _ in range(len(self.Xn))]
                for sub in range(len(self.Xn)):
                    if self.Xn[sub] != []:
                        Vx_e_total[sub] = potentialSolution_total[sub][0] * u_inf
                        Vy_e_total[sub] = potentialSolution_total[sub][1] * u_inf
                        u_e_total[sub] = potentialSolution_total[sub][2] * u_inf
                        p_e_total[sub] = potentialSolution_total[sub][3]
                        rho_e_total[sub] = potentialSolution_total[sub][4]
                        M_e_total[sub] = potentialSolution_total[sub][5]
                        if self.arc_length[sub] != []:  # only for solid bodies
                            phi_d_total[sub] = (np.arctan2(Vy_e_total[sub], Vx_e_total[sub]))
                            ue_BC_total[sub] = u_e_total[sub] * 1
                        # Pressure coefficient for compressible flow
                        Cp_e_total[sub] = (p_e_total[sub] - self.atmos.pressure) / (0.5 * self.atmos.ext_props['rho'] *
                                                                                    self.atmos.ext_props['u'] ** 2)

                # get input values for boundary layer computation of fuselage
                # (boundary layer computation of nacelle not included yet)
                delta = delta_total[0] * 1
                potentialSolution = potentialSolution_total[0]
                phi_d = phi_d_total[0] * 1
                Cp_e = Cp_e_total[0] * 1

                # Recompute Boundary Layer
                laminarSolution = laminarRegion(self.atmos, delta, self.flags[2], potentialSolution, surface[0])
                if M_inf > 0.3:
                    # Preston
                    thetapl_in_1, Hpl_in_1 = calc_laminar_parameters(self.atmos, delta, 1, potentialSolution,
                                                                     surface[0])
                    # Michel
                    thetapl_in_2, Hpl_in_2 = calc_laminar_parameters(self.atmos, delta, 2, potentialSolution,
                                                                     surface[0])
                    weight_trans = self.flags[8]
                    thetapl_in = weight_trans * thetapl_in_1 + (1 - weight_trans) * thetapl_in_2
                    Hpl_in = weight_trans * Hpl_in_1 + (1 - weight_trans) * Hpl_in_2
                else:
                    thetapl_in, Hpl_in = calc_laminar_parameters(self.atmos, delta, 1, potentialSolution,
                                                                 surface[0])

                if self.flags[1] == 0:
                    Theta, H, delta, C_f, n, delta_starPhys, p_s, theta, Delta_star = turbulentPatel(thetapl_in, Hpl_in,
                                                                                                     laminarSolution,
                                                                                                     potentialSolution,
                                                                                                     surface[0], n,
                                                                                                     self.flags,
                                                                                                     self.eps,
                                                                                                     end, counter, r_f,
                                                                                                     self.atmos, M_inf,
                                                                                                     phi_d)
                else:
                    Theta, H, delta, C_f, n, delta_starPhys, p_s, theta, Delta_star = turbulentGreen(thetapl_in, Hpl_in,
                                                                                                     laminarSolution,
                                                                                                     potentialSolution,
                                                                                                     surface[0], n,
                                                                                                     self.flags,
                                                                                                     self.eps, end,
                                                                                                     counter,
                                                                                                     r_f, self.atmos,
                                                                                                     M_inf,
                                                                                                     phi_d)

                # get input values for displacement effect of fuselage
                # (displacement effect of nacelle not included yet)
                delta_total[0] = delta * 1
                delta_starPhys_total[0] = delta_starPhys * 1
                delta_starPhys_BC_total[0] = delta_starPhys * 1

                print(
                    f'BL calc. displacement thickness error {max(np.abs(delta_starPhys[0:end - 1] - delta_starPhys_1[0:end - 1]))}')

        boundary_layer = []
        boundary_layer.append(delta_starPhys_total)
        boundary_layer.append(delta_total)
        boundary_layer.append(delta_starPhys_BC_total)
        boundary_layer.append(ue_BC_total)
        boundary_layer.append(Theta)
        boundary_layer.append(H)
        boundary_layer.append(theta)
        boundary_layer.append(Delta_star)
        boundary_layer.append(n)
        boundary_layer.append(C_f)
        boundary_layer.append(p_s)

        if 'newSurface_total' in locals():
            surface = newSurface_total

        return potentialSolution, surface, sigma, j_s, j_v, boundary_layer, p_s, C_f, Xtr / L, dS_total, conv
