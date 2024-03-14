"""Solves the potential flow around the fuselage of the PFC using the panel method.

Author:  Anais Habermann

Hypothesis: Geometry is fully axisymmetric (loaded as points on the x-y plane)
            Angle of Attack is zero (free-stream parallel to axis of symmetry)

 Args:
    M_inf           [-]     Freestream Mach number
    Alt:            [m]     Altitude from sea level
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


from panel.potential_flow.calculate_potential_flow import potentialFlow


class PotentialFlow:

    def __init__(self, Xn, Yn, Fm, atmos, flags, counter, bl_characteristics=None):
        self.Fm = Fm
        self.bl_characteristics = bl_characteristics
        self.counter = counter
        self.flags = flags
        self.atmos = atmos
        self.Yn = Yn
        self.Xn = Xn

    def calculate_potential_flow(self):
        # Solve potential flow (incl. compressibility correction)
        # for the first iteration, the variable surface contains the body's geometrical data
        potentialSolution, surface, sigma, j_s, j_v = potentialFlow(self.Xn, self.Yn, self.Fm, self.flags, 'pfc',
                                                                    self.counter, self.atmos, 'pfc',
                                                                    self.bl_characteristics)

        return potentialSolution, surface, sigma, j_s, j_v
