"""Main program for calculation and plotting of boundary layer development around axisymmetric or elliptic body
    (main program to insert input data)
     
        - Calculation in 'calculateBoundaryLayer.py'
        - Plotting in 'plot_boundary_layer.py'

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, Nikolaus Romanow, A. Habermann

"""

# Built-in/Generic Imports
import time
import numpy as np
from scipy.special import ellipe

# Own modules
from panel.solve_panel_method import CalculateBL


class Main:
    """
    The Main object to launch/run the panel method.

    Author: Anais Habermann

        Args:
            filename:   If calc_opt is '0' or '2' (boundary layer is calculated): Name of geometry txt file and name of
                        case, e.g.
                        ['bodyAkron', 'Akron'] (without file extension)
                        If calc_opt is '1' (only plotting): Name of (several) result files that are to be plotted, and
                        corresponding labels e.g.
                        ['body01Tr00LD05Re7M00_delta', 'Case1', '\results\body01Tr00LD05Re7M02_delta', 'Case2']
                        (without file extensions)
            N:          Discretization, number of nodes to distribute along body profile [N_fuselage, N_nacelle, N_jet,
                        N_rotor, N_stator, N_kutta]. If subbody should not be calculated, set N[i] = 0.
                        Maximum value for N_kutta = 1
            plot_opts:  Which plots should be plotted. '1': plot, '0': don't plot. [H (Shape factor), delta
                        (BL thickness), delta_star (displacement thickness), Theta (mom. deficit area), u_e
                        (edge velocity), M_e (edge Mach number), theta (mom. thickness), r_e (location of BL edge),
                        C_p (edge pressure distribution), Uprof (velocity profile at 'pos'), Delta_star (displacement
                        area), C_f (skin friction coefficient), p_t2 (total pressure contour entering fan), RI (radial
                        distortion intensity)]
                        plot[14] - option to plot streamlines and pressure contours of inviscid solution. Takes a long time.
            plot_geom_opts: Which geometries should be plotted. '0': none, '1': last, '2': all geometries.
                        [fuselage, nacelle]
            visc:       Option for viscous flow calculation
                        0: potential flow calculation only
                        1: viscous flow calculation
            calc_opt:   Code options -
                        0: only BL calculation
                        1: only plotting / post-processing (input for "geometry" required)
                        2: BL calculation and plotting / post-processing
            alt:        Flight altitude [m]
            Mach:       Freestream Mach number [-] Upper limit: Ma=1.0
            discret:    Discretization strategy
                        0: equidistant sample points
                        1: parametrised sample points (input value for w required)
            weight:     Discretization, weighting parameter for parametrisation of sample points: 0 <= weight < 1.
                        Default: 0.15
                        0: only arc-length parametrisation
                        1: only curvature parametrisation
            interact:   Viscous-inviscid interaction model
                        0: No viscid & inviscid interaction (recommended for low-drag bodies such as projectiles)
                        1: Equivalent body
                        2: Transpiration velocity (recommended for multibody configurations) *default*
            ell:        True: Elliptical cross-section of fuselage
                        False: Axisymmetric fuselage cross-section
            AR_ell:     Aspect ratio of elliptic cross-section of fuselage [-]
            ell_method:  Method to calculate results for elliptic fuselage
                        0: Equivalent radius (only one meridional section calculated)
                        1: Interpolation (meridional sections at semi-major and semi-minor axis calculated)
            fan:        Boolean. Include calculation of fuselage fan
            FPR:        Fan pressure ratio (if fuselage fan is included) [-]
            eta_pol:    Polytropic fan efficiency (if fuselage fan is included) [-]
            turb:       Turbulence model
                        0: Nakayama & Patel (thick BL) *default*
                        1: Green
            trans:      Transition model
                        0 < (Xtr/L) < 1: Transition point input by user, any value between 0 and 1 (there MUST be a
                        laminar region for initial values)
                        1: Preston 1958 - Re_theta > 320 *default*
                        2: Michel-e9 as in Parsons 1972
                        3: Crabtree 1958
            compr:      Compressibility correction model:
                        0: No Compressibility correction of potential flow
                        1: Kármán-Tsien *default*
                        2: ESDU/Myring 1976 (Really bad for Mach > 0.7)
                        3: Dietrich correction
                        4: Laitone correction
            case:       # 0 - Arbitrary Axisymmetric Geometry loaded with text file (Alt=10668, Ma=0.82)
                              Differentiate between geometries with 1 to n bodies
                        # 1 - 6:1 Ellipsoid (streamline as in Nakayama 1976) ONLY FOR POTENTIAL FLOW VALIDATION
                        # 2 - 8:1 Ellipsoid (streamline following body's surface) for potential flow validation
                        # 3 - Fuselage-like body of revolution for potential flow validation (Lewis 1991)
                        # 4 - Airship Akron for validation (Nakayama 1973, Patel 1974) use Ma = 0.15, alt = 2000, Xtr = 0.06
                        # 5 - Modified 6:1 Spheroid for validation (Patel 1974) use Ma = 0.0365, alt = 1000, Xtr = 0.05
                        # 6 - F-57 Low-Drag Body for validation (Patel 1979) use Ma = 0.045, alt = 500, Xtr = 0.475
                        # 7 - ESDU Body 1 from reports 77028g and 77020a
                        # 8 - Waisted body of revolution (Winter et al 1970) use Ma = 0.597, alt = 7680, Xtr = 0.05
                        # 9 - Fuselage equivalent body (NASA 1969) use Ma = 0.75, alt = 10100 or 3000, Xtr = 0.05
                        # 10 - Propulsive Fuselage generated by GenerateFuselage.py.
                        #                   Axisymmetric: AR_ell = 1
                        #                   Elliptic: AR_ell not 1, pos and nac_h must be equal for index 0 and 1
                        # 11 - Propulsive Fuselage generated by GenerateFuselage.py. Fuselage and nacelle only.
                        #      Mixed Source/Vortex method for nacelle
                        #                         Axisymmetric: AR_ell = 1
                        #                         Elliptic: AR_ell not 1, pos and nac_h must be equal for index 0 and 1
            sing_type:  Which singularity type shall be used for lifting bodies (for streamline bodies, sources will be
                        used automatically):
                        0: sources (default)
                        1: vortices
                        2: doublets
                        3: vortices+doublets
                        4: vortices+sources
    """

    def __init__(self, filename: list, N: list, plot_opts: list, plot_geom_opts: list, visc=1, calc_opt=2, alt=0,
                 Mach=0,
                 discret=1, weight=0.15, interact=2, ell='False', AR_ell=1, ell_method=1, fan='false', FPR=1.0,
                 eta_pol=1.0, turb=0, trans=1, compr=1, case=10, sing_type=0):
        self.filename = filename
        self.no = N
        self.plot_opts = plot_opts
        self.plot_geom_opts = plot_geom_opts
        self.option = calc_opt
        self.altitude_inf = alt
        self.Ma_inf = Mach
        self.discret = discret
        self.weight = weight
        self.interact = interact
        self.ell = ell
        self.AR_ell = AR_ell
        self.ell_method = ell_method
        self.fan = fan
        self.FPR = FPR
        self.eta_pol = eta_pol
        self.turb_model = turb
        self.trans_model = trans
        self.compr = compr
        self.case = case
        self.visc = visc
        self.sing_type = sing_type

    def run(self):
        """
        Function to run axiBL

        Author: Anais Habermann

        """

        # todo: add PFC geometry parameters to arguments (GenerateFuselage.py and GenerateGeom.py might need to be adapted!)
        # todo: add pos and nac_h to arguments
        # Insert position where to obtain inlet parameters
        pos = [0.917, 0.917, 0.917, 0.7, 1]  # 0 < pos = x/L < 1 ---- x-position of nacelle
        nac_h = [0.56, 0.56, 0.0279654, 0.0279654, 1]  # meters above body's surface ---- height of nacelle

        # Calculation options -------------------------------------------------------------------------------------------------
        flags = [self.case,
                 self.turb_model,
                 self.trans_model,
                 self.interact,
                 self.compr,
                 self.discret,
                 self.ell_method,
                 self.visc,

                 ]

        if self.option == 1:
            results = self.filename[::2]
            label = self.filename[1::2]
            name = '_'.join(self.filename[1::2])

        if self.option == 0 or self.option == 2:
            label = self.filename[1]
            name = self.filename[0]

        ###############################################################################################################
        # --------------------------------------------BL CALCULATION---------------------------------------------------
        if self.option == 0 or self.option == 2:
            start_time = time.time()
            counter = 0  # Counts number of interactions between viscid and inviscid flow
            eps = 1e-6  # Relative Tolerance for convergence of linear system and integration loops of turbulent calculations
            eps_2 = 2.5e-2  # Relative tolerance for convergence of inviscid and viscous interaction
            # Compares every element in array, eps_2 < 1e-3 not recommended )
            if (flags[0] == 10 or flags[0] == 11) and self.AR_ell != 1:
                if self.ell_method == 0:
                    # elliptic (one longitudinal section calculated with equivalent radius)
                    Ratio = (2 / np.pi) * ellipe(1 - self.AR_ell ** 2)  # ratio of equivalent radius to semi-major axis
                    self.Xn_d, self.Yn_d, file, sigma, j_s, j_v, self.pot_sol, self.surface, self.atmos, self.boundarylayer = CalculateBL(
                        Ratio, 0).calculateBoundaryLayer(self.altitude_inf, self.Ma_inf, self.no, self.weight, self.FPR,
                                                         self.eta_pol, self.filename[0], flags, counter, eps, eps_2,
                                                         name)
                    results = [file]
                    label = ["Elliptic body %s" % (self.filename[1])]
                else:
                    # elliptic (two longitudinal sections at semi-major and semi-minor axis calculated with equal Xn and then interpolated)
                    Xn_old, Yn_old, file1, sigma1, j_s1, j_v1 = CalculateBL(1, 0).calculateBoundaryLayer(
                        self.altitude_inf, self.Ma_inf, self.no, self.weight, self.FPR, self.eta_pol, self.filename[0],
                        flags, counter, eps, eps_2, name, self.sing_type)
                    self.Xn_d, self.Yn_d, file2, sigma2, j_s2, j_v2, self.pot_sol2, self.surface2, self.boundarylayer2 = CalculateBL(
                        self.AR_ell, Xn_old).calculateBoundaryLayer(self.altitude_inf, self.Ma_inf, self.no,
                                                                    self.weight, self.FPR, self.eta_pol,
                                                                    self.filename[0], flags, counter, eps, eps_2, name,
                                                                    self.sing_type)
                    results = [file1, file2]
                    label = ["Semi-major axis %s" % (self.filename[1]), "Semi-minor axis %s" % (self.filename[1])]
            else:
                # axisymmetric (only one longitudinal section calculated)
                self.Xn_d, self.Yn_d, file, sigma, j_s, j_v, self.pot_sol, self.surface, self.atmos, self.boundarylayer = CalculateBL(
                    1, 0).calculateBoundaryLayer(self.altitude_inf, self.Ma_inf, self.no, self.weight, self.FPR,
                                                 self.eta_pol, self.filename[0], flags, counter, eps, eps_2, name,
                                                 self.sing_type)
                results = [file]

            print('---------------------------')
            print("time elapsed: {:.2f}s".format(time.time() - start_time))
