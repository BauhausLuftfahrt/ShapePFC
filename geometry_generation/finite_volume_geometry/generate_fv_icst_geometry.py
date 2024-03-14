"""
Generate a fuselage and nacelle geometry for a propulsive fuselage using a mix of Bezier curved and
intuitive Class/Shape Transformation.

Author:  A. Habermann
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from geometry_generation.intuitive_class_shape_transformation.cst_functions import cst
from geometry_generation.intuitive_class_shape_transformation.bpi_calc import bpi_fuse_preint, bpi_fuse_int, \
    bpi_fuse_noz1, bpi_fuse_noz2, bpi_nac_int, \
    bpi_nac_noz, bpi_nac_cowl, bpi_fuse_int2
from scipy.optimize import fsolve
from finite_volume.open_foam.prepare_body_force_model.prepare_centreline_fan_stage.get_scaled_bfm_outline import \
    get_bfm_outline


class GenerateGeomICST:
    """
        Author: Anais Habermann
        Date: 18.11.2022

    Args:
        :param f_x_ff: (float)      [-]     Rel. FF rotor inlet position (station 12) x_ff/l_fuse_tot
        :param r_cent_f: (float)    [m]     Fuselage center section radius
        :param l_cent_f: (float)    [m]     Fuselage nose and center section length
        :param f_rmax: (float)      [-]     Ratio of max. outer nacelle radius to fuselage center section radius
        :param f_xmax: (float)      [-]     Rel. position of max. nacelle radius w.r.t. nacelle length
        :param f_lnac: (float)      [-]     Rel. nacelle length l_nac/r_12_hub
        :param h_duct: (float)      [m]     Station 12 duct height
        :param f_r12: (float)       [-]     Station 12 hub to tip ratio r_hub_12/r_tip_12
        :param f_lint: (float)      [-]     Rel. intake length l_int/l_nac
        :param teta_f_cone: (float) [°]     Fuselage angle of nozzle one
        :param f_rho_le: (float)    [-]     Inner LE nacelle radius upper to lower ratio
        :param f_l_nose: (float)    [-]     Relative fuselage nose length l_nose/l_cent_f
        :param ahi_athr: (float)    [-]     Highlight to throat area ratio
        :param athr_a12: (float)    [-]     Throat to station 12 area ratio
        :param a18_a13: (float)     [-]     Station 18 (nozzle exit) to 12 area ratio
        :param f_xthr: (float)      [-]     Rel. position of throat x_thr/l_int
        :param beta_te_up: (float)  [°]     Nacelle outer boattail angle
        :param beta_te_low: (float) [°]     Nacelle inner boattail angle
        :param r_te_hub: (float)    [m]     Fuselage TE radius (usually 0)
        :param f_r18hub: (float)    [-]     Ratio between hub radius at station 18 and 13
        :param f_rthrtip: (float)   [-]     Ratio between throat and highlight tip radius r_thr_tip/r_hi_tip
        :param teta_int_in: (float) [°]     Angle at fuselage inlet

Returns:
        full_list_fuselage: (list)          List of fuselage coordinates, starting from the leading edge
        full_list_nacelle_top: (list)       List of nacelle upper contour coordinates, starting from the leading edge
        full_list_nacelle_bottom: (list)    List of nacelle lower contour coordinates, starting from the trailing edge
        l_fuse_tot: (float)         [m]     Total fuselage length
        f_slr: (float)              [-]     Fuselage Slenderness Ratio l_fuse_tot/d_cent_f
        tc_max: (float)             [-]     Nacelle max. thickness-to-chord ratio
        tc_max_x: (float)           [m]     Rel. location of nacelle max. t/c
        i_nac: (float)              [°]     Nacelle incidence angle
        c_nac: (float)              [m]     Nacelle chord length
        teta_f_aft: (float)         [°]     Angle of fuselage aft section
        Astar_A2: (float)           [-]     Narrowest cross section to station 2 area ratio
"""

    # ensure geometry similitude for prepare_centreline_fan_stage fan stage
    FAN_STAGE_AR = 1.4282549216350495

    def __init__(self, f_x_ff: float, r_cent_f: float, l_cent_f: float, f_rmax: float, f_xmax: float, f_lnac: float,
                 h_duct: float, f_r12: float, f_lint: float, teta_f_cone: float = 10,
                 f_rho_le: float = 1.0, f_l_nose: float = 0.1, ahi_athr: float = 1.05, athr_a12: float = 0.9,
                 a18_a13: float = 0.85, f_xthr: float = 0.05, delta_beta_te: float = 8,
                 beta_te_low: float = 4, r_te_hub: float = 0, f_r18hub: float = 1.2, f_rthrtip: float = 0.95,
                 teta_int_in: float = 10, beta_ff_in: float = -4, teta_ff_in: float = -4, plot: bool = False,
                 samplevars: bool = False, savepath: str = ''):

        self.teta_int_in = teta_int_in
        self.f_rthrtip = f_rthrtip
        self.f_r18hub = f_r18hub
        self.r_te_hub = r_te_hub
        self.beta_te_low = beta_te_low
        self.delta_beta_te = delta_beta_te
        self.f_xthr = f_xthr
        self.a18_a13 = a18_a13
        self.athr_a12 = athr_a12
        self.ahi_athr = ahi_athr
        self.f_l_nose = f_l_nose
        self.f_rho_le = f_rho_le
        self.teta_f_cone = teta_f_cone
        self.f_lint = f_lint
        self.f_r12 = f_r12
        self.h_duct = h_duct
        self.f_lnac = f_lnac
        self.f_xmax = f_xmax
        self.f_rmax = f_rmax
        self.l_cent_f = l_cent_f
        self.r_cent_f = r_cent_f
        self.f_x_ff = f_x_ff
        self.beta_ff_in = beta_ff_in
        self.teta_ff_in = teta_ff_in
        self.plot = plot
        self.samplevars = samplevars
        self.savepath = savepath

    def build_geometry(self):
        """
        Method executing the main code.
        """

        # scale fan stage
        l_ff_stage = self.h_duct * self.FAN_STAGE_AR

        # CALCULATE REQUIRED INPUT PARAMETERS
        r_cent_f = self.r_cent_f
        r_f_te = self.r_te_hub
        l_cent_f = self.l_cent_f
        beta_te_low = -np.deg2rad(self.beta_te_low)
        beta_te_up = np.deg2rad(self.beta_te_low + self.delta_beta_te)
        teta_f_cone_actual = np.deg2rad(self.teta_f_cone)  # value, which is used at cone te
        teta_f_cone_calc = np.deg2rad(
            np.rad2deg(teta_f_cone_actual) - 0.3)  # value, which is used for length calculation

        l_nose_f = self.f_l_nose * self.l_cent_f

        ar_nose = l_nose_f / r_cent_f

        r_12_tip = self.h_duct / (1 - self.f_r12)
        r_12_hub = r_12_tip - self.h_duct
        l_nac = self.f_lnac * r_12_hub
        l_int = l_nac * self.f_lint
        l_thr = self.f_xthr * l_int
        l_max = self.f_xmax * l_nac
        l_noz = l_nac - l_int - l_ff_stage

        a_12 = np.pi * (r_12_tip ** 2 - r_12_hub ** 2)
        a_thr = a_12 * self.athr_a12
        a_hi = a_thr * self.ahi_athr

        def sys_equs(vars):
            x1, x2, x3, x4 = vars
            eq1 = x4 ** 2 - x3 ** 2 - a_thr / np.pi
            eq2 = x2 ** 2 - x1 ** 2 - a_hi / np.pi
            eq3 = self.f_rthrtip * x2 - x4
            eq4 = (x1 - x3) / l_thr - np.tan(np.deg2rad(self.teta_int_in))
            return [eq1, eq2, eq3, eq4]

        r_hi_hub, r_hi_tip, r_thr_hub, r_thr_tip = fsolve(sys_equs, (1, 1, 1, 1))

        r_max = self.f_rmax * r_hi_tip

        # scale fan stage
        station_121, station_122, station_131, station_132 = get_bfm_outline(self.h_duct, r_12_hub, 0)

        r_13_tip = station_132[1][-1]
        r_13_hub = station_132[1][0]
        r_18_hub = self.f_r18hub * r_13_hub

        l_cone_f = (r_18_hub - r_f_te) / np.tan(teta_f_cone_calc)
        l_f_preint = (l_int - self.l_cent_f * (self.f_x_ff - 1) - (l_nac + l_cone_f) * self.f_x_ff) / (self.f_x_ff - 1)
        l_fuse_tot = l_cent_f + l_f_preint + l_int + l_ff_stage + l_noz + l_cone_f
        l_ff = self.f_x_ff * l_fuse_tot

        teta_f_aft = np.arctan((r_cent_f - r_hi_hub) / l_f_preint)

        l_12 = l_ff - l_cent_f - l_f_preint
        l_13 = l_12 + l_ff_stage
        x_12 = self.l_cent_f + l_f_preint + l_12

        x_121_f = station_121[0][0] + x_12
        x_121_n = station_121[0][-1] + x_12
        x_122_f = station_122[0][0] + x_12
        x_122_n = station_122[0][-1] + x_12
        x_131_f = station_131[0][0] + x_12
        x_131_n = station_131[0][-1] + x_12
        x_132_f = station_132[0][0] + x_12
        x_132_n = station_132[0][-1] + x_12

        y_121_f = station_121[1][0]
        y_121_n = station_121[1][-1]
        y_122_f = station_122[1][0]
        y_122_n = station_122[1][-1]
        y_131_f = station_131[1][0]
        y_131_n = station_131[1][-1]
        y_132_f = station_132[1][0]
        y_132_n = station_132[1][-1]

        l_noz_f = l_nac - l_int - (x_132_f - x_121_f)
        l_n_preint = l_f_preint + (x_131_n - x_131_f)
        l_n_int = x_121_n - l_f_preint - l_cent_f

        # l_13 = l_12+l_ff_stage
        # l_12_f = self.l_cent_f+l_f_preint+l_13
        x_thr = self.l_cent_f + l_f_preint + l_thr
        l_hi = l_cent_f + l_f_preint
        l_18 = l_hi + l_nac
        x_13_f = x_132_f
        l_hi_f = l_cent_f + l_f_preint
        l_thr_f = l_hi_f + l_thr
        x_12_f = x_121_f

        beta_ff = np.arctan((y_122_n - y_121_n) / (x_122_n - x_121_n))
        beta_ff_in = np.deg2rad(self.beta_ff_in)
        beta_ff_out = np.arctan((y_132_n - y_131_n) / (x_132_n - x_131_n))

        teta_ff = np.arctan((y_122_f - y_121_f) / (x_122_f - x_121_f))
        teta_ff_in = np.deg2rad(self.teta_ff_in)
        teta_ff_out = np.arctan((y_132_f - y_131_f) / (x_132_f - x_131_f))

        a_13 = np.pi * (r_12_tip ** 2 - r_13_hub ** 2)
        a_18 = a_13 * self.a18_a13
        r_18_tip = np.sqrt(a_18 / np.pi + r_18_hub ** 2)

        # NUMBER OF POINTS FOR EACH GEOMETRY SECTION

        def nonlin_spacing(lb, ub, steps, spacing=1.0):
            span = (ub - lb)
            dx = 1.0 / (steps - 1)
            return np.array([lb + (i * dx) ** spacing * span for i in range(steps)])

        n_n1 = 100
        x_n1 = nonlin_spacing(0, 1, n_n1, 1.5)
        n_n2 = 80
        x_n2 = np.linspace(0, 1, n_n2)
        x_n21 = np.linspace(0, 1, int(2 * n_n2 / 3))
        x_n22 = np.linspace(0, 1, int(n_n2 / 3))
        n_n3 = 150
        x_n3 = np.linspace(0, 1, n_n3)
        n_n4 = 200
        xn41 = np.linspace(0.5, 1, int(n_n4 / 2))
        x_n4_orig = np.append(nonlin_spacing(0, 0.5, int(n_n4 / 2), 1.4), xn41[1:])
        n_f1 = 100
        n_f2 = 100
        x_f2 = np.linspace(0, 1, n_f2)
        n_f3 = 100
        x_f3 = np.linspace(0, 1, n_f3)
        n_f4 = 80
        x_f4 = np.linspace(0, 1, n_f4)
        n_f5 = n_n2
        x_f5 = np.linspace(0, 1, n_f5)
        x_f51 = np.linspace(0, 1, int(2 * n_f5 / 3))
        x_f52 = np.linspace(0, 1, int(n_f5 / 3))
        n_f6 = n_n3
        x_f6 = nonlin_spacing(0, 1, int(n_f6), 1 / 1.2)
        n_f7 = 200
        x_f7 = nonlin_spacing(0, 1, int(n_f7), 1.2)

        # NACELLE GEOMETRY
        # N1 - nacelle intake
        bpi_n1 = bpi_nac_int(r_hi_tip - r_hi_tip, r_thr_tip - r_hi_tip, l_thr, r_12_tip - r_hi_tip, l_12, -beta_ff_in,
                             l_n_int)
        y_n1 = cst(x_n1, bpi_n1, ((r_hi_tip - r_hi_tip) / l_int, (r_12_tip - r_hi_tip) / l_n_int), n1=0.5, n2=1)
        x_n1 = x_n1 * l_n_int + l_cent_f + l_f_preint
        y_n1 = y_n1 * l_n_int + r_hi_tip

        # # N2 - FF stage
        # y_n2 = [r_12_tip+x_n2[i]*l_ff_stage*np.tan(beta_ff_in) for i in range(0, len(x_n2))]
        # x_n2 = x_n2*l_ff_stage+l_cent_f+l_int+l_f_preint

        x_n21 = np.linspace(0, 1, int(n_n2 / 3))
        # N2_1 - FF stage rotor
        y_n21 = [y_121_n + x_n21[i] * (x_122_n - x_121_n) * np.tan(beta_ff) for i in range(0, len(x_n22))]
        x_n21 = x_n21 * (x_122_n - x_121_n) + x_121_n

        # N2_2 - FF stage gap
        bpi_n22 = bpi_nac_noz(y_122_n - y_122_n, y_131_n - y_122_n, x_122_n, x_131_n, 0.25 * beta_ff, -beta_ff_out)
        # bpi_nac_noz(r_13_tip-r_13_tip, r_18_tip-r_13_tip, l_13, l_noz, beta_ff_out, beta_te_low)
        y_n22 = cst(x_n22, bpi_n22,
                    ((y_122_n - y_122_n) / (x_131_n - x_122_n), (y_131_n - y_122_n) / (x_131_n - x_122_n)), n1=1, n2=1)
        x_n22 = x_n22 * (x_131_n - x_122_n) + x_122_n
        y_n22 = y_n22 * (x_131_n - x_122_n) + y_122_n

        # N2_3 - FF stage stator
        x_n23 = np.linspace(0, 1, int(n_n2 / 3))
        y_n23 = [y_131_n + x_n23[i] * (x_132_n - x_131_n) * np.tan(beta_ff_out) for i in range(0, len(x_n23))]
        x_n23 = x_n23 * (x_132_n - x_131_n) + x_131_n

        l_noz_n = l_18 - x_132_n

        # N3 - nacelle nozzle
        bpi_n3 = bpi_nac_noz(r_13_tip - r_13_tip, r_18_tip - r_13_tip, l_13, l_noz_n, beta_ff_out, beta_te_low)
        y_n3 = cst(x_n3, bpi_n3, ((r_13_tip - r_13_tip) / l_noz_n, (r_18_tip - r_13_tip) / l_noz_n), n1=1, n2=1)
        x_n3 = x_n3 * l_noz_n + x_132_n  # l_13+l_cent_f+l_f_preint
        y_n3 = y_n3 * l_noz_n + y_n23[-1]  # r_13_tip

        delta_y_max = r_max - r_hi_tip
        delta_x_max = l_max
        rho_le_up = (delta_y_max / delta_x_max) / 9 * self.f_rho_le

        # N4 - nacelle cowling
        bpi_n4 = bpi_nac_cowl(r_hi_tip - r_hi_tip, r_18_tip - r_hi_tip, r_max - r_hi_tip, l_max, rho_le_up, beta_te_up,
                              l_nac)
        y_n4 = cst(x_n4_orig, bpi_n4, ((r_hi_tip - r_hi_tip) / l_nac, (r_18_tip - r_hi_tip) / l_nac), n1=0.5, n2=1)
        x_n4 = x_n4_orig * l_nac + l_cent_f + l_f_preint
        y_n4 = y_n4 * l_nac + r_hi_tip

        # FUSELAGE GEOMETRY
        # F1 - Fuselage nose
        # Haack Series
        x_f1 = np.arange(0.0, l_nose_f + l_nose_f / (n_f1 + 1), l_nose_f / (n_f1 - 1))
        teta = [np.arccos(1 - (2 * x_f1[i] / l_nose_f)) for i in range(0, len(x_f1))]
        C_Haack = 2 / 3
        y_f1 = [r_cent_f / (np.sqrt(np.pi)) * np.sqrt(
            teta[i] - (np.sin(2 * teta[i]) / 2) + C_Haack * (np.sin(teta[i]) ** 3)) for i in range(0, len(x_f1))]
        if np.isnan(y_f1[-1]):
            y_f1[-1] = self.r_cent_f

        # F2 - Fuselage constant section
        x_f2 = l_nose_f + x_f2 * (l_cent_f - l_nose_f)
        y_f2 = [self.r_cent_f] * len(x_f2)

        # F3 - Fuselage Pre-intake
        bpi_f3 = bpi_fuse_preint(r_cent_f - r_cent_f, r_hi_hub - r_cent_f, -np.deg2rad(self.teta_int_in), l_f_preint)
        y_f3 = cst(x_f3, bpi_f3, ((r_cent_f - r_cent_f) / l_f_preint, (r_hi_hub - r_cent_f) / l_f_preint), n1=1, n2=1)
        x_f3 = x_f3 * l_f_preint + l_cent_f
        y_f3 = y_f3 * l_f_preint + r_cent_f

        # F4 - Fuselage intake
        bpi_f4 = bpi_fuse_int(r_hi_hub - r_hi_hub, r_thr_hub - r_hi_hub, r_12_hub - r_hi_hub, l_hi_f, l_thr_f, x_12_f,
                              -np.deg2rad(self.teta_int_in), teta_ff_in, l_int)
        y_f4 = cst(x_f4, bpi_f4, ((r_hi_hub - r_hi_hub) / (l_int), (r_12_hub - r_hi_hub) / (l_int)), n1=1, n2=1)
        x_f4 = x_f4 * (l_int) + l_cent_f + l_f_preint
        y_f4 = y_f4 * (l_int) + r_hi_hub

        # F5_1 - FF stage rotor and gap
        bpi_f51 = bpi_fuse_int2(y_121_f - y_121_f, y_122_f - y_121_f, y_131_f - y_121_f, x_121_f, x_122_f, x_131_f,
                                teta_ff_in,
                                teta_ff_out, x_131_f - x_121_f)
        y_f51 = cst(x_f51, bpi_f51,
                    ((y_121_f - y_121_f) / (x_131_f - x_121_f), (y_131_f - y_121_f) / (x_131_f - x_121_f)), n1=1, n2=1)
        x_f51 = x_f51 * (x_131_f - x_121_f) + l_cent_f + l_int + l_f_preint
        y_f51 = y_f51 * (x_131_f - x_121_f) + y_121_f

        # F5_2 - FF stage stator
        y_f52 = [y_131_f + x_f52[i] * (x_132_f - x_131_f) * np.tan(teta_ff_out) for i in range(0, len(x_f52))]
        x_f52 = x_f52 * (x_132_f - x_131_f) + x_131_f

        # F6 - Fuselage nozzle inside nacelle
        bpi_f6 = bpi_fuse_noz1(r_13_hub - r_13_hub, r_18_hub - r_13_hub, x_13_f, l_18, teta_ff_out, l_noz_f)
        y_f6 = cst(x_f6, bpi_f6, ((r_13_hub - r_13_hub) / l_noz_f, (r_18_hub - r_13_hub) / l_noz_f), n1=1, n2=1)
        x_f6 = x_f6 * l_noz_f + l_cent_f + l_f_preint + l_int + (x_132_f - x_121_f)
        y_f6 = y_f6 * l_noz_f + r_13_hub

        # F7 - Fuselage nozzle outside nacelle
        bpi_f7 = bpi_fuse_noz2(r_18_hub - r_18_hub, r_f_te - r_18_hub, -teta_f_cone_actual, l_cone_f)
        y_f7 = cst(x_f7, bpi_f7, ((r_18_hub - r_18_hub) / l_cone_f, (r_f_te - r_18_hub) / l_cone_f), n1=1, n2=1)
        x_f7 = x_f7 * l_cone_f + (l_fuse_tot - l_cone_f)
        y_f7 = y_f7 * l_cone_f + r_18_hub

        # merge all geometry parts to full geometries
        full_list_fuselage = [[x_f1[i], y_f1[i]] for i in range(0, len(x_f1))]
        full_list_nacelle_top = [[x_n4[i], y_n4[i]] for i in range(0, len(x_n4))]
        full_list_nacelle_bottom = [[x_n1[i], y_n1[i]] for i in range(0, len(x_n1))]

        full_list_fuselage += [[x_f2[i], y_f2[i]] for i in range(1, len(x_f2))]
        full_list_fuselage += [[x_f3[i], y_f3[i]] for i in range(1, len(x_f3))]
        full_list_fuselage += [[x_f4[i], y_f4[i]] for i in range(1, len(x_f4))]
        full_list_fuselage += [[x_f51[i], y_f51[i]] for i in range(1, len(x_f51))]
        full_list_fuselage += [[x_f52[i], y_f52[i]] for i in range(1, len(x_f52))]
        full_list_fuselage += [[x_f6[i], y_f6[i]] for i in range(1, len(x_f6))]
        full_list_fuselage += [[x_f7[i], y_f7[i]] for i in range(1, len(x_f7))]

        full_list_nacelle_bottom += [[x_n21[i], y_n21[i]] for i in range(1, len(x_n21))]
        full_list_nacelle_bottom += [[x_n22[i], y_n22[i]] for i in range(1, len(x_n22))]
        full_list_nacelle_bottom += [[x_n23[i], y_n23[i]] for i in range(1, len(x_n23))]
        full_list_nacelle_bottom += [[x_n3[i], y_n3[i]] for i in range(1, len(x_n3))]

        # simplified back for panel calculations
        full_list_fuselage_panel = [[x_f1[i], y_f1[i]] for i in range(0, len(x_f1))]
        full_list_fuselage_panel += [[x_f2[i], y_f2[i]] for i in range(1, len(x_f2))]

        # CHECK FOR ERRONEOUS GEOMETRIES
        f1_nac_up = interp1d(np.array([full_list_nacelle_top[i][0] for i in range(0, len(full_list_nacelle_top))]),
                             np.array([full_list_nacelle_top[i][1] for i in range(0, len(full_list_nacelle_top))]),
                             kind='cubic')
        f1_nac_low = interp1d([full_list_nacelle_bottom[i][0] for i in range(0, len(full_list_nacelle_bottom))],
                              [full_list_nacelle_bottom[i][1] for i in range(0, len(full_list_nacelle_bottom))],
                              kind='cubic')
        f2_nac = interp1d(x_n1, y_n1, kind='cubic')
        f2_fus = interp1d(x_f4, y_f4, kind='cubic')
        x_duct = np.arange(min(x_f4), max(x_f4), 0.001)
        diff_duct = f2_nac(x_duct) - f2_fus(x_duct)

        nacelle_top_interp = interp1d([i[0] for i in full_list_nacelle_top], [i[1] for i in full_list_nacelle_top])
        nacelle_bot_interp = interp1d([i[0] for i in full_list_nacelle_bottom],
                                      [i[1] for i in full_list_nacelle_bottom])
        x_bottom_interp = np.linspace(min([i[0] for i in full_list_nacelle_bottom]),
                                      max([i[0] for i in full_list_nacelle_bottom]), 100)
        x_top_interp = np.linspace(min([i[0] for i in full_list_nacelle_top]),
                                   max([i[0] for i in full_list_nacelle_top]), 100)

        if self.plot:
            plt.close()
            cl = np.genfromtxt(
                '/finite_volume_post/open_foam/prepare_body_force_model/prepare_centreline_fan_stage/CENTRELINE_rev07_geom.csv',
                delimiter=';', filling_values=np.nan)
            plt.plot(cl[:, 0], cl[:, 1], color='0.7', linestyle='--')
            plt.plot(cl[:, 2], cl[:, 3], color='0.7', linestyle='--')
            plt.plot(cl[:, 4], cl[:, 5], color='0.7', linestyle='--')
            # plot for debugging
            plt.plot(x_n1, y_n1, color='k')
            plt.plot(x_n21, y_n21, color='k')
            plt.plot(x_n22, y_n22, color='k')
            plt.plot(x_n23, y_n23, color='k')
            plt.plot(x_n3, y_n3, color='k')
            plt.plot(x_n4, y_n4, color='k')
            plt.plot(x_f1, y_f1, color='k')
            plt.plot(x_f2, y_f2, color='k')
            plt.plot(x_f3, y_f3, color='k')
            plt.plot(x_f4, y_f4, color='k')
            plt.plot(x_f51, y_f51, color='k')
            plt.plot(x_f52, y_f52, color='k')
            plt.plot(x_f6, y_f6, color='k')
            plt.plot(x_f7, y_f7, color='k')
            plt.plot(station_121[0] + x_12, station_121[1], color='k')
            plt.plot(station_122[0] + x_12, station_122[1], color='k')
            plt.plot(station_131[0] + x_12, station_131[1], color='k')
            plt.plot(station_132[0] + x_12, station_132[1], color='k')
            # plt.scatter(l_max+l_cent_f+l_f_preint, r_max)
            plt.axis('equal')
            # plt.xlim(self.l_cent_f)
            plt.show()
            # plt.savefig('l_fuse%.2f_r_fuse%.2f_lFFstage%.2f_lnac%.2f_teta_cone%.2f.png'
            #             %(l_fuse_tot, self.r_cent_f, self.l_ff_stage, l_nac, self.teta_f_cone))

        nacelle_top_comp = nacelle_top_interp(x_top_interp)
        nacelle_bot_comp = nacelle_bot_interp(x_bottom_interp)

        # CHECK CONSTRAINTS AND RELATION OF PARAMETERS
        if l_nac / l_ff_stage < 1.0:
            raise Warning("The FF module length exceeds the nacelle length. Check input.")
        if r_max < r_hi_tip:
            raise Warning("Nacelle max. radius too small. Check input.")
        if l_thr > l_12:
            raise Warning("Throat and FF rotor inlet position mismatch. Check input.")
        if self.f_x_ff > 1.0 or self.f_xmax > 1.0 or self.f_r12 > 1.0 or self.f_lint > 1.0 or \
                self.f_l_nose > 1.0 or self.ahi_athr < 1.0 or self.athr_a12 > 1.0 or self.a18_a13 > 1.0 or \
                self.f_xthr > 1.0 or self.f_r18hub < 1.0 or self.f_rthrtip > 1.0:
            raise Warning("Constraints of input parameters violated. Check input.")
        if l_f_preint < 0:
            raise Warning("No solution could be found for input parameters. Adapt input.")
        if station_132[0][-1] + x_12 > full_list_nacelle_bottom[-1][0] or station_132[0][-1] + x_12 > \
                full_list_nacelle_top[-1][0]:
            raise Warning("Resulting geometry not valid. Fuselage fan located behind nacelle.")
        if any([full_list_fuselage[i][1] < -1E-4 for i in range(0, len(full_list_fuselage))]) or \
                any([full_list_nacelle_bottom[i][1] < -1E-4 for i in range(0, len(full_list_nacelle_bottom))]) or \
                any([full_list_nacelle_top[i][1] < -1E-4 for i in range(0, len(full_list_nacelle_top))]):
            raise Warning("Resulting geometry not valid. Y < 0.")
        elif any([full_list_fuselage[i][1] > (r_cent_f + 1E-4) for i in range(0, len(full_list_fuselage))]):
            raise Warning("Resulting fuselage geometry not valid. Y > r_fuse.")
        elif any((nacelle_top_comp - nacelle_bot_comp) < 0):
            raise Warning("Resulting nacelle geometry not valid. Y_bottom > Y_top.")
        elif any(diff_duct < 0):
            raise Warning("Resulting geometry not valid. Check duct.")
        elif np.rad2deg(np.arctan((y_f3[0] - y_f3[-1]) / (x_f3[-1] - x_f3[0]))) > 85:
            raise Warning("Resulting geometry not valid. FF pre-intake angle of fuselage too steep (> 85 deg).")
        elif bpi_n1[0] >= 0:
            raise Warning("Resulting geometry not valid (rho_LE).")

        # CALCULATE IMPORTANT DESIGN CHARACTERISTICS
        # Nacelle incidence angle
        i_nac = np.rad2deg(np.arctan((r_hi_tip - r_18_tip) / l_nac))

        # Nacelle chord length
        c_nac = l_nac / np.cos(np.deg2rad(i_nac))

        # Fuselage SLR
        f_slr = l_fuse_tot / (2 * r_cent_f)

        # Nacelle max. thickness-to-chord ratio
        x_nac_max = np.arange(l_cent_f + l_f_preint, l_cent_f + l_f_preint + l_nac, 0.001)
        diff = f1_nac_up(x_nac_max) - f1_nac_low(x_nac_max)
        t_max = max(diff)
        tc_max_idx = np.where(diff == t_max)
        tc_max_x = float((x_nac_max[tc_max_idx] - l_cent_f - l_f_preint) / (l_nac))
        tc_max = t_max / l_nac

        x_nac_max_tot = x_nac_max[tc_max_idx][0]

        if tc_max > 0.25:
            raise Warning("Resulting geometry not valid (t/c)_max too high.")

        # Narrowest section area ratio
        x_int_fus = np.arange(min(x_f4), max(x_f4), 0.001)
        x_int_nac = np.arange(min(x_f4), max(x_f4), 0.001)
        diff = f2_nac(x_int_nac) - f2_fus(x_int_fus)
        Astar_y = min(diff)
        Astar_idx = np.where(diff == Astar_y)
        Astar_x = x_int_nac[Astar_idx]
        Astar = np.pi * (f2_nac(Astar_x) ** 2 - f2_fus(Astar_x) ** 2)
        Astar_A2 = Astar[0] / a_12

        full_nacelle_bottom_reverse = list(reversed(full_list_nacelle_bottom))

        rotor_inlet = [[station_121[0][i] + x_12, station_121[1][i]] for i in range(0, len(station_121[0]))]
        rotor_outlet = [[station_122[0][i] + x_12, station_122[1][i]] for i in range(0, len(station_122[0]))]
        stator_inlet = [[station_131[0][i] + x_12, station_131[1][i]] for i in range(0, len(station_131[0]))]
        stator_outlet = [[station_132[0][i] + x_12, station_132[1][i]] for i in range(0, len(station_132[0]))]

        x_le_rotor = [i[0] for i in rotor_inlet]
        y_le_rotor = [i[1] for i in rotor_inlet]

        x_le_stator = [i[0] for i in stator_inlet]
        y_le_stator = [i[1] for i in stator_inlet]

        rotor_le_coeffs = np.polyfit(y_le_rotor, x_le_rotor, 3)
        stator_le_coeffs = np.polyfit(y_le_stator, x_le_stator, 3)

        h_duct_in = r_hi_tip - r_hi_hub
        h_duct_out = r_18_tip - r_18_hub

        if self.samplevars is True:
            with open(f'{self.savepath}//samplevars', 'w') as f:
                f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n'
                        '| =========                 |                                                 | \n'
                        '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n'
                        '|  \\    /   O peration     | Version:  v2206                                 | \n'
                        '|   \\  /    A nd           | Website:  www.openfoam.com                      | \n'
                        '|    \\/     M anipulation  |                                                 | \n'
                        '\*---------------------------------------------------------------------------*/ \n')
                f.write(f'x1 {l_hi}; \n'
                        f'x2 {x_thr}; \n'
                        f'x3 {x_12_f}; \n'
                        f'x5 {x_132_n}; \n'
                        f'x6 {l_18}; \n'
                        f'x7 {l_fuse_tot + 1}; \n'
                        f'x8 {l_hi - 1}; \n'
                        f'x9 {l_fuse_tot}; \n'
                        f'z11 {r_hi_hub}; \n'
                        f'z12 {r_hi_tip}; \n'
                        f'z21 {r_thr_hub}; \n'
                        f'z22 {r_thr_tip}; \n'
                        f'z31 {r_12_hub}; \n'
                        f'z32 {r_12_tip}; \n'
                        f'z51 {r_13_hub}; \n'
                        f'z52 {r_13_tip}; \n'
                        f'z61 {r_18_hub}; \n'
                        f'z62 {r_18_tip}; \n'
                        f'z72 {50}; \n'
                        )
                f.write('// ************************************************************************* //')

        return full_list_fuselage, full_list_nacelle_top, full_nacelle_bottom_reverse, rotor_inlet, rotor_outlet, \
               stator_inlet, stator_outlet, l_fuse_tot, f_slr, tc_max, tc_max_x, i_nac, c_nac, np.rad2deg(teta_f_aft), \
               Astar_A2, x_thr, x_121_f, x_132_n, rotor_le_coeffs, stator_le_coeffs, h_duct_in, h_duct_out, a_12, a_13, \
               a_18, r_12_tip, x_nac_max_tot, ar_nose, full_list_fuselage_panel
