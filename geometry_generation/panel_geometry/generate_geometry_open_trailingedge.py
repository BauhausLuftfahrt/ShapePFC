"""
Generate a fuselage and nacelle geometry for a propulsive fuselage.

Copyright Bauhaus Luftfahrt e.V.
"""

from typing import List
import geomdl.BSpline as bspline
from geomdl import utilities
from geomdl.visualization import VisMPL
from geomdl import multi
import numpy as np
from scipy.interpolate import interp1d


class GenerateGeom:
    """
        Author: Anais Habermann
        Date: 23.02.2021

    Args:
        l_fus: (float)                            [m], Range: [30, 80]          Fuselage total length
        ff_in: (float)                            [-], Range: [0.85, 0.95]      Fuselage Fan (FF) rel. rotor inlet position
        c_3: (float)                              [-], Range: [8, 20]           Fuselage slenderness ratio SLR = L_fus / d_fus
        c_4: (float)                              [-]                           FF hub diameter to FF tip diameter: c_4= r_hub / r_tip
        h_duct: (float)                           [m], Range: [0.3, 0.9]        FF rotor inlet hub height
        A18_A2: (float)                           [-], Range: [0.3, 1.0]        Ratio of FF exit (station 18) and FF rotor inlet area (station 2)
        delta_xFF: (float)                        [m]                           Length of FF stage (x_FF,stator,out - x_FF,rotor,in)
        AR_ell: (float)                           [-], Range: [0, 1.0]          Aspect ratio of elliptic cross-section, optional
        c_nac: (float)                            [m]                           Nacelle chord lengt, optional
        A13_A2: (float)                           [-]                           Ratio of FF stator outlet (station 12) and FF rotor inlet area (station 2), optional
        alpha_1: (float)                          [deg]                         Slope of fuselage to FF inlet contour, optional
        t_c_max: (float)                          [-], Range: [0.06, 0.16]      Nacelle thickness-to-chord ratio, optional
        x_tcmax: (float)                          [-], Range: [0, 0.5]          Rel. position of nacelle thickness-to-chord ratio, optional
        r_inl: (float)                            [-], Range: [0.1, 0.5]        Relative nacelle inlet length

    Returns:
        full_list_fuselage: (list)                  List of fuselage coordinates, starting from the leading edge
        full_list_nacelle_top: (list)               List of nacelle upper contour coordinates, starting from the leading edge
        full_list_nacelle_bottom: (list)            List of nacelle lower contour coordinates, starting from the trailing edge
        x_ff: (float)                               FF rotor inlet x coordinate
        self.y3: (float)                            FF rotor inlet y coordinate of fuselage contour
        self.x6: (float)                            FF stator outlet x coordinate
        self.y6 (float)                             FF stator outlet y coordinate of fuselage contour
        A1_A2: (float)                              Station 1 to station 2 area ratio
        Astar_A2: (float)                           Narrowest cross section to station 2 area ratio
        A13_A2: (float)                             Station 13 to station 2 area ratio
        self.A18_A2: (float)                        Station 18 to station 2 area ratio
    """

    def __init__(self, l_fus: float, ff_in: float, c_3: float, c_4: float, h_duct: float, A18_A2: float,
                 delta_xFF: float, name: str, AR_ell=1.0, c_nac=0.0, A13_A2=0.0, alpha_1=0, t_c_max=0, x_tcmax=0, r_inl=0):
        self.l_fus = l_fus
        self.ff_in = ff_in
        self.c_3 = c_3
        self.c_4 = c_4
        self.h_duct = h_duct
        self.A18_A2 = A18_A2
        self.delta_xFF = delta_xFF
        self.name = name
        self.AR_ell = AR_ell
        self.c_nac = c_nac
        self.A13_A2 = A13_A2
        self.alpha_1 = alpha_1
        self.t_c_max = t_c_max
        self.x_tcmax = x_tcmax
        self.r_inl = r_inl

    def build_geometry(self):
        """
        Method executing the main code.
        """

        def rotate(origin, point, angle):
            """
            Rotate a point counterclockwise by a given angle around a given origin. Clockwise rotation for negative angle.

            The angle should be given in radians.
            """
            ox, oy = origin
            px, py = point

            qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
            qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
            return qx, qy

        n = 10000
        n1 = 1000

        d_fus = self.l_fus / self.c_3

        if self.alpha_1 == 0:
            alpha_1 = 13 * np.pi / 180          # similarity to prepare_centreline_fan_stage geometry Rev 07
        else:
            alpha_1 = self.alpha_1 * np.pi / 180

        if self.c_nac == 0:
            c_nac = 0.84 * d_fus/2               # similarity to prepare_centreline_fan_stage geometry Rev 07
        else:
            c_nac = self.c_nac

        if self.A13_A2 == 0:
            A13_A2 = 0.975                      # Result of prepare_centreline_fan_stage D2.03
        else:
            A13_A2 = self.A13_A2

        if self.t_c_max == 0:
            t_c_max = 0.1                     # Result of prepare_centreline_fan_stage D2.03
        else:
            t_c_max = self.t_c_max

        if self.x_tcmax == 0:
            x_tcmax = 0.3                     # Result of prepare_centreline_fan_stage D2.03
        else:
            x_tcmax = self.x_tcmax

        if self.r_inl == 0:
            l_inl = 0.4 * c_nac                     # Result of prepare_centreline_fan_stage D2.03
        else:
            l_inl = self.r_inl * c_nac

        r_max = d_fus / 2  # Radius of cylindrical section of the fuselage
        r_tip = self.h_duct/(1-self.c_4)
        r_min = self.c_4*r_tip
        l_nose = 1.6 * 2 * r_max   # literature HOWE - Nose elipse -> add reference!
        i = np.radians(np.degrees(alpha_1) - 5.5)
        x_ff = self.ff_in * self.l_fus  # x position of FF rotor inlet (station 2)
        l_nac = c_nac*np.cos(i)

        if l_nac <= 1.1*self.delta_xFF:
            raise Warning("The FF module length exceeds the nacelle length. Check input.")

        if x_tcmax*c_nac >= l_inl:
            raise Warning("The max. thickness position of the nacelle is behind the FF rotor inlet position. "
                          "The parameterization cannot handle this. Check input.")

        if self.h_duct >= d_fus/2:
            raise Warning("The duct height is bigger than the fuselage radius. Check input.")

        # define main coordinates of fuselage
        self.x0 = 0
        self.y0 = 0
        self.x1 = l_nose
        self.y2 = r_max * self.AR_ell
        self.x3 = x_ff
        self.y3 = r_min
        self.x6 = self.x3 + self.delta_xFF
        self.x7 = self.l_fus

        # define nacelle coordinates for nacelle with zero degree incidence angle
        self.x20 = self.x3
        self.y20 = r_tip
        self.x21 = self.x6
        self.x24 = self.x3+(c_nac-l_inl)
        self.x28 = self.x29 = self.x30 = self.x3-l_inl
        self.x31 = self.x28 + 0.01*c_nac
        self.x27 = self.x28 + 0.15*c_nac
        self.x26 = self.x32 = self.x28 + x_tcmax*c_nac
        self.y32 = self.y20 - (self.x20 - self.x32)*np.tan(np.deg2rad(1.5))
        self.y31 = self.y32 - (self.x32 - self.x31)*np.tan(np.deg2rad(1.5))
        self.y26 = self.y32 + t_c_max*c_nac
        self.x25 = self.x28 + 0.75*c_nac
        self.y27 = self.y26 - (self.x26 - self.x27)*np.tan(np.deg2rad(1.5))
        self.y25 = self.y26 + (self.x25 - self.x26)*np.tan(np.deg2rad(1.5))
        self.y24 = self.y29 = (self.y32 + self.y26)/2
        self.y28 = self.y29 + 0.4*t_c_max*c_nac
        self.y30 = self.y29 - 0.45*t_c_max*c_nac
        self.x22 = self.x24 - 0.1*c_nac
        self.x23 = self.x24 - 0.03*c_nac
        self.y21 = self.y20 + (self.x21 - self.x20)*np.tan(np.deg2rad(1.5))
        self.y22 = self.y21 + (self.x22 - self.x21)*np.tan(np.deg2rad(1.5))
        self.y23 = self.y24 + (self.x24 - self.x23)*np.tan(np.deg2rad(6))
        self.x4 = self.x24
        self.y12 = self.y0 + 0.01

        # rotate nacelle around FF rotor inlet point
        origin = (self.x20, self.y20)
        self.x21, self.y21 = rotate(origin, (self.x21, self.y21), -i)
        self.x22, self.y22 = rotate(origin, (self.x22, self.y22), -i)
        self.x23, self.y23 = rotate(origin, (self.x23, self.y23), -i)
        self.x24, self.y24 = rotate(origin, (self.x24, self.y24), -i)
        self.x25, self.y25 = rotate(origin, (self.x25, self.y25), -i)
        self.x26, self.y26 = rotate(origin, (self.x26, self.y26), -i)
        self.x27, self.y27 = rotate(origin, (self.x27, self.y27), -i)
        self.x28, self.y28 = rotate(origin, (self.x28, self.y28), -i)
        self.x29, self.y29 = rotate(origin, (self.x29, self.y29), -i)
        self.x30, self.y30 = rotate(origin, (self.x30, self.y30), -i)
        self.x31, self.y31 = rotate(origin, (self.x31, self.y31), -i)
        self.x32, self.y32 = rotate(origin, (self.x32, self.y32), -i)

        # define fuselage coordinates, which are dependent on nacelle geometry
        A2 = np.pi*(self.y20**2-self.y3**2)     # FF rotor inlet area
        self.y4 = self.y14 = self. y15 = np.sqrt(self.y24**2-self.A18_A2*A2/np.pi)
        self.y6 = np.sqrt(self.y21**2-A13_A2*(self.y20**2-self.y3**2))
        teta_1 = np.arctan((self.y6-self.y3)/self.delta_xFF)    # Angle between rotor inlet and stator outlet area to ensure A13_A2
        self.x13 = (self.x4 - self.x6)/3+self.x6
        self.x14 = (self.x4 - self.x6)/4*3+self.x6
        self.y13 = self.y6 + (self.x13 - self.x6)*np.tan(teta_1)
        self.x9 = self.x3 - 0.15*(self.y2 - self.y3) / np.tan(alpha_1)
        self.y9 = self.y3 - (self.x3-self.x9)*np.tan(teta_1)
        self.x5 = self.x9 - 0.5*(self.y2 - self.y9)/np.tan(alpha_1)
        self.x8 = self.x9 - (self.y2 - self.y9)/np.tan(alpha_1)
        self.y5 = self.y9 + 0.5*(self.y2 - self.y9)
        self.y8 = self.y9 + (self.y2 - self.y9)
        self.x2 = self.x8 - 0.15*(self.y2 - self.y9) / np.tan(alpha_1)
        l_cone = self.l_fus - self.x4   # length of fuselage aft-cone
        self.x15 = self.x4+0.1*l_cone
        alpha_2 = np.arctan(self.y4/(self.l_fus-self.x15))  # angle of fuselage aft-cone
        self.x12 = self.l_fus - self.y12*np.tan(alpha_2)
        self.x11 = self.x4+0.2*l_cone
        self.y11 = np.tan(alpha_2)*(self.l_fus-self.x11)

        # points definition for fuselage
        a0 = [self.x0, self.y0]
        a1 = [self.x0, self.y2]
        a2 = [self.x1, self.y2]
        a3 = [self.x2, self.y2]
        a4 = [self.x8, self.y2]
        a5 = [self.x5, self.y5]
        a6 = [self.x9, self.y9]
        a7 = [self.x3, self.y3]
        a8 = [self.x6, self.y6]
        a9 = [self.x13, self.y13]
        a10 = [self.x14, self.y14]
        a11 = [self.x4, self.y4]
        a12 = [self.x15, self.y15]
        a13 = [self.x11, self.y11]
        a14 = [self.x12, self.y12]

        # points definition for nacelle
        b0 = [self.x20, self.y20]
        b1 = [self.x21, self.y21]
        b2 = [self.x22, self.y22]
        b3 = [self.x23, self.y23]
        b4 = [self.x24, self.y24]
        b4_1 = [self.x24, self.y24+0.01] #open TE
        b5 = [self.x25, self.y25]
        b6 = [self.x26, self.y26]
        b7 = [self.x27, self.y27]
        b8 = [self.x28, self.y28]
        b9 = [self.x29, self.y29]
        b10 = [self.x30, self.y30]
        b11 = [self.x31, self.y31]
        b12 = [self.x32, self.y32]

        # Curves definition of the fuselage geometry
        c0 = bspline.Curve()
        c0.degree = 2
        c0.ctrlpts = [a0, a1, a2]
        c0.knotvector = utilities.generate_knot_vector(c0.degree, len(c0.ctrlpts))
        c0.delta = 0.1
        c0.evaluate()
        ratio_0 = (a2[0] - a0[0]) / self.l_fus
        c0.sample_size = int(round(ratio_0 * n))*10
        c0_points = c0.evalpts

        c1 = bspline.Curve()
        c1.degree = 1
        c1.ctrlpts = [a2, a3]
        c1.knotvector = utilities.generate_knot_vector(c1.degree, len(c1.ctrlpts))
        c1.delta = 0.1
        c1.evaluate()
        ratio_1 = (a3[0] - a2[0]) / self.l_fus
        c1.sample_size = int(round(ratio_1 * n))
        c1_points = c1.evalpts

        c2 = bspline.Curve()
        c2.degree = 2
        c2.ctrlpts = [a3, a4, a5]
        c2.knotvector = utilities.generate_knot_vector(c2.degree, len(c2.ctrlpts))
        c2.delta = 0.1
        c2.evaluate()
        ratio_2 = (a5[0] - a3[0]) / self.l_fus
        c2.sample_size = int(round(ratio_2 * n))
        c2_points = c2.evalpts

        c3 = bspline.Curve()
        c3.degree = 2
        c3.ctrlpts = [a5, a6, a7]
        c3.knotvector = utilities.generate_knot_vector(c3.degree, len(c3.ctrlpts))
        c3.delta = 0.1
        c3.evaluate()
        ratio_3 = (a7[0] - a5[0]) / self.l_fus
        c3.sample_size = int(round(ratio_3 * n))
        c3_points = c3.evalpts

        # fuselage duct
        c4 = bspline.Curve()
        c4.degree = 1
        c4.ctrlpts = [a7, a8]
        c4.knotvector = utilities.generate_knot_vector(c4.degree, len(c4.ctrlpts))
        c4.delta = 0.1
        ratio_4 = (a8[0]-a7[0]) / self.l_fus
        c4.sample_size = int(round(ratio_4 * n))
        c4.evaluate()
        c4_points = c4.evalpts

        c5 = bspline.Curve()
        c5.degree = 3
        c5.ctrlpts = [a8, a9, a10, a11]
        c5.knotvector = utilities.generate_knot_vector(c5.degree, len(c5.ctrlpts))
        c5.delta = 0.1
        c5.evaluate()
        ratio_5 = (a11[0]-a8[0]) / self.l_fus
        c5.sample_size = int(round(ratio_5 * n))
        c5_points = c5.evalpts

        c6 = bspline.Curve()
        c6.degree = 2
        c6.ctrlpts = [a11, a12, a13]
        c6.knotvector = utilities.generate_knot_vector(c6.degree, len(c6.ctrlpts))
        c6.delta = 0.1
        ratio_6 = (a13[0] - a11[0]) / self.l_fus
        c6.sample_size = int(round(ratio_6 * n))
        c6.evaluate()
        c6_points = c6.evalpts

        c7 = bspline.Curve()
        c7.degree = 1
        c7.ctrlpts = [a13, a14]
        c7.knotvector = utilities.generate_knot_vector(c7.degree, len(c7.ctrlpts))
        c7.delta = 0.1
        ratio_7 = (a14[0] - a13[0]) / self.l_fus
        c7.sample_size = int(round(ratio_7 * n))
        c7.evaluate()
        c7_points = c7.evalpts

        # Curves definition of the nacelle geometry
        c10 = bspline.Curve()
        c10.degree = 1
        c10.ctrlpts = [b0, b1]
        c10.knotvector = utilities.generate_knot_vector(c10.degree, len(c10.ctrlpts))
        c10.delta = 0.1
        c10.evaluate()
        ratio_10 = (b1[0] - b0[0]) / c_nac
        c10.sample_size = int(round(ratio_10 * n1))*10
        c10_points = c10.evalpts

        c11 = bspline.Curve()
        c11.degree = 3
        c11.ctrlpts = [b1, b2, b3, b4]
        c11.knotvector = utilities.generate_knot_vector(c11.degree, len(c11.ctrlpts))
        c11.delta = 0.1
        c11.evaluate()
        ratio_11 = (b4[0] - b1[0]) / c_nac
        c11.sample_size = int(round(ratio_11 * n1))
        c11_points = c11.evalpts

        c12 = bspline.Curve()
        c12.degree = 2
        c12.ctrlpts = [b4_1, b5, b6]   #open TE
        c12.knotvector = utilities.generate_knot_vector(c12.degree, len(c12.ctrlpts))
        c12.delta = 0.1
        c12.evaluate()
        ratio_12 = (b4[0] - b6[0]) / c_nac
        c12.sample_size = int(round(ratio_12 * n1))
        c12_points = c12.evalpts

        c13 = bspline.Curve()
        c13.degree = 3
        c13.ctrlpts = [b6, b7, b8, b9]
        c13.knotvector = utilities.generate_knot_vector(c13.degree, len(c13.ctrlpts))
        c13.delta = 0.1
        c13.evaluate()
        ratio_13 = (b6[0] - b9[0]) / c_nac
        c13.sample_size = int(round(ratio_13 * n1))
        c13_points = c13.evalpts

        c14 = bspline.Curve()
        c14.degree = 3
        c14.ctrlpts = [b9, b10, b11, b12]
        c14.knotvector = utilities.generate_knot_vector(c14.degree, len(c14.ctrlpts))
        c14.delta = 0.1
        c14.evaluate()
        ratio_14 = (b12[0] - b9[0]) / c_nac
        c14.sample_size = int(round(ratio_14 * n1))
        c14_points = c14.evalpts

        c15 = bspline.Curve()
        c15.degree = 1
        c15.ctrlpts = [b12, b0]
        c15.knotvector = utilities.generate_knot_vector(c10.degree, len(c10.ctrlpts))
        c15.delta = 0.1
        c15.evaluate()
        ratio_15 = (b0[0] - b12[0]) / c_nac
        c15.sample_size = int(round(ratio_15 * n1))*10
        c15_points = c15.evalpts

        # generate lists for nacelle and fuselage coordinates
        fuselage = c0_points + c1_points + c2_points + c3_points + c4_points + c5_points + c6_points + c7_points
        nacelle = c12_points + c13_points + c14_points + c15_points + c10_points + c11_points

        # split nacelle list in top an bottom
        LE = min(nacelle)   # identify leading edge point of nacelle
        TE = max(nacelle)   # identify trailing edge point of nacelle
        LE_idx = nacelle.index(LE)
        nacelle_top = nacelle[0:LE_idx]
        nacelle_bottom = nacelle[LE_idx:len(nacelle)]

        full_list_fuselage: List[List[int]] = []
        full_list_nacelle_top: List[List[int]] = []
        full_list_nacelle_bottom: List[List[int]] = []

        for i in fuselage:
            if i not in full_list_fuselage:
                full_list_fuselage.append(i)

        for i in nacelle_top:
            if i not in full_list_nacelle_top:
                full_list_nacelle_top.append(i)

        for i in nacelle_bottom:
            if i not in full_list_nacelle_bottom:
                full_list_nacelle_bottom.append(i)

        # calculate important area ratios. Inlet area ratio
        x1 = [x for x,y in c3_points]
        y1 = [y for x,y in c3_points]
        f1 = interp1d(x1, y1, kind='cubic')
        A1_y_fuse = f1(LE[0])
        A1 = np.pi*(LE[1]**2-A1_y_fuse**2)
        A1_A2 = A1/A2

        # Narrowest section area ratio
        nac_inl = c14_points[0:len(c14_points)-1] + c15_points
        x2 = [x for x,y in nac_inl]
        y2 = [y for x,y in nac_inl]
        f2 = interp1d(x2, y2, kind='cubic')
        x_int = np.arange(min(x2),max(x2),0.001)
        fuse_x = f1(x_int)
        nac_x = f2(x_int)
        diff = nac_x - fuse_x
        Astar_y = min(diff)
        Astar_idx = np.where(diff == Astar_y)
        Astar_x = x_int[Astar_idx]
        Astar = np.pi*(f2(Astar_x)**2-f1(Astar_x)**2)
        Astar_A2 = Astar[0]/A2

        '''# visualize fuselage geometry. used for debugging
        vis_config = VisMPL.VisConfig(legend=False)
        vis_obj = VisMPL.VisCurve2D(vis_config)
        curves_total = multi.CurveContainer()
        curves_total.add(c0)
        curves_total.add(c1)
        curves_total.add(c2)
        curves_total.add(c3)
        curves_total.add(c4)
        curves_total.add(c5)
        curves_total.add(c6)
        curves_total.add(c7)
        curves_total.add(c10)
        curves_total.add(c11)
        curves_total.add(c12)
        curves_total.add(c13)
        curves_total.add(c14)
        curves_total.add(c15)
        curves_total.vis = vis_obj
        curves_total.render()
        '''

        return full_list_fuselage, full_list_nacelle_top, full_list_nacelle_bottom, (x_ff, self.y3), \
               (self.x6, self.y6), A1_A2, Astar_A2, A13_A2, self.A18_A2


if __name__ == "__main__":
    step_3_val = GenerateGeom(67.2, 0.93, 11.03, 0.51, 0.57, 0.69, 0.7, 'baseline')
    full_list_fuselage, full_list_nacelle_top, full_list_nacelle_bottom, (x_ff_in, y_ff_in), \
    (x_ff_out, y_ff_out), A1_A2, Astar_A2, A13_A2, A18_A2 = step_3_val.build_geometry()

    print('FF rotor inlet:',("{:.3f}".format(x_ff_in), "{:.3f}".format(y_ff_in)),'\nFF stator outlet:',
          ("{:.3f}".format(x_ff_out), "{:.3f}".format(y_ff_out)), '\nA1/A2:', "{:.3f}".format(A1_A2),
          '\nA*/A2:', "{:.3f}".format(Astar_A2), '\nA13/A2:', "{:.3f}".format(A13_A2), '\nA18/A2:',
          "{:.3f}".format(A18_A2))