"""Post-process results from hybrid panel/finite volume method of PFC geometries.

Author:  A. Habermann
"""

import numpy as np
import csv

from post_processing.finite_volume_post.calc_station_data import calc_averaged_station_data
from post_processing.hybrid_method_post.forces_ibl import calc_ibl_forces
from post_processing.finite_volume_post.calc_average import calc_average
from post_processing.finite_volume_post.calc_forces import calc_forces
from post_processing.hybrid_method_post.forces_fuselage_hybrid import int_forces_smooth


class PostProcess:

    def __init__(self, atmos, geometry, surface, potential_results, ibl_results, interface_loc, casepath,
                 allresultspath, average='mass_flow', case_type='Default'):
        self.allresultspath = allresultspath
        self.surface = surface
        self.interface_loc = interface_loc
        self.atmos = atmos
        self.geometry = geometry
        self.potential_results = potential_results
        self.ibl_results = ibl_results
        self.casepath = casepath
        self.case_type = case_type
        self.average = average

    def calc_metrics(self):
        # calculate surface forces from panel and finite volume method
        force_fuse_x, force_nac_x, force_tot_x = self.cumulate_forces_smooth()
        # calculate time averaged fuselage fan metrics from time averaged station data
        if self.case_type == 'Default':
            fpr, eta_ad, eta_pol, m_dot, ked_aip, ked_ff_in, ked_rot_in, ked_te, ked_wake, \
            p_shaft_is, p_shaft_act, net_thrust, gross_thrust, fan_force, pt2_pt0 = (
                self.calc_ff(no=np.arange(6000, 12000, 100), samples=10))
        elif self.case_type == 'Post':
            fpr, eta_ad, eta_pol, m_dot, ked_aip, ked_ff_in, ked_rot_in, ked_te, ked_wake, \
            p_shaft_is, p_shaft_act, net_thrust, gross_thrust, fan_force, pt2_pt0 = (
                self.calc_ff(no=np.arange(11000, 12100, 100), samples=1))
        # calculate net propulsive force and f_eta_bli from averaged characteristics
        npf = fan_force - force_tot_x
        f_eta_bli = self.calc_fetabli(npf, p_shaft_act)
        rad_dist = self.calc_rad_distortion('rotor_inlet', 12000)
        # add results to file of all study results
        result_rows = [self.casepath,
                       force_fuse_x, force_nac_x,
                       force_tot_x, pt2_pt0, fpr, eta_ad, eta_pol, m_dot, ked_aip, ked_ff_in, ked_rot_in, ked_te,
                       ked_wake, p_shaft_is, p_shaft_act, net_thrust, gross_thrust, fan_force, npf, f_eta_bli[0],
                       rad_dist]
        with open(f'{self.allresultspath}//results.csv', 'a') as resfile:
            writer = csv.writer(resfile)
            writer.writerow(result_rows)

    def cumulate_forces(self):
        # finite volume forces on rear part of fuselage-propulsor, incl. nacelle
        fvm_fuse_forces, fvm_nac_forces = \
            calc_forces(f'{self.casepath}', self.atmos, type='hybrid_method_post')

        # panel method forces on front part of fuselage
        pm_fuse_pressure, pm_fuse_viscous, pm_fuse_total = calc_ibl_forces(self.surface, self.potential_results,
                                                                           self.ibl_results, self.interface_loc,
                                                                           self.atmos)

        fuse_total_x = fvm_fuse_forces['total'][2] + pm_fuse_total
        nac_total_x = fvm_nac_forces['total'][2]
        total_x = fuse_total_x + nac_total_x

        return pm_fuse_total, fvm_fuse_forces['total'][0], fvm_nac_forces['total'][
            0], fuse_total_x, nac_total_x, total_x

    def cumulate_forces_smooth(self):
        # finite volume forces on fuselage and nacelle. cp and cf distribution between PM and FVM is smoothed
        tot_fuse_forces, fvm_nac_forces = \
            int_forces_smooth(f'{self.casepath}', self.atmos, self.surface, self.potential_results,
                              self.ibl_results, self.interface_loc, self.geometry, type='hybrid_method_post')

        fuse_total_x = tot_fuse_forces['total'][2]
        nac_total_x = fvm_nac_forces['total'][2]
        total_x = fuse_total_x + nac_total_x

        return fuse_total_x, nac_total_x, total_x

    # get mass flow averaged values at all defined fan stations
    def calc_ff(self, no, samples=10):
        m_dot = []
        fpr = []
        eta_ad = []
        eta_pol = []
        net_thrust = []
        gross_thrust = []
        kin_energy_defect = []
        ked_aip = []
        ked_ff_in = []
        ked_te = []
        ked_rot_in = []
        actual_shaft_power = []
        isentropic_shaft_power = []
        fan_force = []
        pt2_pt0 = []
        self.atmos.ext_props['gamma'] = 1.4
        station_names = ['bl_front', 'ff_inlet', 'ff_outlet', 'rotor_inlet', 'stator_outlet', 'bl_wake', 'bl_fuse_te']
        k = 0
        for i in no:
            # calc massflow averaged station values and kinetic energy defect
            self.mass_flow_ave_station_values, kin_en_defect = \
                calc_averaged_station_data(f'{self.casepath}//postProcessing//samples//{i}',
                                           station_names, 4., self.atmos, self.average)
            m_dot.append(self.mass_flow_ave_station_values['rotor_inlet']['mdot'])
            fpr.append(
                self.mass_flow_ave_station_values['stator_outlet']['pt_avg'] /
                self.mass_flow_ave_station_values['rotor_inlet']['pt_avg'])
            eta_pol.append(self.calc_eta_poly(fpr[-1]))
            eta_ad.append(self.calc_eta_ad(fpr[-1]))
            kin_energy_defect.append(kin_en_defect['bl_wake'])
            ked_aip.append(kin_en_defect['bl_front'])
            ked_ff_in.append(kin_en_defect['ff_inlet'])
            ked_te.append(kin_en_defect['bl_fuse_te'])
            ked_rot_in.append(kin_en_defect['rotor_inlet'])
            actual_shaft_power.append(self.calc_actual_power())
            isentropic_shaft_power.append(actual_shaft_power[-1] / eta_pol[-1])
            gross_thrust.append(self.calc_grossthrust())
            net_thrust.append(self.calc_netthrust(gross_thrust[-1]))
            fan_force.append(self.calc_fanforce())
            pt2_pt0.append(
                self.mass_flow_ave_station_values['rotor_inlet']['pt_avg'] /
                self.atmos.ext_props['p_t'])
            k += 1

        fpr_ave = calc_average(fpr, samples)
        eta_ad_ave = calc_average(eta_ad, samples)
        eta_pol_ave = calc_average(eta_pol, samples)
        m_dot_ave = calc_average(m_dot, samples)
        kin_energy_defect_ave = calc_average(kin_energy_defect, samples)
        ked_aip_ave = calc_average(ked_aip, samples)
        ked_ff_in_ave = calc_average(ked_ff_in, samples)
        ked_te_ave = calc_average(ked_te, samples)
        ked_rot_in_ave = calc_average(ked_rot_in, samples)
        p_shaft_is_ave = calc_average(isentropic_shaft_power, samples)
        p_shaft_act_ave = calc_average(actual_shaft_power, samples)
        net_thrust_ave = calc_average(net_thrust, samples)
        gross_thrust_ave = calc_average(gross_thrust, samples)
        fan_force_ave = calc_average(fan_force, samples)
        pt2_pt0_ave = calc_average(pt2_pt0, samples)

        return fpr_ave, eta_ad_ave, eta_pol_ave, m_dot_ave, ked_aip_ave, ked_ff_in_ave, ked_rot_in_ave, ked_te_ave, \
               kin_energy_defect_ave, p_shaft_is_ave, p_shaft_act_ave, net_thrust_ave, gross_thrust_ave, fan_force_ave, \
               pt2_pt0_ave

    # actual, i.e. polytropic shaft power [W]
    def calc_actual_power(self):
        c_p = self.atmos.ext_props['c_p']
        P = self.mass_flow_ave_station_values['rotor_inlet']['mdot'] * c_p * (
                self.mass_flow_ave_station_values['stator_outlet']['T_avg'] -
                self.mass_flow_ave_station_values['rotor_inlet']['T_avg'])
        return P

    # polytropic fan efficiency [-]
    def calc_eta_poly(self, fpr):
        kappa = self.atmos.ext_props['gamma']
        eta_poly = (kappa - 1) / kappa * (
                np.log(fpr) / np.log(self.mass_flow_ave_station_values['stator_outlet']['Tt_avg'] /
                                     self.mass_flow_ave_station_values['rotor_inlet']['Tt_avg']))
        return eta_poly

    # adiabatic fan efficiency [-]
    def calc_eta_ad(self, fpr):
        gamma = self.atmos.ext_props['gamma']
        eta_ad = ((fpr) ** ((gamma - 1) / gamma) - 1) / \
                 ((self.mass_flow_ave_station_values['stator_outlet']['Tt_avg'] /
                   self.mass_flow_ave_station_values['rotor_inlet']['Tt_avg']) - 1)
        return eta_ad

    # gross thrust [N]
    def calc_grossthrust(self):
        T_g = self.mass_flow_ave_station_values['rotor_inlet']['mdot'] * self.mass_flow_ave_station_values['ff_outlet'][
            'U_avg'] + \
              self.geometry['A_18'] * (self.mass_flow_ave_station_values['ff_outlet']['p_avg'] - self.atmos.pressure)
        return T_g

    # net thrust [N]
    def calc_netthrust(self, T_g):
        T_n = T_g - self.mass_flow_ave_station_values['rotor_inlet']['mdot'] * self.atmos.ext_props['u']
        return T_n

    # fan volume force [N]
    def calc_fanforce(self):
        # get fan volume force from momentum source terms
        with open(f'{self.casepath}//integrated_data.csv', 'r') as int_data:
            csv_reader = csv.reader(int_data)
            header = next(csv_reader)
            data = []
            for row in csv_reader:
                record = {}
                for i, value in enumerate(row[:-1]):
                    record[header[i]] = float(value)
                data.append(record)
        F_f = float(data[0]['F:0']) * 80
        # F_f = self.mass_flow_ave_station_values['rotor_inlet']['mdot'] * (
        #                   self.mass_flow_ave_station_values['stator_outlet']['U_avg'] -
        #                   self.mass_flow_ave_station_values['rotor_inlet']['U_avg']) + \
        #       self.geometry['A_13'] * self.mass_flow_ave_station_values['stator_outlet']['p_avg'] - \
        #       self.geometry['A_12'] * self.mass_flow_ave_station_values['rotor_inlet']['p_avg']
        return F_f

    # calculate BLI fan efficiency factor
    def calc_fetabli(self, npf, p_shaft):
        return npf * self.atmos.ext_props['u'] / p_shaft

    def calc_ff_metrics_post_sim(self, casepath_orig, orig_res_path, new_res_path, samples=1):
        no = np.arange(11000, 12100, 100)
        m_dot = []
        fpr = []
        eta_ad = []
        eta_pol = []
        net_thrust = []
        gross_thrust = []
        kin_energy_defect = []
        actual_shaft_power = []
        isentropic_shaft_power = []
        fan_force = []
        pt2_pt0 = []
        ked_aip = []
        ked_ff_in = []
        ked_te = []
        ked_rot_in = []
        self.atmos.ext_props['gamma'] = 1.4
        station_names = ['bl_front', 'ff_inlet', 'ff_outlet', 'rotor_inlet', 'stator_outlet', 'bl_wake', 'bl_fuse_te']
        k = 0
        for i in no:
            # calc massflow averaged station values and kinetic energy defect
            self.mass_flow_ave_station_values, kin_en_defect = \
                calc_averaged_station_data(f'{self.casepath}//postProcessing//samples//{i}',
                                           station_names, 4., self.atmos, self.average)
            m_dot.append(self.mass_flow_ave_station_values['rotor_inlet']['mdot'])
            fpr.append(
                self.mass_flow_ave_station_values['stator_outlet']['pt_avg'] /
                self.mass_flow_ave_station_values['rotor_inlet']['pt_avg'])
            eta_pol.append(self.calc_eta_poly(fpr[-1]))
            eta_ad.append(self.calc_eta_ad(fpr[-1]))
            kin_energy_defect.append(kin_en_defect['bl_wake'])
            ked_aip.append(kin_en_defect['bl_front'])
            ked_ff_in.append(kin_en_defect['ff_inlet'])
            ked_te.append(kin_en_defect['bl_fuse_te'])
            ked_rot_in.append(kin_en_defect['rotor_inlet'])
            actual_shaft_power.append(self.calc_actual_power())
            isentropic_shaft_power.append(actual_shaft_power[-1] / eta_pol[-1])
            gross_thrust.append(self.calc_grossthrust())
            net_thrust.append(self.calc_netthrust(gross_thrust[-1]))
            fan_force.append(self.calc_fanforce())
            pt2_pt0.append(
                self.mass_flow_ave_station_values['rotor_inlet']['pt_avg'] /
                self.atmos.ext_props['p_t'][0])
            k += 1

        fpr_ave = calc_average(fpr, samples)
        eta_ad_ave = calc_average(eta_ad, samples)
        eta_pol_ave = calc_average(eta_pol, samples)
        m_dot_ave = calc_average(m_dot, samples)
        kin_energy_defect_ave = calc_average(kin_energy_defect, samples)
        ked_aip_ave = calc_average(ked_aip, samples)
        ked_ff_in_ave = calc_average(ked_ff_in, samples)
        ked_te_ave = calc_average(ked_te, samples)
        ked_rot_in_ave = calc_average(ked_rot_in, samples)
        p_shaft_is_ave = calc_average(isentropic_shaft_power, samples)
        p_shaft_act_ave = calc_average(actual_shaft_power, samples)
        net_thrust_ave = calc_average(net_thrust, samples)
        gross_thrust_ave = calc_average(gross_thrust, samples)
        fan_force_ave = calc_average(fan_force, samples)
        pt2_pt0_ave = calc_average(pt2_pt0, samples)

        with open(f'{orig_res_path}') as orig_data:
            csv_reader = csv.reader(orig_data)
            header = next(csv_reader)  # Read the header
            data = []
            for row in csv_reader:
                record = {}
                for i, value in enumerate(row):
                    record[header[i]] = value
                data.append(record)

        idx = next((idx for idx, d in enumerate(data) if d.get('case') == casepath_orig))

        npf = fan_force_ave - float(data[idx]['force_tot_x'])
        f_eta_bli = self.calc_fetabli(npf, p_shaft_act_ave)

        rad_dist = self.calc_rad_distortion('rotor_inlet', 12000)

        result_rows = [self.casepath, data[idx]['force_fuse_x'], data[idx]['force_nac_x'],
                       data[idx]['force_tot_x'], pt2_pt0_ave, fpr_ave, eta_ad_ave, eta_pol_ave, m_dot_ave, ked_aip_ave,
                       ked_ff_in_ave, ked_rot_in_ave, ked_te_ave, kin_energy_defect_ave, p_shaft_is_ave,
                       p_shaft_act_ave, net_thrust_ave, gross_thrust_ave, fan_force_ave, npf, f_eta_bli[0], rad_dist]
        with open(f'{new_res_path}', 'a') as resfile:
            writer = csv.writer(resfile)
            writer.writerow(result_rows)

        return fpr_ave, eta_ad_ave, eta_pol_ave, m_dot_ave, ked_aip_ave, ked_ff_in_ave, ked_rot_in_ave, ked_te_ave, \
               kin_energy_defect_ave, p_shaft_is_ave, p_shaft_act_ave, net_thrust_ave, gross_thrust_ave, fan_force_ave, \
               pt2_pt0_ave

    def calc_rad_distortion(self, facename, time):
        gamma = 1.4
        wedge_angle = 4
        face_ave_tot_pressure = self.mass_flow_ave_station_values[facename]['pt_avg']
        p = []
        z = []
        rhoU = []
        Ma = []
        with open(f'{self.casepath}//postProcessing//samples//{time}//{facename}_Ma_T_p_rho_U_rhoU.csv',
                  mode='r') as file:
            temp = csv.DictReader(file)
            for line in temp:
                rhoU.append(
                    np.sqrt(float(line['rhoU_0']) ** 2 + float(line['rhoU_1']) ** 2 + float(line['rhoU_2']) ** 2))
                z.append(float(line['z']))
                Ma.append(float(line['Ma']))
                p.append(float(line['p']))

        pt = [p[k] * (1 + (gamma - 1) / 2 * Ma[k] ** 2) ** (gamma / (gamma - 1)) for k in range(0, len(p))]

        cell_area = []
        mdot_cell = []

        for j in range(len(z)):
            if j == len(z) - 1:
                cell_height = 2 * (0.5 * (z[j] - z[j - 1]))
            else:
                cell_height = 2 * (0.5 * (z[j + 1] - z[j]))
            z_up = z[j] + 0.5 * cell_height
            z_low = z[j] - 0.5 * cell_height
            cell_area.append((z_up ** 2 - z_low ** 2) * np.pi / 360 * wedge_angle)
            mdot_cell.append(rhoU[j] * cell_area[j])

        ring_rad_distortion = []

        for i in pt:
            ring_rad_distortion.append((face_ave_tot_pressure - i) / face_ave_tot_pressure)

        face_rad_distortion = max([abs(i) for i in ring_rad_distortion])

        return face_rad_distortion


if __name__ == '__main__':
    from misc_functions.air_properties.create_atmos import create_atmos
    from geometry_generation.finite_volume_geometry.generate_fv_icst_geometry_sensitivity import GenerateGeomICST
    from interface.interface import Interface
    from panel.solve_potential_flow_pfc_hybrid import PotentialFlow
    from panel.solve_panel_method_pfc_hybrid import BoundaryLayerCalculation
    from geometry_generation.panel_geometry.prepare_fuselage_panels import sampleFuselageGeometry

    """CREATE SAMPLE"""
    individual_sims = [{'rmax': 1.415925, 'x_ff': 62.3785, 'r_cent_f': 3.045, 'l_cent_f': 50.9, 'f_xmax': 0.118,
                        'lnac': 2.53018, 'h_duct': 0.56, 'f_r12': 0.51003, 'f_lint': 0.3795, 'teta_f_cone': 13.59,
                        'f_rho_le': 1.0, 'f_l_nose': 0.1975, 'ahi_athr': 1.304, 'athr_a12': 0.994, 'a18_a13': 0.727,
                        'f_xthr': 0.225, 'delta_beta_te': 13, 'beta_te_low': 0, 'f_r18hub': 1.134, 'f_rthrtip': 0.898,
                        'teta_int_in': 10.5, 'teta_ff_in': 0, 'beta_ff_in': -4, 'mach_number': 0.82, 'altitude': 10680,
                        'omega_rot': 293}]
    i = 0
    Geometry = GenerateGeomICST(x_ff=individual_sims[i]['x_ff'], r_cent_f=individual_sims[i]['r_cent_f'],
                                l_cent_f=individual_sims[i]['l_cent_f'], rmax=individual_sims[i]['rmax'],
                                f_xmax=individual_sims[i]['f_xmax'], lnac=individual_sims[i]['lnac'],
                                h_duct=individual_sims[i]['h_duct'], f_r12=individual_sims[i]['f_r12'],
                                f_lint=individual_sims[i]['f_lint'],
                                teta_f_cone=individual_sims[i]['teta_f_cone'],
                                f_rho_le=individual_sims[i]['f_rho_le'],
                                f_l_nose=individual_sims[i]['f_l_nose'],
                                ahi_athr=individual_sims[i]['ahi_athr'],
                                athr_a12=individual_sims[i]['athr_a12'],
                                a18_a13=individual_sims[i]['a18_a13'],
                                f_xthr=individual_sims[i]['f_xthr'],
                                delta_beta_te=individual_sims[i]['delta_beta_te'],
                                beta_te_low=individual_sims[i]['beta_te_low'],
                                r_te_hub=0.0, f_r18hub=individual_sims[i]['f_r18hub'],
                                f_rthrtip=individual_sims[i]['f_rthrtip'],
                                teta_int_in=individual_sims[i]['teta_int_in'],
                                teta_ff_in=individual_sims[i]['teta_ff_in'],
                                beta_ff_in=individual_sims[i]['beta_ff_in'], plot=False, samplevars=False)

    fuselage, nacelle_top, nacelle_bottom, rotor_inlet, rotor_outlet, stator_inlet, stator_outlet, l_fuse, f_slr, tc_max, \
    tc_max_x, c_nac, i_nac, teta_f_aft, Astar_A2, _, x_12, x_13, rotor_le_coeffs, stator_le_coeffs, h_duct_in, h_duct_out, A_12, A_13, A_18 \
        = Geometry.build_geometry()

    geometry = {'fuselage': fuselage, 'nacelle_top': nacelle_top, 'nacelle_bottom': nacelle_bottom,
                'rotor_inlet': rotor_inlet, 'rotor_outlet': rotor_outlet, 'stator_inlet': stator_inlet,
                'stator_outlet': stator_outlet, 'l_fuse': l_fuse, 'r_fus': individual_sims[i]['r_cent_f'],
                'l_cent_f': individual_sims[i]['l_cent_f'], 'h_duct_in': h_duct_in,
                'h_duct_out': h_duct_out, 'teta_int_in': individual_sims[i]['teta_int_in'],
                'tc_max_x': tc_max_x, 'beta_te_low': individual_sims[i]['beta_te_low'],
                'beta_te_up': individual_sims[i]['beta_te_low'] + individual_sims[i]['delta_beta_te'],
                'A_12': A_12, 'A_13': A_13, 'A_18': A_18}
    X = [[fuselage[i][0] for i in range(0, len(fuselage))], [nacelle_top[i][0] for i in range(0, len(nacelle_top))],
         [nacelle_bottom[i][0] for i in reversed(range(0, len(nacelle_bottom)))]]
    Y = [[fuselage[i][1] for i in range(0, len(fuselage))], [nacelle_top[i][1] for i in range(0, len(nacelle_top))],
         [nacelle_bottom[i][1] for i in reversed(range(0, len(nacelle_bottom)))]]

    X_panel = [[fuselage[i][0] for i in range(0, len(fuselage))]]
    Y_panel = [[fuselage[i][1] for i in range(0, len(fuselage))]]

    atmos = create_atmos(individual_sims[i]['altitude'], individual_sims[i]['mach_number'], max(X[0]), 0)
    """PANEL METHOD"""
    # initialize potential flow solution without BL interaction
    flags = [15,  # 15: Case type
             0,  # Turbulence model: Patel
             1,  # 1: Transition model: Preston
             1,  # 1: Viscous-inviscid interaction model: equivalent body
             1,  # Compressibility correction model: Karman-Tsien
             1,  # Discretization strategy: parameterised sample points with weight w specified
             False,  # Elliptic cross section: no, axisymmetric
             0  # Viscous flow calculation: First, only inviscid
             ]

    counter = 0  # Counts number of interactions between viscid and inviscid flow
    eps = 1e-6  # Relative Tolerance for convergence of linear system and integration loops of turbulent calculations
    eps_2 = 5e-2  # Relative tolerance for convergence of inviscid and viscous interaction

    # use simplified fuselage geometry for panel code with open trailing edge
    Xn_fuse, Yn_fuse, Fm_fuse, arc_length_fuse = sampleFuselageGeometry(X_panel[0][0:-20], Y_panel[0][0:-20], 90, 0)

    fuselage_panel_pot_init = PotentialFlow([Xn_fuse], [Yn_fuse], [Fm_fuse], atmos, flags, 0)
    pot_init, surface_init, sigma_init, j_s_init, j_v_init = fuselage_panel_pot_init.calculate_potential_flow()

    pm_max_it = 20
    # calculate potential flow with IBL interaction
    flags[-1] = 1
    fuselage_panel_ibl = BoundaryLayerCalculation([Xn_fuse], [Yn_fuse], [Fm_fuse], [arc_length_fuse], atmos, flags,
                                                  counter,
                                                  # eps=eps, eps2=eps_2)#
                                                  pot_init, surface_init, sigma_init, j_s_init, j_v_init, eps, eps_2,
                                                  pm_max_it)
    pot_final, surface_final, sigma_final, j_s_final, j_v_final, bl_final, p_s, C_f, x_tr_rel, _, pm_conv = fuselage_panel_ibl.calculateIBL()

    interface_loc = 0.8
    h_domain_fv = 1
    casepath = '//home//anais//bli_hybrid_fv_gci//hybrid_fr_fine'
    allresultspath = '//home//anais//bli_hybrid_fv_gci//results'

    interface = Interface(bl_final, surface_init, j_s_final, sigma_final, pot_final, Xn_fuse, Yn_fuse, atmos,
                          individual_sims[i]['l_cent_f'], h_domain_fv)
    interface.interface_location(interface_loc)

    post = PostProcess(atmos, geometry, surface_final, pot_final, bl_final, interface, casepath, allresultspath)
    post.calc_metrics()
