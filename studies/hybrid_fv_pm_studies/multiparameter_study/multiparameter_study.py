import os
import shutil
import csv
import subprocess
import platform

from misc_functions.air_properties.create_atmos import create_atmos
from geometry_generation.finite_volume_geometry.generate_fv_icst_geometry import GenerateGeomICST
from geometry_generation.panel_geometry.prepare_fuselage_panels import sampleFuselageGeometry_refnose
from finite_volume.open_foam.mesh_preparation.generate_mesh_hybrid import generate_mesh_and_blade_data
from interface.interface import Interface
from finite_volume.open_foam.setup_cases.prepare_openfoam_case import PrepareOFCase
from finite_volume.open_foam.setup_cases.run_openfoam_simulation import run_sim
from post_processing.hybrid_method_post.postprocess import PostProcess
from hybrid_method.automate_pm import process_pfchybrid_ibl

os_name = platform.system()

if os_name == 'Windows':
    sample_path = "..//..//..//sample_generation//doe_generation_data//individual_samples.csv"
    parent_path = "C://Users//anais.habermann//Desktop//Sensitivity_Study_2"
    default_case_path = "..//..//..//hybrid_method//default_fv_case//fully_resolved"
    pathtopvpython = "C:/Program Files/ParaView 5.8.1-Windows-Python3.7-msvc2015-64bit/bin/pvpython.exe"
    pathtoparaviewscript = 'C://Users//anais.habermann//Documents//axibl//finite_volume_post\openFoam\postProcess\paraview_cp_cf.py'
    pathtoparaviewscript2 = 'C://Users//anais.habermann//Documents//axibl//finite_volume_post//open_foam//post_processing//paraview_int_forces.py'
elif os_name == 'Linux':
    sample_path = "..//..//..//sample_generation//doe_generation_data//individual_samples.csv"
    parent_path = "//home//anais//Sensitivity_Study_2"
    default_case_path = "..//..//..//hybrid_method//default_fv_case//fully_resolved"
    pathtopvpython = "//home//anais//Paraview5.8-mesa//bin//pvpython"#"/usr/bin/pvpython"
    pathtoparaviewscript = '//home//anais//axibl//finite_volume_post//open_foam//post_processing//paraview_cp_cf_linux.py'
    pathtoparaviewscript2 = '//home//anais//axibl//finite_volume_post//open_foam//post_processing//paraview_int_forces.py'
else:
    raise Exception('For this OS system paths are not configured.')

# interface as ratio of (fuselage nose + center section) length
interface_loc = 0.8
# max. no. of iterations for viscous-inviscid interaction of panel method
pm_max_it = 20

if not os.path.exists(parent_path):
    os.makedirs(parent_path)

with open(f"{parent_path}//study.log", "w", newline="") as logfile:
    """CREATE FILE FOR CUMULATED RESULTS"""
    allresultspath = os.path.join(parent_path, "results")
    if not os.path.exists(allresultspath):
        os.makedirs(allresultspath)
    resultkeys =['case', 'force_fuse_x', 'force_nac_x',
                 'force_tot_x', 'pt2_pt0', 'fpr', 'eta_ad', 'eta_pol', 'm_dot', 'ked_aip', 'ked_ff_in', 'ked_rot_in',
                 'ked_te', 'ked_wake', 'p_shaft_is', 'p_shaft_act', 'net_thrust', 'gross_thrust', 'fan_force', 'npf',
                 'f_eta_bli','rad_dist']
    with open(f'{allresultspath}//results.csv','w') as resfile:
        writergeo = csv.writer(resfile)
        writergeo.writerow(resultkeys)

    """IMPORT SAMPLE"""
    with open(sample_path, 'r') as multidim_sample:
        csv_reader = csv.reader(multidim_sample)
        header = next(csv_reader)
        individual_sims = []
        for row in csv_reader:
            record = {}
            for i, value in enumerate(row[:-1]):
                record[header[i]] = float(value)
            individual_sims.append(record)

    for i in range(0, len(individual_sims)):
        """COPY PARENT FOLDER TO CASE FOLDER"""
        casepath = os.path.join(parent_path, str(i))
        if os.path.exists(casepath):
            shutil.rmtree(casepath)
        shutil.copytree(default_case_path, casepath)

        samplevarspath = os.path.join(casepath, 'system//include')
        if not os.path.exists(samplevarspath):
            os.makedirs(samplevarspath)

        """GEOMETRY"""
        try:
            Geometry = GenerateGeomICST(f_x_ff=individual_sims[i]['f_x_ff'], r_cent_f=individual_sims[i]['r_cent_f'],
                                        l_cent_f=individual_sims[i]['l_cent_f'], f_rmax=individual_sims[i]['f_rmax'],
                                        f_xmax=individual_sims[i]['f_xmax'], f_lnac=individual_sims[i]['f_lnac'],
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
                                        beta_ff_in=individual_sims[i]['beta_ff_in'], plot=False, samplevars=True,
                                        savepath=samplevarspath)

            fuselage, nacelle_top, nacelle_bottom, rotor_inlet, rotor_outlet, stator_inlet, stator_outlet, l_fuse, f_slr, tc_max, \
                tc_max_x, c_nac, i_nac, teta_f_aft, Astar_A2, _, x_12, x_13, rotor_le_coeffs, stator_le_coeffs, \
            h_duct_in, h_duct_out, A_12, A_13, A_18, r_12_tip, x_nac_max_tot, ar_nose, fuselage_panel \
                = Geometry.build_geometry()
            logfile.write(f'Geometry generation successful for index {i}.\n')
            logfile.flush()

            """SAVE GEOMETRY"""
            resultpath = os.path.join(casepath, "results")
            if not os.path.exists(resultpath):
                os.makedirs(resultpath)

            geomkeys = ['geotype', 'f_rmax', 'f_x_ff', 'r_cent_f', 'l_cent_f', 'f_xmax', 'f_lnac', 'h_duct', 'f_r12',
                        'f_lint',
                        'teta_f_cone', 'f_rho_le', 'f_l_nose', 'ahi_athr', 'athr_a12', 'a18_a13', 'f_xthr',
                        'delta_beta_te',
                        'beta_te_low', 'f_r18hub', 'f_rthrtip', 'teta_int_in', 'teta_ff_in', 'beta_ff_in',
                        'mach_number',
                        'altitude', 'omega_rot']
            geom_data_row = ['generateGeomBFM_ICST', individual_sims[i]['f_rmax'],individual_sims[i]['f_x_ff'],
                             individual_sims[i]['r_cent_f'], individual_sims[i]['l_cent_f'],
                             individual_sims[i]['f_xmax'], individual_sims[i]['f_lnac'],
                             individual_sims[i]['h_duct'], individual_sims[i]['f_r12'],
                             individual_sims[i]['f_lint'],
                             individual_sims[i]['teta_f_cone'],
                             individual_sims[i]['f_rho_le'],
                             individual_sims[i]['f_l_nose'],
                             individual_sims[i]['ahi_athr'],
                             individual_sims[i]['athr_a12'],
                             individual_sims[i]['a18_a13'],
                             individual_sims[i]['f_xthr'],
                             individual_sims[i]['delta_beta_te'],
                             individual_sims[i]['beta_te_low'],
                             individual_sims[i]['f_r18hub'],
                             individual_sims[i]['f_rthrtip'],
                             individual_sims[i]['teta_int_in'],
                             individual_sims[i]['teta_ff_in'],
                             individual_sims[i]['beta_ff_in'], individual_sims[i]['mach_number'],
                             individual_sims[i]['altitude'], individual_sims[i]['omega_rot']]

            with open(f"{resultpath}//geo_parameters.csv", 'w', newline="") as geofile1:
                writergeo1 = csv.writer(geofile1)
                writergeo1.writerow(geomkeys)
                writergeo1.writerow(geom_data_row)

        except Exception as a:
            print(f'Geometry cannot be generated for index {i}.')
            print(a)
            logfile.write(f'Geometry cannot be generated for index {i}.\n')
            logfile.write(str(a))
            logfile.flush()
            continue

        try:
            """FV MESH GENERATION"""
            geometry = {'fuselage': fuselage, 'nacelle_top': nacelle_top, 'nacelle_bottom': nacelle_bottom,
                        'rotor_inlet': rotor_inlet, 'rotor_outlet': rotor_outlet, 'stator_inlet': stator_inlet,
                        'stator_outlet': stator_outlet, 'l_fuse': l_fuse, 'r_fus': individual_sims[i]['r_cent_f'],
                        'l_cent_f': individual_sims[i]['l_cent_f'], 'h_duct_in': h_duct_in,
                        'h_duct_out': h_duct_out, 'teta_int_in': individual_sims[i]['teta_int_in'],
                        'tc_max_x': tc_max_x, 'beta_te_low': individual_sims[i]['beta_te_low'],
                        'beta_te_up': individual_sims[i]['beta_te_low'] + individual_sims[i]['delta_beta_te'],
                        'A_12': A_12, 'A_13': A_13, 'A_18': A_18, 'x_nac_max': x_nac_max_tot}

            _, h_domain_fv = generate_mesh_and_blade_data('fr', 'fine', individual_sims[i]['mach_number'],
                                                          individual_sims[i]['altitude'], geometry, interface_loc,
                                                          casepath, geotype='pfc')

            logfile.write(f'FV Mesh generation successful for index {i}.\n')
            logfile.flush()

        except Exception as b:
            print(f'FV Mesh cannot be generated for index {i}.')
            print(b)
            logfile.write(f'FV Mesh cannot be generated for index {i}.\n')
            logfile.write(str(b))
            logfile.flush()
            continue

        X = [[fuselage[i][0] for i in range(0, len(fuselage))], [nacelle_top[i][0] for i in range(0, len(nacelle_top))],
             [nacelle_bottom[i][0] for i in reversed(range(0, len(nacelle_bottom)))]]
        Y = [[fuselage[i][1] for i in range(0, len(fuselage))], [nacelle_top[i][1] for i in range(0, len(nacelle_top))],
             [nacelle_bottom[i][1] for i in reversed(range(0, len(nacelle_bottom)))]]

        X_panel = [[fuselage[i][0] for i in range(0, len(fuselage))]]
        Y_panel = [[fuselage[i][1] for i in range(0, len(fuselage))]]

        """ATMOSPHERE"""
        atmos = create_atmos(individual_sims[i]['altitude'], individual_sims[i]['mach_number'], max(X[0]), 0)

        try:
            """PANEL METHOD"""
            # initialize potential flow solution without BL interaction
            flags = [15,  # 15: Case type
                     0,  # Turbulence model: Patel
                     1,  # 1: Transition model: Preston
                     1,  # 1: Viscous-inviscid interaction model: equivalent body
                     1,  # 1: Compressibility correction model: Karman-Tsien
                     1,#1,  # Discretization strategy: parameterised sample points with weight w specified
                     False,  # Elliptic cross section: no, axisymmetric
                     0,  # Viscous flow calculation: First, only inviscid
                     0  # weight for calculation of BL characteristics at transition point. 0: fully Michel, 1: fully Preston
                     ]

            eps = 1e-6  # Relative Tolerance for convergence of linear system and integration loops of turbulent calculations
            eps_2 = 5e-2  # Relative tolerance for convergence of inviscid and viscous interaction

            # use simplified fuselage geometry for panel code with open trailing edge
            Xn_fuse, Yn_fuse, Fm_fuse, arc_length_fuse = sampleFuselageGeometry_refnose(X_panel[0][0:-20],
                                                                                        Y_panel[0][0:-20],90,0,
                                                                                        X_panel[0][99])

            pm_conv, surface_init, surface_final, j_s_final, sigma_final, pot_final, bl_final = \
                process_pfchybrid_ibl(Xn_fuse, Yn_fuse, Fm_fuse, arc_length_fuse, atmos, flags, eps, eps_2, pm_max_it,
                                      resultpath)

            if pm_conv is True:
                logfile.write(f'PM calculation succesful for index {i}.\n')
                logfile.flush()
            else:
                raise Exception("PM calculation did not converge.")

        except Exception as c:
            # if supersonic region at rear part of fuselage or if simulation did not converge due to any another reason,
            # run simulation again with simplified fuselage rear part
            if (len(c.args) > 1 and c.args[1] > 0.8 * individual_sims[i]['l_cent_f']) or len(c.args) == 1:
                cent_mask = [i <= Geometry.l_cent_f for i in Xn_fuse]
                idx_cent = [i for i in range(0, len(Xn_fuse)) if cent_mask[i] == True][-1]
                Xn_fuse_new = Xn_fuse[:idx_cent]
                Yn_fuse_new = Yn_fuse[:idx_cent]
                Fm_fuse_new = Fm_fuse[:idx_cent]
                trans_weights = [0, 0.05, 0.1, 0.3]
                for attempts in range(0, len(trans_weights)):
                    try:
                        flags[8] = trans_weights[attempts]
                        pm_conv, surface_init, surface_final, j_s_final, sigma_final, pot_final, bl_final = \
                            process_pfchybrid_ibl(Xn_fuse_new, Yn_fuse_new, Fm_fuse_new, arc_length_fuse, atmos, flags,
                                                  eps, eps_2, pm_max_it, resultpath)
                        if pm_conv is False:
                            raise Exception("PM calculation did not converge.")
                        else:
                            log_message = f'PM calculation of simplified geometry succesful with trans_weight ' \
                                          f'{trans_weights[attempts]} for index {i}.\n'
                            break
                    except Exception as c1:
                        log_message = f"PM calculation did not converge.\nPM calculation problem for simplified " \
                                      f"geometry for index {i}.\n{str(c1.args[0])}\n"
                logfile.write(log_message)
                logfile.flush()
            # if supersonic region near fuselage nose
            else:
                logfile.write(f'PM calculation problem for index {i}.\n')
                logfile.write(f'{str(c.args[0])}\n')
                logfile.flush()

        """ INTERFACE """
        # calculate b.l. characteristics at interface location
        interface = Interface(bl_final, surface_init, j_s_final, sigma_final, pot_final, Xn_fuse, Yn_fuse, atmos,
                              individual_sims[i]['l_cent_f'], h_domain_fv)
        interface.interface_location(interface_loc)
        interface.profiles(casepath)

        """PREPARE OPENFOAM CASE FOLDER"""
        # create/adapt controlvars, fvOptions and freestreamConditions files
        prepareOFCase = PrepareOFCase(atmos, individual_sims[i], max(X[0]), rotor_le_coeffs, stator_le_coeffs,
                                      casepath)
        prepareOFCase.prepare_files()

        try:
            """RUN SIMULATIONS"""
            run_sim(casepath, i)
            logfile.write(f'FVM calculation successful for index {i}.\n')
            logfile.flush()

        except Exception as d:
            print(f'FVM calculation did not converge for index {i}.')
            print(d)
            logfile.write(f'FVM calculation did not converge for index {i}.\n')
            logfile.write(str(d))
            logfile.flush()
            continue
        try:
            """POSTPROCESS"""
            # postprocess FV results with Paraview
            component = ['fuse_center', 'fuse_hub_gap', 'fuse_hub_inlet', 'fuse_hub_nozzle', 'fuse_hub_rotor',
                         'fuse_hub_stator', 'fuse_sweep', 'fuse_tail', 'nac_inlet', 'nac_rotor', 'nac_gap',
                         'nac_stator', 'nac_nozzle', 'nac_cowling']

            for k in component:
                para = subprocess.run(
                    f"{pathtopvpython} {pathtoparaviewscript} {atmos.pressure[0]} {atmos.ext_props['rho'][0]} "
                    f"{atmos.ext_props['u'][0]} {casepath} {k}", shell=True, text=True)
                print(para.stderr)

            para2 = subprocess.run(
                f"{pathtopvpython} {pathtoparaviewscript2} {casepath}", shell=True, text=True)
            print(para2.stderr)

            # calculate cumulated results of panel and finite volume method
            post = PostProcess(atmos, geometry, surface_final, pot_final, bl_final, interface, casepath,
                               allresultspath, case_type='Default')
            post.calc_metrics()
        except Exception as e:
            print('Some error with postprocessing')
            logfile.write(f'Post-processing error for index {i}.\n')
            logfile.write(str(e))
            logfile.flush()
            continue
