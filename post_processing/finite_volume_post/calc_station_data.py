"""Import and calculate the area- or mass flow-averaged data on the the different fuselage fan stations from finite
volume simualation results.

Author:  A. Habermann
"""

import csv
import numpy as np
from post_processing.hybrid_method_post.bl_postprocess import (calc_kin_en_area_ratio_3,
                                                               calc_momentum_defect_area_ratio,
                                                               calc_wake_kin_en_excess)


def calc_averaged_station_data(path, station_names, wedge_angle, atmos=None, average='mass_flow'):
    station_averages = {}
    R = 287.058
    gamma = 1.4
    tot_kin_en_area_defect = {}
    station_data = []

    for i in station_names:
        Ux = []
        Uy = []
        Uz = []
        U = []
        z = []
        T = []
        rho = []
        p = []
        rhoUx = []
        rhoUy = []
        rhoUz = []
        rhoU = []
        Ma = []

        with open(f'{path}/{i}_Ma_T_p_rho_U_rhoU.csv', mode='r') as file:
            temp = csv.DictReader(file)
            for line in temp:
                Ux.append(float(line['U_0']))
                Uy.append(float(line['U_1']))
                Uz.append(float(line['U_2']))
                U.append(np.sqrt(float(line['U_0']) ** 2 + float(line['U_1']) ** 2 + float(line['U_2']) ** 2))
                rhoUx.append(float(line['rhoU_0']))
                rhoUy.append(float(line['rhoU_1']))
                rhoUz.append(float(line['rhoU_2']))
                rhoU.append(
                    np.sqrt(float(line['rhoU_0']) ** 2 + float(line['rhoU_1']) ** 2 + float(line['rhoU_2']) ** 2))
                z.append(float(line['z']))
                Ma.append(float(line['Ma']))
                T.append(float(line['T']))
                p.append(float(line['p']))
                rho.append(float(line['rho']))

        station_data.append({'station': i, 'z': z, 'U': U, 'rho': rho})

        pt = [p[k] * (1 + (gamma - 1) / 2 * Ma[k] ** 2) ** (gamma / (gamma - 1)) for k in range(0, len(p))]
        Tt = [T[k] * (1 + (gamma - 1) / 2 * Ma[k] ** 2) for k in range(0, len(T))]

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
        if average == 'mass_flow':
            pt_average = np.dot(pt, mdot_cell) / np.sum(mdot_cell)
            p_average = np.dot(p, mdot_cell) / np.sum(mdot_cell)
            Tt_average = np.dot(Tt, mdot_cell) / np.sum(mdot_cell)
            T_average = np.dot(T, mdot_cell) / np.sum(mdot_cell)
            U_average = np.dot(U, mdot_cell) / np.sum(mdot_cell)
            Ux_average = np.dot(Ux, mdot_cell) / np.sum(mdot_cell)
            Ma_average = np.dot(Ma, mdot_cell) / np.sum(mdot_cell)
            rho_average = np.dot(rho, mdot_cell) / np.sum(mdot_cell)
        elif average == 'area':
            pt_average = np.dot(pt, cell_area) / np.sum(cell_area)
            p_average = np.dot(p, cell_area) / np.sum(cell_area)
            Tt_average = np.dot(Tt, cell_area) / np.sum(cell_area)
            T_average = np.dot(T, cell_area) / np.sum(cell_area)
            U_average = np.dot(U, cell_area) / np.sum(cell_area)
            Ux_average = np.dot(Ux, cell_area) / np.sum(cell_area)
            Ma_average = np.dot(Ma, cell_area) / np.sum(cell_area)
            rho_average = np.dot(rho, cell_area) / np.sum(cell_area)
        else:
            raise Warning('Specify mass_flow or area averaging.')

        mdot_total = np.sum(mdot_cell) * 360 / wedge_angle

        station_averages[i] = {'mdot': mdot_total, 'U_avg': U_average, 'Ux_avg': Ux_average, 'Ma_avg': Ma_average,
                               'p_avg': p_average, 'pt_avg': pt_average, 'T_avg': T_average, 'Tt_avg': Tt_average,
                               'rho_avg': rho_average}

        tot_kin_en, _ = calc_kin_en_area_ratio_3({'U': U, 'rho': rho, 'z': z}, atmos, 0.)
        tot_kin_en_area_defect[i] = tot_kin_en

    aip_data = next((item for item in station_data if item['station'] == 'bl_front'), None)
    inlet_data = next((item for item in station_data if item['station'] == 'ff_inlet'), None)
    wake_data = next((item for item in station_data if item['station'] == 'bl_wake'), None)
    y_hi = min(aip_data['z']) + (max(inlet_data['z']) - min(inlet_data['z']))

    _, ingested_kinetic_energy_defect = calc_kin_en_area_ratio_3(aip_data, atmos, y_hi)
    _, ingested_momentum_defect = calc_momentum_defect_area_ratio(aip_data, atmos, y_hi)
    wake_kinetic_energy_excess = calc_wake_kin_en_excess(wake_data, atmos)

    return (station_averages, tot_kin_en_area_defect, ingested_kinetic_energy_defect, ingested_momentum_defect,
            wake_kinetic_energy_excess)


def calc_residuals(var):
    residuals = []
    for i in range(0, len(var) - 1):
        residuals.append(np.abs(var[i + 1] / var[i] - 1))
    return residuals
