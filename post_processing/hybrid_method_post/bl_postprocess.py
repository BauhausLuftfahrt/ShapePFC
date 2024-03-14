"""Calculate boundary layer metrics from finite volume solution.

Author:  A. Habermann
"""

import numpy as np
import csv
from misc_functions.helpers.find_nearest import find_nearest_index


def get_station(dir, stationname, it_no):
    U = []
    y = []
    rho = []
    with open(dir + f'/postProcessing/sampleDict/{it_no}/{stationname}_U.csv', mode='r') as file:
        temp = csv.DictReader(file)
        for line in temp:
            U.append(np.sqrt(float(line['U_0'])**2+float(line['U_1'])**2+float(line['U_2'])**2))
            y.append(float(line['y']))
    with open(dir + f'/postProcessing/sampleDict/{it_no}/{stationname}_T_p_rho.csv', mode='r') as file:
        temp = csv.DictReader(file, delimiter=',')
        for line in temp:
            rho.append(float(line['rho']))

    bl = {'U': U, 'rho': rho, 'y': y}
    return bl


# area based kinetic energy defect acc. to Drela and ingested ratio of kinetic energy defect
def calc_kin_en_area_ratio(bl, atmos, y_hi):
    u_edge = 0.99*atmos.ext_props['u']          # this is NOT the edge velocity. It is integrated over the whole domain,
                                                # because the edge velocity cannot be determined here
    i = len(bl['U'].dropna().tolist())-1
    while bl['U'].dropna().tolist()[i] > u_edge:
        i -= 1
    y = [0]*len(bl['U'].dropna().tolist()[:i])
    y[0] = bl['y'].dropna().tolist()[0]
    rho = [0]*len(bl['U'].dropna().tolist()[:i])
    rho[0] = bl['rho'].dropna().tolist()[0]
    U = [0]*len(bl['U'].dropna().tolist()[:i])
    for j in range(1,len(bl['y'].dropna().tolist()[:i])):
        y[j] = (bl['y'].dropna().tolist()[j-1]+bl['y'].dropna().tolist()[j])/2
        rho[j] = (bl['rho'].dropna().tolist()[j-1]+bl['rho'].dropna().tolist()[j])/2
        U[j] = (bl['U'].dropna().tolist()[j-1]+bl['U'].dropna().tolist()[j])/2

    area = [y[i]**2*np.pi for i in range(len(U))]
    kin_en_area_int = [rho[i]*U[i]*(u_edge**2-U[i]**2) for i in range(len(U))]

    j = find_nearest_index(y, y_hi)

    # area based kin. en. defect [kg m/s^2]
    tot_kin_en_area_defect = np.trapz(kin_en_area_int, area, axis=0)
    ing_kin_en_area_defect = np.trapz(kin_en_area_int[:j], area[:j], axis=0)

    return tot_kin_en_area_defect, ing_kin_en_area_defect/tot_kin_en_area_defect


# area based kinetic energy defect acc. to Drela and ingested ratio of kinetic energy defect
def calc_kin_en_area_ratio_2(bl, atmos, y_hi, wedge_angle=4):
    u_edge = 0.99*atmos.ext_props['u']          # this is NOT the edge velocity. It is integrated over the whole domain,
                                                # because the edge velocity cannot be determined here
    i = len(bl['U'])-1
    while bl['U'][i] > u_edge:
        i -= 1
    y = bl['y'][:i]
    rho = bl['rho'][:i]
    U = bl['U'][:i]
    cell_area = []
    for j in range(0,len(bl['y'][:i])):
        if j == len(y)-1:
            cell_height = 2*(0.5*(y[j]-y[j-1]))
        else:
            cell_height = 2*(0.5*(y[j+1]-y[j]))
        y_up = y[j]+0.5*cell_height
        y_low = y[j]-0.5*cell_height
        cell_area.append((y_up**2-y_low**2)* np.pi / 360 * wedge_angle)

    kin_en_area_int = [rho[i]*U[i]*(u_edge**2-U[i]**2) for i in range(len(U))]

    j = find_nearest_index(y, y_hi)

    # area based kin. en. defect [kg m/s^2]
    tot_kin_en_area_defect = np.trapz(kin_en_area_int, cell_area, axis=0)
    ing_kin_en_area_defect = np.trapz(kin_en_area_int[:j], cell_area[:j], axis=0)

    # import matplotlib.pyplot as plt
    # plt.plot([i[0] for i in kin_en_area_int], y, markersize=0)
    # plt.show()

    return tot_kin_en_area_defect, ing_kin_en_area_defect/tot_kin_en_area_defect


# area based kinetic energy defect acc. to Drela and ingested ratio of kinetic energy defect
def calc_kin_en_area_ratio_3(bl, atmos, y_hi, wedge_angle=4):
    u_edge = 0.99*atmos.ext_props['u']          # this is NOT the edge velocity. It is integrated over the whole domain,
                                                # because the edge velocity cannot be determined here
    i = len(bl['U'])-1
    while bl['U'][i] > u_edge:
        i -= 1
    y = bl['z'][:i]
    rho = bl['rho'][:i]
    U = bl['U'][:i]
    cell_area = []
    for j in range(0,len(bl['z'][:i])):
        if j == len(y)-1:
            cell_height = 2*(0.5*(y[j]-y[j-1]))
        else:
            cell_height = 2*(0.5*(y[j+1]-y[j]))
        y_up = y[j]+0.5*cell_height
        y_low = y[j]-0.5*cell_height
        cell_area.append((y_up**2-y_low**2)* np.pi / 360 * wedge_angle)

    # kinetic energy area
    kin_en_area_int = [(rho[i]*U[i])*(u_edge**2-U[i]**2) for i in range(len(U))]

    j = find_nearest_index(y, y_hi)

    # area based kin. en. defect [kg m/s^2]
    tot_kin_en_area_defect = np.sum(np.dot([i[0] for i in kin_en_area_int], cell_area))
    ing_kin_en_area_defect = np.sum(np.dot([i[0] for i in kin_en_area_int[:j]], cell_area[:j]))

    return tot_kin_en_area_defect, ing_kin_en_area_defect/tot_kin_en_area_defect


# SAE International Recommended Practice Gas Turbine Engine Inlet Flow Distortion Guidelines 2017
def calc_rad_dist(dir, station, atmos, it_no, stationname):
    R = atmos.ext_props['R']
    kappa = atmos.ext_props['gamma']
    U = []
    T = []
    p = []
    with open(dir + f'/postProcessing/sampleDict/{it_no}/{stationname}_U.csv', mode='r') as file:
        temp = csv.DictReader(file)
        for line in temp:
            U.append(np.sqrt(float(line['U_0'])**2+float(line['U_1'])**2+float(line['U_2'])**2))
    with open(dir + f'/postProcessing/sampleDict/{it_no}/{stationname}_T_p_rho.csv', mode='r') as file:
        temp = csv.DictReader(file, delimiter=',')
        for line in temp:
            T.append(float(line['T']))
            p.append(float(line['p']))
    pfav = station['pt_avg']
    pav = [p[i] * pow((1 + (U[i]) ** 2 / (1.4 * R * T[i]) * (kappa - 1) / 2),
                     (kappa / (kappa - 1))) for i in range(0,len(p))]
    rad_dist = [pav[i]/pfav for i in range(0,len(p))]
    min_rad_dist = min(rad_dist)
    max_rad_dist = max(rad_dist)
    delta_rad_dist = abs(max_rad_dist-min_rad_dist)

    return rad_dist, min_rad_dist, max_rad_dist, delta_rad_dist


# area based momentum defect acc. to Drela and ingested ratio of kinetic energy defect
def calc_momentum_defect_area_ratio(bl, atmos, y_hi, wedge_angle=4):
    u_edge = 0.99*atmos.ext_props['u']          # this is NOT the edge velocity. It is integrated over the whole domain,
                                                # because the edge velocity cannot be determined here
    i = len(bl['U'])-1
    while bl['U'][i] > u_edge:
        i -= 1
    y = bl['z'][:i]
    rho = bl['rho'][:i]
    U = bl['U'][:i]
    cell_area = []
    for j in range(0,len(bl['z'][:i])):
        if j == len(y)-1:
            cell_height = 2*(0.5*(y[j]-y[j-1]))
        else:
            cell_height = 2*(0.5*(y[j+1]-y[j]))
        y_up = y[j]+0.5*cell_height
        y_low = y[j]-0.5*cell_height
        cell_area.append((y_up**2-y_low**2)* np.pi / 360 * wedge_angle)

    # kinetic energy area
    mom_area_int = [(rho[i]*U[i])*(u_edge-U[i]) for i in range(len(U))]

    j = find_nearest_index(y, y_hi)

    # area based mom. defect [kg m/s^2]
    tot_mom_area_defect = np.sum(np.dot([i[0] for i in mom_area_int], cell_area))
    ing_mom_area_defect = np.sum(np.dot([i[0] for i in mom_area_int[:j]], cell_area[:j]))

    return tot_mom_area_defect, ing_mom_area_defect/tot_mom_area_defect


def calc_wake_kin_en_excess(bl, atmos, wedge_angle=4):
    u_edge = 0.99*atmos.ext_props['u']          # this is NOT the edge velocity. It is integrated over the whole domain,
                                                # because the edge velocity cannot be determined here
    i = len(bl['U'])-1
    while bl['U'][i] > u_edge:
        i -= 1
    y = bl['z'][:i]
    rho = bl['rho'][:i]
    U = bl['U'][:i]
    cell_area = []
    for j in range(0,len(bl['z'][:i])):
        if j == len(y)-1:
            cell_height = 2*(0.5*(y[j]-y[j-1]))
        else:
            cell_height = 2*(0.5*(y[j+1]-y[j]))
        y_up = y[j]+0.5*cell_height
        y_low = y[j]-0.5*cell_height
        cell_area.append((y_up**2-y_low**2)* np.pi / 360 * wedge_angle)

    # wake kinetic energy excess
    wake_kin_en_excess_area_int = [(rho[i]*U[i])*(u_edge-U[i])**2 for i in range(len(U))]
    wake_kin_en_excess_area = np.sum(np.dot(wake_kin_en_excess_area_int, cell_area))

    return wake_kin_en_excess_area
