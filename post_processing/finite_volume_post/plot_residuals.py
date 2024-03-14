"""Plot the residuals of a finite volume simulation.

Author:  A. Habermann
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_res(path, start):
    res = []
    time = []
    with open(f'{path}/log', 'r') as f:
        for l_no, line in enumerate(f):
            if 'GMRES iteration: 0' in line:
                res.append((line.replace('(', '').replace(')', '')).split())
            if 'Time =' in line:
                time.append(line.split())
    t = [int(i[2]) for i in time[start::2]]
    rho_res = [float(i[4]) for i in res]
    rhoUx_res = [float(i[5]) for i in res]
    rhoUy_res = [float(i[6]) for i in res]
    rhoUz_res = [float(i[7]) for i in res]
    rhoE_res = [float(i[8]) for i in res]
    dellen = len(rho_res) - len(t)
    rho_res = rho_res[dellen:]
    rhoUx_res = rhoUx_res[dellen:]
    rhoUy_res = rhoUy_res[dellen:]
    rhoUz_res = rhoUz_res[dellen:]
    rhoE_res = rhoE_res[dellen:]

    plt.plot(t, rho_res, label=r'$\rho$')
    plt.plot(t, rhoUx_res, label=r'$\rho Ux$')
    plt.plot(t, rhoUy_res, label=r'$\rho Uy$')
    plt.plot(t, rhoUz_res, label=r'$\rho Uz$')
    plt.plot(t, rhoE_res, label=r'$\rho E$')
    plt.legend()
    plt.yscale("log")
    # plt.ylim([-0.1,0.1])
    plt.show()


def plot_res_ext(path, start, res_error, ave_error, var_error):
    res = []
    time = []
    with open(f'{path}/log', 'r') as f:
        for l_no, line in enumerate(f):
            if 'GMRES iteration: 0' in line:
                res.append((line.replace('(', '').replace(')', '')).split())
            if 'Time =' in line:
                time.append(line.split())
    t = [int(i[2]) for i in time[start::2]]
    delta_t_rel = [(t[i] - t[i - 1]) / t[i - 1] for i in range(1, len(t))]
    delta_t_rel.insert(0, 1)

    rho_res = [float(i[4]) for i in res]
    rhoUx_res = [float(i[5]) for i in res]
    rhoUy_res = [float(i[6]) for i in res]
    rhoUz_res = [float(i[7]) for i in res]
    rhoE_res = [float(i[8]) for i in res]

    if len(rho_res) > len(t):
        dellen = len(rho_res) - len(t)
        rho_res = rho_res[dellen:]
        rhoUx_res = rhoUx_res[dellen:]
        rhoUy_res = rhoUy_res[dellen:]
        rhoUz_res = rhoUz_res[dellen:]
        rhoE_res = rhoE_res[dellen:]
    elif len(rho_res) < len(t):
        dellen = len(t) - len(rho_res)
        t = t[dellen:]

    delta_rho_rel = [(rho_res[i] - rho_res[i - 1]) for i in range(1, len(rho_res))]
    delta_rho_rel.insert(0, 1)
    delta_rhoUx_rel = [(rhoUx_res[i] - rhoUx_res[i - 1]) for i in range(1, len(rhoUx_res))]
    delta_rhoUx_rel.insert(0, 1)
    delta_rhoUy_rel = [(rhoUy_res[i] - rhoUy_res[i - 1]) for i in range(1, len(rhoUy_res))]
    delta_rhoUy_rel.insert(0, 1)
    delta_rhoUz_rel = [(rhoUz_res[i] - rhoUz_res[i - 1]) for i in range(1, len(rhoUz_res))]
    delta_rhoUz_rel.insert(0, 1)
    delta_rhoE_rel = [(rhoE_res[i] - rhoE_res[i - 1]) for i in range(1, len(rhoE_res))]
    delta_rhoE_rel.insert(0, 1)

    def rolling_mean(data, range):
        return np.convolve(data, np.ones(range) / range, mode='same')

    def variation_about_rolling_mean(data, range):
        mean = rolling_mean(data, range)
        variation = data - mean
        mean_variation = rolling_mean(variation, range)
        return mean, variation, mean_variation

    ave_rho, _, var_rho = variation_about_rolling_mean(rho_res, 200)
    ave_rhoUx, _, var_rhoUx = variation_about_rolling_mean(rhoUx_res, 200)
    ave_rhoUy, _, var_rhoUy = variation_about_rolling_mean(rhoUy_res, 200)
    ave_rhoUz, _, var_rhoUz = variation_about_rolling_mean(rhoUz_res, 200)
    ave_rhoE, _, var_rhoE = variation_about_rolling_mean(rhoE_res, 200)

    valrange = 5000
    try:
        idx_conv = next((i for i in range(len(t)) if all(res < res_error for res in rho_res[i - valrange:i])
                         and all(res < res_error for res in rhoUx_res[i - valrange:i])
                         and all(res < res_error for res in rhoUy_res[i - valrange:i])
                         and all(res < res_error for res in rhoUz_res[i - valrange:i])
                         and all(res < res_error for res in rhoE_res[i - valrange:i])
                         and all(
            val < ave_error for val in [ave_rho[i], ave_rhoUx[i], ave_rhoUy[i], ave_rhoUz[i], ave_rhoE[i]])
                         and all(
            val < var_error for val in [var_rho[i], var_rhoUx[i], var_rhoUy[i], var_rhoUz[i], var_rhoE[i]])))
    except:
        idx_conv = 0

    print(idx_conv)

    plt.plot(t, ave_rho)  # , label=r'$\rho$')
    plt.plot(t, ave_rhoUx)  # , label=r'$\rho Ux$')
    plt.plot(t, ave_rhoUy)  # , label=r'$\rho Uy$')
    plt.plot(t, ave_rhoUz)  # , label=r'$\rho Uz$')
    plt.plot(t, ave_rhoE)  # , label=r'$\rho E$')
    plt.plot(t, rho_res, label=r'$\rho$')
    plt.plot(t, rhoUx_res, label=r'$\rho Ux$')
    plt.plot(t, rhoUy_res, label=r'$\rho Uy$')
    plt.plot(t, rhoUz_res, label=r'$\rho Uz$')
    plt.plot(t, rhoE_res, label=r'$\rho E$')
    plt.plot([t[idx_conv], t[idx_conv]], [min(min(rho_res, rhoUx_res, rhoUy_res, rhoUz_res, rhoE_res)),
                                          max(max(rho_res, rhoUx_res, rhoUy_res, rhoUz_res, rhoE_res))], linestyle="-",
             color='k',
             label=f't=t{idx_conv}')
    plt.legend()
    plt.yscale("log")
    # plt.ylim([-0.1,0.1])
    plt.show()
