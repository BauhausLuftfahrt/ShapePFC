"""Calculate the pressure and skin friction forces of fuselage and nacelle of a PFC geometry from panel method skin
friction coefficient and pressure coefficient results.

Author:  A. Habermann
"""

import numpy as np
from post_processing.panel.plot.drag_computation import dragBody
from scipy import interpolate


def calc_ibl_forces(surface, pot, ibl, int, atmos):
    C_f = ibl[9]
    Xs_in = surface[0][0]
    p_s = ibl[10]
    Xn = surface[0][4]
    Yn = surface[0][5]

    # Compute pressure at stagnation point
    Cp_i = 1  # Bernoulli incompressible
    Cp = Cp_i / ((1 - atmos.ext_props['mach'] ** 2) ** 0.5 + (atmos.ext_props['mach'] ** 2) *
                 (Cp_i / 2) / (1 + (1 - atmos.ext_props['mach'] ** 2) ** 0.5))  # (Karman-Tsien)
    p_stag = Cp * (0.5*atmos.ext_props['rho'][0]*atmos.ext_props['u'][0]**2) + atmos.pressure[0] # Static pressure in stagnation point

    idx_preint = np.where(Xs_in < int.x_int)[0][-1]
    x_preint = Xs_in[idx_preint]
    x_postint = Xs_in[idx_preint+1]
    Xs = np.append(Xs_in[:idx_preint],int.x_int)
    Xs_2 = np.insert(Xs,0,0.0)
    Xn_new = np.linspace(min(Xn),int.x_int,1000)
    Xs_new = np.array([Xn_new[i]+0.5*(Xn_new[i+1]-Xn_new[i]) for i in range(0,len(Xn_new)-1)])
    C_f_int = interp(x_preint, C_f[idx_preint], x_postint, C_f[idx_preint + 1], int.x_int)
    C_f = list(C_f[0:idx_preint])
    C_f.append(C_f_int)
    C_f_fun = interpolate.UnivariateSpline(Xs, C_f, s=0)
    C_f_new = C_f_fun(Xs_new)
    p_s_int = interp(x_preint, p_s[idx_preint], x_postint, p_s[idx_preint + 1], int.x_int)
    p_s = list(p_s[:idx_preint])
    p_s.append(p_s_int)
    p_s_add = np.insert(p_s,0,p_stag)
    p_s_fun = interpolate.UnivariateSpline(Xs_2, p_s_add, s=0)
    p_s_new = p_s_fun(Xs_new)
    Xn_int = interp(x_preint, Xn[idx_preint+1], x_postint, Xn[idx_preint + 2], int.x_int)
    Xn = list(Xn[:idx_preint+1])
    Xn.append(Xn_int)
    Yn_int = interp(x_preint, Yn[idx_preint+1], x_postint, Yn[idx_preint + 2], int.x_int)
    Yn = list(Yn[:idx_preint+1])
    Yn.append(Yn_int)
    Yn_fun = interpolate.UnivariateSpline(Xn, Yn, s=0)
    Yn_new = Yn_fun(Xn_new)

    alpha_new = [np.arctan((Yn_new[i+1]-Yn_new[i])/(Xn_new[i+1]-Xn_new[i])) for i in range(0,len(Xn_new)-1)]

    dA_new = [np.pi*(Yn_new[i+1]+Yn_new[i])*np.sqrt((Xn_new[i+1]-Xn_new[i])**2+(Yn_new[i+1]-Yn_new[i])**2) for i in range(0,len(Xn_new)-1)]

    ff, fp = dragBody(atmos.ext_props['rho'], atmos.ext_props['u'], atmos.pressure[0], alpha_new, dA_new, C_p=None, p_s=p_s_new,
                      C_f=C_f_new, tau=None)

    ft = fp+ff

    return fp, ff, ft


def interp(x0,y0,x1,y1,x_int):
    return y0+(y1-y0)/(x1-x0)*(x_int-x0)