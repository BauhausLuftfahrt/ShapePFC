"""
Calculate boundary layer velocity profiles using a number of different methods.

Author:  A. Habermann

u_e         edge velocity [m/s]
delta       boundary layer thickness [m]
n           power law exponent
"""

import numpy as np
from scipy.optimize import fsolve

"""
Calculate boundary layer velocity profile at specific location acc. to specified function.

u_e         edge velocity [m/s]
delta       boundary layer thickness [m]
n           power law exponent
"""


def calc_velocity_profile(u_e: float, delta: float, r_0: float, C_f: float, nue: float, y: np.ndarray,
                          function_type: str = 'power_law',
                          n: float = None, delta_star: float = None, tau_w: float = None, dp_e_dx: float = None):
    kappa = 0.4  # v. Karman constant [2]
    C_plus = 5.1  # for smooth walls [2]
    u_tau = u_e * np.sqrt(C_f / 2)  # friction velocity
    y_wall = [i - r_0 for i in y]
    y_plus = [i * float(u_tau) / float(nue) for i in y_wall]  # dimensionless wall coordinate
    if function_type == 'power_law':  # whole BL
        u_prof = power_law_vel_profile(u_e, delta, n, y_wall)
        u_plus = [i / float(u_tau) for i in u_prof]
    elif function_type == 'spalding':  # whole BL
        u_plus = spalding_vel_profile(y_plus, kappa, C_plus, u_tau)
    elif function_type == 'coles_logwall_edge':  # y+ >= 30
        u_plus = coles_vel_profile(delta_star, tau_w, dp_e_dx, delta, y_plus, y_wall, nue, u_e, u_tau, kappa,
                                   C_plus, law_of_the_wall_option='log', wake_parameter_option='edge')
    elif function_type == 'coles_logwall_clauser':  # y+ >= 30
        u_plus = coles_vel_profile(delta_star, tau_w, dp_e_dx, delta, y_plus, y_wall, nue, u_e, u_tau, kappa,
                                   C_plus, law_of_the_wall_option='log', wake_parameter_option='Clauser')
    elif function_type == 'coles_spalding_edge':  # whole BL
        u_plus = coles_vel_profile(delta_star, tau_w, dp_e_dx, delta, y_plus, y_wall, nue, u_e, u_tau, kappa,
                                   C_plus, law_of_the_wall_option='spalding', wake_parameter_option='edge')
    elif function_type == 'coles_spalding_Clauser':  # whole BL
        u_plus = coles_vel_profile(delta_star, tau_w, dp_e_dx, delta, y_plus, y_wall, nue, u_e, u_tau, kappa,
                                   C_plus, law_of_the_wall_option='spalding', wake_parameter_option='Clauser')
    elif function_type == 'granville_logwall':  # y+ >= 30
        u_plus = granville_vel_profile(delta, y_plus, y_wall, nue, u_e, u_tau, C_f, kappa, C_plus,
                                       law_of_the_wall_option='log')
    elif function_type == 'granville_spalding':  # whole BL
        u_plus = granville_vel_profile(delta, y_plus, y_wall, nue, u_e, u_tau, C_f, kappa, C_plus,
                                       law_of_the_wall_option='spalding')
    elif function_type == 'thompson':  # whole BL
        u_plus = thompson_vel_profile(delta, y_plus, y_wall, nue, u_e, u_tau, kappa, C_plus)
    else:
        raise ValueError('Wrong velocity profile option prescribed.')
    if function_type != 'power_law':
        u_prof = [i * float(u_tau) for i in u_plus]
    return u_prof, u_plus, y_plus


"""
Calculate power law function.

u_e         edge velocity [m/s]
delta       boundary layer thickness [m]
n           power law exponent
"""


def power_law_vel_profile(u_e: float, delta: float, n: float, y_wall: list):
    u = [u_e * (y_wall[i] / delta) ** (1 / n) for i in range(0, len(y_wall))]
    return u


"""
Calculate turbulent boundary layer velocity profile according to formula by Spalding.

Sources:    [1] Spalding 1961
            [2] Schlichting, Hermann; Gersten, K. (2000), Boundary-layer Theory (8th revisited.)

y:      y-coordinates for boundary layer evaluation
r_0:    body radius at x-coordinate
tau_w:  local shear stress in fluid at x-coordinate, assumed to be indepencent of y
rho:    fluid density
nue:    kinematic viscosity

"""


def spalding_vel_profile(y_plus: list, kappa: float, C_plus: float, u_tau: float):
    u_plus = [fsolve(spalding_equation, 100.0, args=(kappa, C_plus, float(y_plus[i])), xtol=1e-10, maxfev=1000)[0] for
              i in range(0, len(y_plus))]
    return u_plus


def spalding_equation(vars, *data):  # [1]
    kappa, C_plus, y_plus = data
    u_plus = vars
    f1 = u_plus + np.e ** (-kappa * C_plus) * (
                np.e ** (kappa * u_plus) - 1 - kappa * u_plus - 0.5 * (kappa * u_plus) ** 2 -
                (1 / 6) * (kappa * u_plus) ** 3 - (1 / 24) * (kappa * u_plus) ** 4) - y_plus
    return f1


"""
Calculate turbulent boundary layer velocity profile according to formula by Coles. Applicable to flows with and without 
pressure gradients. 

law_of_the_wall:        defines, which law of the wall function is employed. Default: 'log'
                        log - valid only for y+ >= 30
                        spalding - valid for whole region

wake_parameter_option:  defines, which method is employed to calculate the wake parameter/wake strength. Default: 'edge'
                        options: edge, Clauser
"""


def coles_vel_profile(delta_star: float, tau_w: float, dp_e_dx: float, delta: float, y_plus: list,
                      y_wall: list, nue, u_e, u_tau, kappa: float = 0.4, C_plus: float = 5.1,
                      law_of_the_wall_option: str = 'log', wake_parameter_option: str = 'edge'):
    if wake_parameter_option == 'edge':
        C_coles = coles_coeff_edge(u_e, u_tau, kappa, delta, nue, C_plus)
    elif wake_parameter_option == 'Clauser':
        C_coles = fsolve(coles_coeff_Clauser, 0.45, args=(delta_star, tau_w, dp_e_dx), xtol=1e-10, maxfev=1000)
    if law_of_the_wall_option == 'log':
        C_1 = log_law_of_the_wall(kappa, y_plus, C_plus)
    elif law_of_the_wall_option == 'spalding':
        C_1 = spalding_vel_profile(y_plus, kappa, C_plus, u_tau)
    C_2 = law_of_the_wake(kappa, C_coles, y_wall, delta)
    u_plus = [C_1[i] + C_2[i] for i in range(0, len(y_plus))]
    return u_plus


"""
Calculation of Coles wake parameter/wake strength according to White 1991, p. 451, using the Clauser pressure gradient 
parameter beta = delta_star/tau_w*dp_e_dx = 0.42*C_cole**2+0.76*C_cole-0.4.
C_cole = 0.62 for dpe_dx = 0
"""


def coles_coeff_Clauser(vars, *data):
    delta_star, tau_w, dp_e_dx = data
    C_cole = vars
    f2 = 0.42 * C_cole ** 2 + 0.76 * C_cole - 0.4 - delta_star / tau_w * dp_e_dx
    return f2


"""
Calculation of Coles wake parameter/wake strength according to Cebeci 2013, p. 118, equ. 4.4.37, at the edge of the 
boundary layer.
"""


def coles_coeff_edge(u_e, u_tau, kappa, delta, nue, C_plus):
    return (u_e / u_tau - (1 / kappa) * np.log(delta * u_tau / nue) - C_plus) * kappa / 2


"""
Calculation of the log law of the wall function.
"""


def log_law_of_the_wall(kappa, y_plus, C_plus):
    y_plus = [1 if i == 0 else i for i in y_plus]
    return 1 / kappa * np.log(y_plus) + C_plus


"""
Calculation of the law of the wake acc. to Coles.
"""


def law_of_the_wake(kappa, C_coles, y_wall, delta):
    return [2 * float(C_coles) / kappa * (np.sin(np.pi / 2 * i / delta)) ** 2 for i in y_wall]


"""
Calculate turbulent boundary layer velocity profile according to formula by Granville. Applicable to flows with and without 
pressure gradients. Modification of Coles velocity profile function to ensure that du_dy=0 at y=delta.

law_of_the_wall:        defines, which law of the wall function is employed. Default: 'log'
                        log - valid only for y+ >= 30
                        spalding - valid for whole region
"""


def granville_vel_profile(delta: float, y_plus: list, y_wall: list, nue, u_e, u_tau, C_f, kappa: float = 0.4,
                          C_plus: float = 5.1, law_of_the_wall_option: str = 'log'):
    C_coles = granville_coeff_edge(u_e, u_tau, kappa, delta, nue, C_plus)
    if law_of_the_wall_option == 'log':
        C_1 = log_law_of_the_wall(kappa, y_plus, C_plus)
    elif law_of_the_wall_option == 'spalding':
        C_1 = spalding_vel_profile(y_plus, kappa, C_plus, u_tau)
    C_2 = mod_law_of_the_wake(kappa, C_coles, y_wall, delta)
    u_plus = [C_1[i] + C_2[i] for i in range(0, len(y_plus))]
    return u_plus


"""
Calculation of the law of the wake acc. to Granville.
"""


def mod_law_of_the_wake(kappa, C_coles, y_wall, delta):
    return [1 / kappa * (C_coles[0] * (1 - np.cos(np.pi * i / delta) + ((i / delta) ** 2 - i / delta) ** 3)) for i in
            y_wall]


"""
Calculation of Granville wake parameter/wake strength according to Cebeci 2013, p. 119, equ. 4.4.40, at the edge of the 
boundary layer.
"""


def granville_coeff_edge(u_e, u_tau, kappa, delta, nue, C_plus):
    return ((u_e / u_tau - C_plus) * kappa - np.log(delta * u_tau / nue)) / 2


"""
Calculate turbulent boundary layer velocity profile according to formula by Thomspon. Applicable to flows with and without 
pressure gradients. Valid for the whole boundary layer. Extension of Coles formula using log law of the wall.
"""


def thompson_vel_profile(delta: float, y_plus: list, y_wall: list, nue, u_e, u_tau, kappa: float = 0.4,
                         C_plus: float = 5.1):
    u_plus = np.zeros(len(y_plus))
    y_plus = np.array(y_plus)
    C_coles = coles_coeff_edge(u_e, u_tau, kappa, delta, nue, C_plus)
    C_1 = np.array(log_law_of_the_wall(kappa, y_plus, C_plus))
    C_2 = np.array(mod_law_of_the_wake(kappa, C_coles, y_wall, delta))
    inner = np.where(y_plus < 4)
    mid = np.where(np.logical_and(4 <= y_plus, y_plus <= 30))
    outer = np.where(y_plus > 30)
    u_plus[outer] = C_1[outer] + C_2[outer]
    u_plus[inner] = y_plus[inner]
    u_plus[mid] = 1.0828 - 0.414 * np.log(y_plus[mid]) + 2.2661 * np.log(y_plus[mid]) ** 2 - 0.324 * np.log(
        y_plus[mid]) ** 3
    return u_plus


# McLean 2013, Equ. 4.6.7
# approximation of Temperature profile for laminar and turbulent boundary layers with adiabatic wall
def calc_temperature_profile(u_prof: np.ndarray, u_edge: float, M_edge: float, T_edge: float):
    rec = 0.89  # for air, McLean 2013, Equ. 4.6.6, Pr = 0.7
    gamma = 1.4
    return [T_edge * (1 + (gamma - 1) / 2 * rec * M_edge ** 2 * (1 - i ** 2 / u_edge ** 2)) for i in u_prof]
