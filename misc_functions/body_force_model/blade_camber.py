"""Calculate and plot the camber of a rotor blade based on a specified number of cross-section of the blade defined
by x- and y-coordinates.

Author:  A. Habermann
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# use cm instead of inch for figure size
def cm2inch(value):
    return value / 2.54


def fun(x, a):
    return a * x * (x - 1)


def fun2(x, a, b, c):
    return a * x ** 2 + b * x + c


def plot_blade(top_orig_x, top_orig_y, bot_orig_x, bot_orig_y, camber_x, camber_y, fun_popt, name):
    # plt.style.use('./PhD.mplstyle')
    fig, ax = plt.subplots(figsize=(cm2inch(20), cm2inch(20)))
    ax.set(xlabel=r'x', ylabel=r'y', xlim=[0, 1.5], ylim=[-1, 1])
    ax.plot(top_orig_x, top_orig_y, label=r'Original blade', marker='None', color='k', linestyle='-')
    ax.plot(bot_orig_x, bot_orig_y, marker='None', color='k', linestyle='-')
    ax.plot(camber_x, camber_y, label=r'Camber rotated, normalized', marker='None')
    ax.plot(camber_x, fun(camber_x, *fun_popt), label=r'Approximation: y=' + str(fun_popt[0]) + '*x*(x-1)',
            marker='None')
    ax.legend()
    fig.savefig('Blade_' + str(name) + '.pdf')


def plot_camber(popt, pos, name):
    fig, ax = plt.subplots(figsize=(cm2inch(20), cm2inch(20)))
    ax.set(xlabel=r'x', ylabel=r'y', xlim=[0, 1], ylim=[-0.2, 0.2])
    x = np.linspace(0, 1, 20)
    for i in range(0, len(pos)):
        a = fun2(pos[i], *popt)
        ax.plot(x, fun(x, a), label=str(pos[i]), marker='None')
    ax.legend()
    fig.savefig('Camber_' + str(name) + '.pdf')


def plot_spanwise_camber(b, x, y, name):
    fig, ax = plt.subplots(figsize=(cm2inch(20), cm2inch(20)))
    plt.axis('equal')
    ax.set(xlabel=r'x', ylabel=r'y')
    for i in range(0, len(b)):
        ax.plot(x[i], y[i], label=str(b[i]), marker='None')
    ax.legend()
    fig.savefig('Camber_spanwise_' + str(name) + '.pdf')


def rotate(x_rot_centre, y_rot_centre, x, y, angle_rot):
    """Transform coordinates around a coordinate by given angle
    Author:  Anais Habermann
     Args:
        x_rot_centre, y_rot_centre  [m]     Coordinates of rotation center
        x, y  [m]                           Coordinates to be rotated
        angle_rot [rad]                     Rotation angle

    Returns:
        x_trans, y_trans            [m]     Transformed coordinates
    """
    x_trans = (x - x_rot_centre) * np.cos(angle_rot) + (y - y_rot_centre) * np.sin(angle_rot) + x_rot_centre
    y_trans = -(x - x_rot_centre) * np.sin(angle_rot) + (y - y_rot_centre) * np.cos(angle_rot) + y_rot_centre

    return x_trans, y_trans


def normalize(x, y):
    """Normalize coordinates
    Author:  Anais Habermann
     Args:
        x, y  [m]                           Coordinates to be normalized

    Returns:
        x_norm, y_norm           [m]     Normalized coordinates
    """
    x_norm = (x - min(x)) / (max(x) - min(x))
    y_norm = (y - min(x)) / (max(x) - min(x)) - (y[np.argmax(x)] - min(x)) / (max(x) - min(x))

    return x_norm, y_norm


def blade_camber(x_up, y_up, x_bot, y_bot):
    """Compute function to describe body_force_model of rotor and stator blades.
    Author:  Anais Habermann
     Args:
        x_up, y_up  [m]       Original coordinates of upper side of blade
        x_bot, y_bot  [m]       Original coordinates of lower side of blade

    Returns:
        x, y            [-]       Normalized and de-rotated x and y coordinates of local blade body_force_model
        a               [-]       Coefficient of function, which describes blade body_force_model y = a*x*(x-1)
        perr                      Standard deviation of curve fit
        rsme                      Root Mean Square Error for curve fit
    """

    # Interpolate
    x_int = np.linspace(min(min(x_up), min(x_bot)), max(max(x_up), max(x_bot)), 100)
    y_up_int = interpolate.interp1d(x_up, y_up, fill_value="extrapolate")
    y_bot_int = interpolate.interp1d(x_bot, y_bot, fill_value="extrapolate")

    # Calculate centreline of blade
    x_cent = x_int
    y_cent = (y_up_int(x_int) - y_bot_int(x_int)) / 2 + y_bot_int(x_int)

    idxnan = np.where(np.isnan(y_cent))
    y_cent = np.delete(y_cent, [idxnan], axis=0)
    x_cent = np.delete(x_cent, [idxnan], axis=0)

    # Approximate
    popt, pcov = curve_fit(fun, x_cent, y_cent)
    perr = np.sqrt(np.diag(pcov))  # standard deviation sigma
    rmse = np.mean((y_cent - fun(x_cent, *popt)) ** 2)

    return x_cent, y_cent, popt, perr, rmse


def blade_camber_norm(x_up, y_up, x_bot, y_bot):
    """Compute function to describe normalized and de-rotated body_force_model of rotor and stator blades for transonic BLI
    fuselage fan.
    Original data from Castillo Pardo & Hall 2020, Fig. 5
    Author:  Anais Habermann
     Args:
        x_up, y_up  [m]       Original coordinates of upper side of blade
        x_bot, y_bot  [m]       Original coordinates of lower side of blade

    Returns:
        x, y            [-]       Normalized and de-rotated x and y coordinates of local blade body_force_model
        a               [-]       Coefficient of function, which describes blade body_force_model y = a*x*(x-1)
        perr                      Standard deviation of curve fit
        rsme                      Root Mean Square Error for curve fit
    """

    # Interpolate
    x_int = np.linspace(min(min(x_up), min(x_bot)), max(max(x_up), max(x_bot)), 50)
    y_up_int = interpolate.interp1d(x_up, y_up, fill_value="extrapolate")
    y_bot_int = interpolate.interp1d(x_bot, y_bot, fill_value="extrapolate")

    # Calculate centreline of blade
    x_cent = x_int
    y_cent = (y_up_int(x_int) - y_bot_int(x_int)) / 2 + y_bot_int(x_int)

    # Calculate rotation angle
    ang_rot = np.arctan((y_cent[np.argmax(x_cent)] - y_cent[np.argmin(x_cent)]) / (max(x_cent) - min(x_cent)))

    # Rotate centreline about leading edge of blade
    x_le = min(x_cent)
    y_le = y_cent[np.argmin(x_cent)]
    x_rot, y_rot = rotate(x_le, y_le, x_cent, y_cent, ang_rot)

    # Normalize
    x, y = normalize(x_rot, y_rot)

    # Approximate
    popt, pcov = curve_fit(fun, x, y)
    perr = np.sqrt(np.diag(pcov))  # standard deviation sigma
    rmse = np.mean((y - fun(x, *popt)) ** 2)

    return x, y, ang_rot, popt, perr, rmse


def blade_norm(x_up, y_up, x_bot, y_bot):
    """Compute function to describe normalized and de-rotated top and bottom side of rotor and stator blades for transonic BLI
    fuselage fan.
    Original data from Castillo Pardo & Hall 2020, Fig. 5
    Author:  Anais Habermann
     Args:
        x_up, y_up  [m]       Original coordinates of upper side of blade
        x_bot, y_bot  [m]       Original coordinates of lower side of blade

    Returns:
        x, y            [-]       Normalized and de-rotated x and y coordinates of local blade body_force_model
        a               [-]       Coefficient of function, which describes blade body_force_model y = a*x*(x-1)
        perr                      Standard deviation of curve fit
        rsme                      Root Mean Square Error for curve fit
    """

    # Interpolate
    x_int = np.linspace(min(min(x_up), min(x_bot)), max(max(x_up), max(x_bot)), 50)
    y_up_int = interpolate.interp1d(x_up, y_up, fill_value="extrapolate")
    y_bot_int = interpolate.interp1d(x_bot, y_bot, fill_value="extrapolate")

    # Calculate centreline of blade
    x_cent = x_int
    y_cent = (y_up_int(x_int) - y_bot_int(x_int)) / 2 + y_bot_int(x_int)

    # Calculate rotation angle
    ang_rot = np.arctan((y_cent[np.argmax(x_cent)] - y_cent[np.argmin(x_cent)]) / (max(x_cent) - min(x_cent)))

    # Rotate top about leading edge of blade
    x_le = min(x_cent)
    y_le = y_cent[np.argmin(x_cent)]
    x_rot_top, y_rot_top = rotate(x_le, y_le, x_int, y_up_int(x_int), ang_rot)
    x_rot_bot, y_rot_bot = rotate(x_le, y_le, x_int, y_bot_int(x_int), ang_rot)

    # Normalize
    x_top, y_top = normalize(x_rot_top, y_rot_top)
    x_bot, y_bot = normalize(x_rot_bot, y_rot_bot)

    # Approximate
    popt_top, pcov_top = curve_fit(fun, x_top, y_top)
    perr_top = np.sqrt(np.diag(pcov_top))  # standard deviation sigma
    rmse_top = np.mean((y_top - fun(x_top, *popt_top)) ** 2)

    popt_bot, pcov_bot = curve_fit(fun, x_bot, y_bot)
    perr_bot = np.sqrt(np.diag(pcov_bot))  # standard deviation sigma
    rmse_bot = np.mean((y_bot - fun(x_bot, *popt_bot)) ** 2)

    return x_top, y_top, popt_top, perr_top, rmse_top, x_bot, y_bot, popt_bot, perr_bot, rmse_bot, ang_rot


def camber_span_interp(popt: list, perc: list):
    """Compute function to describe normalized and de-rotated body_force_model of rotor and stator blades for transonic BLI
    fuselage fan as function of span.
    Original data from Castillo Pardo & Hall 2020, Fig. 5
    Author:  Anais Habermann
     Args:
        popt    Arguments for function that describe body_force_model at local positions
        perc    Lokal span positions of cambers

    Returns:
        popt_a  Coefficients for function that describes coefficients of body_force_model of blade along blade span.
    """

    if len(popt) != len(perc):
        Warning('Interpolation not possible.')
    else:
        # Approximate coefficients
        popt_a, pcov_a = curve_fit(fun2, perc, list(np.concatenate(popt)))

        return popt_a


def chord_span_interp(radius: list, chord_lengths: list):
    """Compute function to describe chord length as function of span.
    Author:  Anais Habermann
     Args:
        radius          Local span positions of chord lengths
        chord_lengths    Local chord lengths

    Returns:
        popt_a  Coefficients for function that describes coefficients of chord length of blade along blade span.
    """

    # Approximate coefficients
    popt_a, pcov_a = curve_fit(fun2, radius, chord_lengths)

    return popt_a


def angle_interp(popt: list, pos: list):
    """Compute function to describe normalized and de-rotated body_force_model of rotor and stator blades for transonic BLI
    fuselage fan as function of span.
    Original data from Castillo Pardo & Hall 2020, Fig. 5
    Author:  Anais Habermann
     Args:
        popt    Arguments for function that describe body_force_model at local positions
        pos    Local span positions of cambers

    Returns:
        popt_a, popt_b  Coefficients for function that describes coefficients of body_force_model of blade along blade span.
    """

    if len(popt) != len(pos):
        Warning('Interpolation not possible.')
    else:
        # Approximate coefficients
        popt_a, pcov_a = curve_fit(fun2, pos, list(np.concatenate(popt)))
        perr_a = np.sqrt(np.diag(pcov_a))  # standard deviation sigma
        rmse_a = np.mean((np.concatenate(popt) - fun2(np.array(pos), *popt_a)) ** 2)

        return popt_a


def scale_camber(pos: list, root_len: float, popt_camber, popt_angle, popt_width):
    """Compute function to describe normalized and de-rotated body_force_model of rotor and stator blades for transonic BLI
    fuselage fan as function of span.
    Original data from Castillo Pardo & Hall 2020, Fig. 5
    Author:  Anais Habermann
     Args:
        pos             Local span positions of cambers
        root_len        Streamwise length of blade root, i.e. max streamwise width
        popt_camber   Coefficients for body_force_model along span
        popt_angle    Coefficients of blade angle along span
        popt_width    Coefficients of blade width function along span

    Returns:
        popt_a, popt_b  Coefficients for function that describes coefficients of body_force_model of blade along blade span.
    """

    x = np.linspace(0, 1, 100)

    x_rot = []
    y_rot = []

    for i in range(0, len(pos)):
        angle = float(fun2(pos[i], *popt_angle))
        width = float(fun2(pos[i], *popt_width))

        # Generate body_force_model for specific position along span
        camber = fun(x, fun2(pos[i], *popt_camber))

        # Scale body_force_model
        beta = np.deg2rad((180 - np.rad2deg(angle)) / 2)
        x_max = width * root_len * (1 + np.tan(angle) / np.tan(beta))

        x_scale = x * x_max
        y_scale = camber * x_max

        # Rotate body_force_model
        x_rotated, y_rotated = rotate(0, 0, x_scale, y_scale, -angle)

        x_rot.append(x_rotated)
        y_rot.append(y_rotated)

    return pos, x_rot, y_rot


if __name__ == "__main__":
    rotor_10_top = [
        [1, 0.99887, 0.91118, 0.86187, 0.79254, 0.76241, 0.67399, 0.67096, 0.57746, 0.55305, 0.47576, 0.43659, 0.39026,
         0.33045, 0.32024, 0.24283, 0.24273, 0.17058, 0.15943, 0.09758, 0.09161, 0.04373, 0.00831],
        [0.73328, 0.73291, 0.71743, 0.70654, 0.68481, 0.67385, 0.6419, 0.64066, 0.59733, 0.58516, 0.54383, 0.51979,
         0.48886, 0.44764, 0.44056, 0.38484, 0.38477, 0.32318, 0.31305, 0.25409, 0.24794, 0.19242, 0.13967]]
    rotor_10_bot = [
        [1, 0.99887, 0.91118, 0.86187, 0.79254, 0.76241, 0.67399, 0.67096, 0.57746, 0.55305, 0.47576, 0.43659, 0.39026,
         0.33045, 0.32024, 0.24283, 0.24273, 0.17058, 0.15943, 0.09758, 0.09161, 0.04373, 0.00831],
        [0.73364, 0.73254, 0.69019, 0.66786, 0.6367, 0.62288, 0.58097, 0.57949, 0.53163, 0.51783, 0.47015, 0.44502,
         0.41567, 0.37741, 0.37067, 0.31657, 0.31649, 0.26269, 0.25409, 0.20523, 0.20059, 0.16444, 0.13819]]

    rotor_50_top = [
        [0.99016, 0.98942, 0.89216, 0.8818, 0.81477, 0.76532, 0.71305, 0.6422, 0.5951, 0.53972, 0.50294, 0.43576,
         0.41889, 0.34728, 0.33556, 0.25142, 0.23969, 0.17769, 0.15045, 0.09363, 0.09289],
        [0.81829, 0.81798, 0.77563, 0.77053, 0.7318, 0.69948, 0.66345, 0.61166, 0.57504, 0.52853, 0.49629, 0.43809,
         0.42348, 0.35895, 0.3477, 0.26249, 0.25037, 0.18539, 0.15453, 0.07615, 0.07504]]
    rotor_50_bot = [
        [0.99016, 0.98942, 0.89216, 0.8818, 0.81477, 0.76532, 0.71305, 0.6422, 0.5951, 0.53972, 0.50294, 0.43576,
         0.41889, 0.34728, 0.33556, 0.25142, 0.23969, 0.17769, 0.15045, 0.09363, 0.09289],
        [0.81649, 0.81598, 0.74683, 0.73923, 0.68877, 0.65082, 0.61059, 0.55498, 0.51633, 0.46954, 0.43844, 0.38113,
         0.3664, 0.30238, 0.29186, 0.2162, 0.20548, 0.14859, 0.12424, 0.07429, 0.07365]]

    rotor_90_top = [
        [0.92488, 0.83044, 0.82312, 0.73598, 0.72943, 0.66892, 0.64373, 0.60103, 0.56477, 0.54716, 0.49098, 0.46376,
         0.39807, 0.398, 0.33165, 0.32569, 0.26812, 0.26669, 0.22755, 0.22754, ],
        [0.99554, 0.90379, 0.89599, 0.79189, 0.78306, 0.69391, 0.65906, 0.60104, 0.54633, 0.51932, 0.43612, 0.39525,
         0.29049, 0.29038, 0.18722, 0.17795, 0.08425, 0.08172, 0.00371, 0.00371, ]]
    rotor_90_bot = [
        [0.92488, 0.83044, 0.82312, 0.73598, 0.72943, 0.66892, 0.64373, 0.60103, 0.56477, 0.54716, 0.49098, 0.46376,
         0.39807, 0.398, 0.33165, 0.32569, 0.26812, 0.26669, 0.22755, 0.22754, ],
        [0.99554, 0.86627, 0.85599, 0.73031, 0.7207, 0.6315, 0.59435, 0.53183, 0.47994, 0.45541, 0.37741, 0.33853,
         0.24452, 0.24443, 0.15208, 0.14339, 0.05498, 0.05288, 0.00297, 0.00297, ]]

    x_r_10, y_r_10, rot_r_10, popt_r_10, _, rmse_r_10 = blade_camber_norm(rotor_10_top[0], rotor_10_top[1],
                                                                          rotor_10_bot[0], rotor_10_bot[1])
    x_r_50, y_r_50, rot_r_50, popt_r_50, _, rmse_r_50 = blade_camber_norm(rotor_50_top[0], rotor_50_top[1],
                                                                          rotor_50_bot[0], rotor_50_bot[1])
    x_r_90, y_r_90, rot_r_90, popt_r_90, _, rmse_r_90 = blade_camber_norm(rotor_90_top[0], rotor_90_top[1],
                                                                          rotor_90_bot[0], rotor_90_bot[1])

    plot_blade(rotor_10_top[0], rotor_10_top[1], rotor_10_bot[0], rotor_10_bot[1], x_r_10, y_r_10, popt_r_10,
               'rotor_10')
    plot_blade(rotor_50_top[0], rotor_50_top[1], rotor_50_bot[0], rotor_50_bot[1], x_r_50, y_r_50, popt_r_50,
               'rotor_50')
    plot_blade(rotor_90_top[0], rotor_90_top[1], rotor_90_bot[0], rotor_90_bot[1], x_r_90, y_r_90, popt_r_90,
               'rotor_90')

    # Calculate coefficients for function, which describes body_force_model along blade span
    popt_r_camber = camber_span_interp([popt_r_10, popt_r_50, popt_r_90], [0.1, 0.5, 0.9])

    # Plot cambers for certain positions along span
    pos = np.linspace(0, 1, 11)
    plot_camber(popt_r_camber, pos, name='rotor')

    # Calculate function that describes angle as function of span wise blade position
    popt_r_angle, _ = curve_fit(fun2, np.array([0.1, 0.5, 0.9]), np.array([rot_r_10, rot_r_50, rot_r_90]))

    # Calculate function that describes width of blade (i.e. projection of blade on streamwise plane)
    popt_r_width, _ = curve_fit(fun2, np.array([0.1, 0.5, 0.9]), np.array([max(rotor_10_top[0]) - min(rotor_10_top[0]),
                                                                           max(rotor_50_top[0]) - min(rotor_50_top[0]),
                                                                           max(rotor_90_top[0]) - min(
                                                                               rotor_90_top[0])]))

    root_width = 0.5
    b, x, y, = scale_camber(list(pos), root_width, popt_r_camber, popt_r_angle, popt_r_width)

    print(b[0], x[0], y[0])

    plot_spanwise_camber(b, x, y, 'rotor')
