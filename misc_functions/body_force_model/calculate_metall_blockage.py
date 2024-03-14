"""Calculate and plot the metall blockage of a rotor blade based on a specified number of cross-section of the blade
defined by x- and y-coordinates.

Author:  A. Habermann
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from misc_functions.body_force_model.blade_camber import fun2


# use cm instead of inch for figure size
def cm2inch(value):
    return value / 2.54


def fun_thickness(x, a, b, c, d, e, f):
    # attention: this function is similar to the function used for the NACA airfoil thickness distribution. Be aware
    # that the trailing edge thickness is NOT 0, but a little bit higher.
    return a * (b * np.sqrt(x) + c * x + d * x ** 2 + e * x ** 3 + f * x ** 4)


def plot_spanwise_blockage(spanwise_position, x, blockage, name):
    fig, ax = plt.subplots(figsize=(cm2inch(20), cm2inch(20)))
    plt.axis('equal')
    ax.set(xlabel=r'x', ylabel=r'y')
    for i in range(0, len(spanwise_position)):
        ax.plot(x[i], blockage[i], label=str(spanwise_position[i]), marker='None')
    ax.legend()
    fig.savefig('./blockage/Blockage_spanwise_' + str(name) + '.pdf')


def blockage(n: int, t, r):
    """Calculate local metall blockage parameter for rotor/stator
    Author:  Anais Habermann
     Args:
        n  [-]                              Number of blades
        t [m]                               Local blade thickness
        r [m]                               Local radial position

    Returns:
        local metall blockage parameter
    """

    return ((2 * np.pi * r / n) - t) / (2 * np.pi * r / n)


def thickness_distribution(x_top, y_top, x_bot, y_bot):
    """Calculate local blade thickness distribution
    Author:  Anais Habermann
     Args:
        x_top, y_top  [m]       Original coordinates of upper side of blade
        x_bot, y_bot  [m]       Original coordinates of lower side of blade

    Returns:
        t
    """
    # Interpolate
    x_int = np.linspace(min(min(x_top), min(x_bot)), max(max(x_top), max(x_bot)), 50)
    y_up_int = interpolate.interp1d(x_top, y_top, fill_value="extrapolate")
    y_bot_int = interpolate.interp1d(x_bot, y_bot, fill_value="extrapolate")

    t_int = abs(y_up_int(x_int) - y_bot_int(x_int))

    idxnan = np.where(np.isnan(t_int))
    t_int = np.delete(t_int, [idxnan], axis=0)
    x_int = np.delete(x_int, [idxnan], axis=0)

    idxinf = np.where(np.isinf(t_int))
    t_int = np.delete(t_int, [idxinf], axis=0)
    x_int = np.delete(x_int, [idxinf], axis=0)

    return x_int, t_int


def fit_thickness(x, t):
    """Calculate local blade thickness distribution of blade
    Author:  Anais Habermann
     Args:
        x  [m]       X-coordinates
        t [m]       Thickness distribution

    Returns:
        popt
    """
    # Approximate
    popt, pcov = curve_fit(fun_thickness, x, t)
    perr = np.sqrt(np.diag(pcov))  # standard deviation sigma
    rmse = np.mean((t - fun_thickness(x, *popt)) ** 2)  # root mean squared error

    return popt, pcov, perr, rmse


def thickness_span_interp(popt: list, perc: list):
    """Compute function to describe thickness distribution of rotor and stator blades for transonic BLI
    fuselage fan as function of blade span.
    Original data from Castillo Pardo & Hall 2020, Fig. 5
    Author:  Anais Habermann
     Args:
        popt    Arguments for function that describes metall blockage at local positions
        perc    Local span positions of metall blockages

    Returns:
        popt_t  Coefficients for function that describes coefficients of thickness along blade span.
    """

    if len(popt) != len(perc):
        Warning('Interpolation not possible.')
    else:
        popt_t = []
        pcov_t = []
        for i in range(0, len(popt[0])):
            # Approximate coefficients
            popt_1, pcov_1 = curve_fit(fun2, perc, [item[i] for item in popt])
            popt_t.append(popt_1)
            pcov_t.append(pcov_1)

        return popt_t


def scale_blockage(pos: list, root_len: float, popt_thickness, popt_width, n_blades, h_blade, r_hub):
    """Compute function to describe normalized and de-rotated metall blockage of rotor and stator blades for transonic BLI
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

    x_blockage = []
    blocked = []

    for i in range(0, len(pos)):
        width = float(fun2(pos[i], *popt_width))

        # Generate thickness for specific position along span
        thickness = fun_thickness(x, fun2(pos[i], *popt_thickness[0]), fun2(pos[i], *popt_thickness[1]),
                                  fun2(pos[i], *popt_thickness[2]), fun2(pos[i], *popt_thickness[3]),
                                  fun2(pos[i], *popt_thickness[4]), fun2(pos[i], *popt_thickness[5]))

        # Scale length
        x_max = width * root_len

        x_scale = x * x_max
        thickness_scale = thickness * x_max

        # Calculate blockage
        block = blockage(n_blades, thickness_scale, r_hub + pos[i] * h_blade)

        x_blockage.append(x_scale)
        blocked.append(block)

    return pos, x_blockage, blocked


if __name__ == "__main__":
    r_hub = 0.3
    h_blade = 0.6
    pos = [0.1, 0.5, 0.9]

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

    x_thick_r_10, t_r_10 = thickness_distribution(rotor_10_top[0], rotor_10_top[1], rotor_10_bot[0], rotor_10_bot[1])

    popt_t_r_10, _, _, _ = fit_thickness(x_thick_r_10, t_r_10)

    # calculate radius
    r_10 = r_hub + pos[0] * h_blade
    # calculate metall blockage
    b_r_10 = blockage(20, fun_thickness(x_thick_r_10, *popt_t_r_10), r_10)

    x_thick_r_50, t_r_50 = thickness_distribution(rotor_50_top[0], rotor_50_top[1], rotor_50_bot[0], rotor_50_bot[1])

    popt_t_r_50, _, _, _ = fit_thickness(x_thick_r_50, t_r_50)

    # calculate radius
    r_50 = r_hub + pos[1] * h_blade
    b_r_50 = blockage(20, fun_thickness(x_thick_r_50, *popt_t_r_50), r_50)

    x_thick_r_90, t_r_90 = thickness_distribution(rotor_90_top[0], rotor_90_top[1], rotor_90_bot[0], rotor_90_bot[1])

    popt_t_r_90, _, _, _ = fit_thickness(x_thick_r_90, t_r_90)

    # calculate radius
    r_90 = r_hub + pos[2] * h_blade
    b_r_90 = blockage(20, fun_thickness(x_thick_r_90, *popt_t_r_90), r_90)

    plt.plot(x_thick_r_10, t_r_10)
    plt.plot(np.linspace(min(x_thick_r_10), max(x_thick_r_10), 100),
             fun_thickness(np.linspace(min(x_thick_r_10), max(x_thick_r_10), 100), *popt_t_r_10))
    plt.plot(x_thick_r_10, b_r_10)
    plt.show()

    # Calculate coefficients for function, which describes metall blockage along blade span
    popt_r_thickness = thickness_span_interp([popt_t_r_10, popt_t_r_50, popt_t_r_90], [0.1, 0.5, 0.9])

    plt.plot(x_thick_r_10, b_r_10)
    plt.plot(np.linspace(min(x_thick_r_10), max(x_thick_r_10), 100),
             blockage(20, fun_thickness(np.linspace(min(x_thick_r_10), max(x_thick_r_10), 100), *popt_t_r_10), r_10))
    plt.show()

    print(blockage(20, fun_thickness(np.linspace(min(x_thick_r_10), max(x_thick_r_10), 100), *popt_t_r_10), r_10))
