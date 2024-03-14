"""Calculate height of first boundary layer cell for given y+ value

Author:  A. Habermann

Sources:
    [1] Frank M. White, "Fluid Mechanics", 8th edition, 2017.
"""

# Built-in/Generic Imports
import numpy as np
from bhlpythontoolbox.atmosphere.atmosphere import Atmosphere


# approximate height of first cell off the wall
def first_cell_height_y_plus(y_plus, Ma, alt, l_ref):
    """
        y_plus:                     dimensionless wall distance [-]. Should be < 1, if boundary layer should be fully resolved
        Ma                          freestream Mach number
        alt                         ambient altitude [m]
        l_ref                       reference length [m]
    """

    atmosphere = Atmosphere(altitude=alt, dt_isa=0, backend='ambiance',
                            ext_props_required=['rho', 'sos', 'mue'])
    u = atmosphere.ext_props['sos'] * Ma

    re_x = atmosphere.ext_props['rho'] * u * l_ref / atmosphere.ext_props[
        'mue']  # reference length based Reynolds number [-]
    c_f = 0.027 / (re_x ** (
                1 / 7))  # skin friction coefficient estimation (Prandtl power law, for low-Reynolds number turbulent boundary layers) [-]
    tau_w = (c_f * atmosphere.ext_props['rho'] * u ** 2) / 2  # wall shear stress [kg/m^2/s]
    u_s = np.sqrt(tau_w / atmosphere.ext_props['rho'])  # shear velocity [m/s]
    delta_s = y_plus * atmosphere.ext_props['mue'] / (u_s * atmosphere.ext_props['rho'])  # height of first cell [m]

    return delta_s


if __name__ == "__main__":
    s = first_cell_height_y_plus(1, 0.8, 10668, 3)
