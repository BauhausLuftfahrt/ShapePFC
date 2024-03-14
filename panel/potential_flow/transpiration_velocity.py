"""Returns transpiration velocity for the simulation of the displacement effect of the boundary layer on the outer
inviscid potential flow

Author:  Carlos E. Ribeiro Santa Cruz Mendoza, Nikolaus Romanow, A. Habermann
"""

import numpy as np


def transpirationVelocity(Xm, Ym, ue_input, delta_starPhys_BC, delta, phi):
    """Computes the transpiration velocity on the control points

    Author:  Nikolaus Romanow

     Args:
        Xm              [m]     1-D array X-coordinate of segment's mid-point
        Ym              [m]     1-D array Y-coordinate of segment's mid-point
        ue_input        [-]     Dimensionless edge velocity (divided by u_inf)
        delta_starPhys_BC  [m]  1-D array Displacement thickness
        delta           [m]     1-D array Boundary layer thickness
        phi             [rad]   1-D array Segment angle w.r.t symmetry axis

    Returns:
        v_trans         [-]     Dimensionless transpiration velocity (divided by u_inf)

    Sources:
        [-] Landweber, L. (1978). On irrotational flows equivalent to the boundary layer and wake
        [-] Katz, J., & Plotkin, A. (2001). Low-Speed Aerodynamics (Second edition)
        [-] Dvorak, F. A., Woodward, F. A., & Maskew, B. (1977).
            A three-dimensional viscous/potential flow interaction analysis method for multi-element wings
    """

    # Calculate derivation of product
    prod = Ym * ue_input * delta_starPhys_BC  # Transpiration velocity for axisymmetric body (helpers case, see Landweber)

    v_trans = np.zeros(len(prod))
    for i in range(len(prod)):
        if i == 0:
            v_trans[i] = -(prod[i + 1] - prod[i]) / (
                        ((Xm[i + 1] - Xm[i]) ** 2 + (Ym[i + 1] - Ym[i]) ** 2) ** 0.5)  # forward DQ
        elif i == len(prod) - 1:
            v_trans[i] = -(prod[i] - prod[i - 1]) / (
                        ((Xm[i] - Xm[i - 1]) ** 2 + (Ym[i] - Ym[i - 1]) ** 2) ** 0.5)  # backward DQ
        else:
            # v_trans[i] = -(prod[i+1] - prod[i]) / (((Xm[i+1] - Xm[i])**2 + (Ym[i+1] - Ym[i])**2)**0.5)  # forward DQ
            # v_trans[i] = -(prod[i] - prod[i-1]) / (((Xm[i] - Xm[i-1])**2 + (Ym[i] - Ym[i-1])**2)**0.5)  # backward DQ
            v_trans[i] = -(prod[i + 1] - prod[i - 1]) / (
                        ((Xm[i + 1] - Xm[i - 1]) ** 2 + (Ym[i + 1] - Ym[i - 1]) ** 2) ** 0.5)  # central DQ

    v_trans = v_trans / (4 * np.pi * (
                Ym + delta * np.cos(phi)))  # Transpiration velocity for axisymmetric body (helpers case, see Landweber)

    return v_trans
