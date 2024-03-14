"""Analytically express Bernstein polynomials in terms of designer imposed constraints (iCST - intuitive Class/Shape
Transformation)

Author:  A. Habermann

Sources.:
    [1] Christie, R., Heidebrecht A. and MacManus, D. An Automated Appraoch to Nacelle Parameterization Using
    Intuitive Class Shape Transformation Curves. ASME. 2019. DOI: 10.1115/1.4035283
    [2] Tejero, F., Christie, R., MacManus, D., Sheaf, C. Non-axisymmetric aero-engine nacelle design by
    surrogate-based methods. Aerospace Science and Technology. 2021. https://doi.org/10.1016/j.ast.2021.106890
    [3] Zhu, F. and Qin, N. Intuitive Class/Shape Function Parameterization. AIAA Journal. Vol. 52, No. 1, 2014.
"""

import numpy as np
import matplotlib.pyplot as plt
from geometry_generation.intuitive_class_shape_transformation.cst_functions import bernstein, cls, cls_deriv1, \
    cls_deriv2, \
    bernstein_deriv1, bernstein_deriv2, cst


def bpi_solve(constraints: list, bp_le: float = 0, bp_te: float = 0, n1: float = 0.5, n2: float = 1,
              delta_xi_te: float = 0.0):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function
    Parameters
    ----------
    constraints : list
        list of tuples [(k: type of constraint, Psi: normalized x-position, Xi(k)(Psi): normalized radial position),...]
        k = 0 : position
        k = 1: gradient
        k = 2: 2nd derivative
    bp_le : float
        Bernstein polynomial coefficient for leading edge
    bp_te : float
        Bernstein polynomial coefficient for trailing edge
    n1, n2: : float
        Class function parameters. Default parameters for round-nosed airfoils.
    delta_xi_te : float
        Trailing edge vs. leading edge radial position delta

    Returns
    -------
    bpi : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    It is assumed that 0 <= x <= 1.
    References
    ----------
    [1] & [2]
    """

    # degree of cst function
    n = len(constraints) + 1

    A = np.zeros((n - 1, n - 1))
    b = np.zeros((1, n - 1))

    a_0 = bp_le  # see nomenclature [3]
    a_n = bp_te

    # fill A matrix and B column
    for i in range(0, n - 1):  # rows
        # type of constraint
        k = constraints[i][0]
        for j in range(0, n - 1):  # columns
            if k == 0:
                A[i, j] = bernstein(constraints[i][1], j + 1, n) * cls(constraints[i][1], n1, n2)
            elif k == 1:
                A[i, j] = bernstein_deriv1(constraints[i][1], j + 1, n) * cls(constraints[i][1], n1, n2) + \
                          bernstein(constraints[i][1], j + 1, n) * cls_deriv1(constraints[i][1], n1, n2)
            elif k == 2:
                A[i, j] = bernstein_deriv2(constraints[i][1], j + 1, n) * cls(constraints[i][1], n1, n2) + \
                          2 * (bernstein_deriv1(constraints[i][1], j + 1, n) * cls_deriv1(constraints[i][1], n1, n2)) + \
                          bernstein(constraints[i][1], j + 1, n) * cls_deriv2(constraints[i][1], n1, n2)
            else:
                Warning('iCST: Geometrical constraint type undefined')

        if k == 0:
            b[0][i] = constraints[i][2] - delta_xi_te * constraints[i][1] - cls(constraints[i][1], n1, n2) * \
                      (a_0 * bernstein(constraints[i][1], 0, n) + a_n * bernstein(constraints[i][1], n, n))
        elif k == 1:
            b[0][i] = constraints[i][2] - delta_xi_te - cls_deriv1(constraints[i][1], n1, n2) * \
                      (a_0 * bernstein(constraints[i][1], 0, n) + bp_te * bernstein(constraints[i][1], n, n)) - \
                      cls(constraints[i][1], n1, n2) * (a_0 * bernstein_deriv1(constraints[i][1], 0, n) +
                                                        bp_te * bernstein_deriv1(constraints[i][1], n, n))
        elif k == 2:
            b[0][i] = constraints[i][2] - cls_deriv2(constraints[i][1], n1, n2) * \
                      (a_0 * bernstein(constraints[i][1], 0, n) + bp_te * bernstein(constraints[i][1], n, n)) - \
                      2 * cls_deriv1(constraints[i][1], n1, n2) * (a_0 * bernstein_deriv1(constraints[i][1], 0, n) +
                                                                   bp_te * bernstein_deriv1(constraints[i][1], n, n)) - \
                      cls(constraints[i][1], n1, n2) * (a_0 * bernstein_deriv2(constraints[i][1], 0, n) +
                                                        bp_te * bernstein_deriv2(constraints[i][1], n, n))
        else:
            Warning('iCST: Geometrical constraint type undefined')

    b = np.squeeze(b)
    # solve linear system of equations
    X = np.linalg.inv(A).dot(b)
    return X


def bpi_solve_nac_int(constraints: list, bp_te: float = 0, n1: float = 0.5, n2: float = 1, delta_xi_te: float = 0.0):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function. This special function
    is required for cases for which bp0 is not calculated analytically beforehand, but should be a result of the
    solution of the linear system of equations.
    Parameters
    ----------
    constraints : list
        list of tuples [(k: type of constraint, Psi: normalized x-position, Xi(k)(Psi): normalized radial position),...]
        k = 0 : position
        k = 1: gradient
        k = 2: 2nd derivative
    bp_te : float
        Bernstein polynomial coefficient for trailing edge
    n1, n2: : float
        Class function parameters. Default parameters for round-nosed airfoils.
    delta_xi_te : float
        Trailing edge vs. leading edge radial position delta

    Returns
    -------
    bpi : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    It is assumed that 0 <= x <= 1.
    References
    ----------
    [1] & [2]
    """

    # degree of cst function
    n = len(constraints)

    A = np.zeros((n, n))
    b = np.zeros((1, n))

    a_n = bp_te

    # fill A matrix and B column
    for i in range(0, n):  # rows
        # type of constraint
        k = constraints[i][0]
        for j in range(0, n):  # columns
            if k == 0:
                A[i, j] = bernstein(constraints[i][1], j, n) * cls(constraints[i][1], n1, n2)
            elif k == 1:
                A[i, j] = bernstein_deriv1(constraints[i][1], j, n) * cls(constraints[i][1], n1, n2) + \
                          bernstein(constraints[i][1], j, n) * cls_deriv1(constraints[i][1], n1, n2)
            elif k == 2:
                A[i, j] = bernstein_deriv2(constraints[i][1], j, n) * cls(constraints[i][1], n1, n2) + \
                          2 * (bernstein_deriv1(constraints[i][1], j, n) * cls_deriv1(constraints[i][1], n1, n2)) + \
                          bernstein(constraints[i][1], j, n) * cls_deriv2(constraints[i][1], n1, n2)
            else:
                Warning('iCST: Geometrical constraint type undefined')

        if k == 0:
            b[0][i] = constraints[i][2] - delta_xi_te * constraints[i][1] - cls(constraints[i][1], n1, n2) * \
                      (a_n * bernstein(constraints[i][1], n, n))
        elif k == 1:
            b[0][i] = constraints[i][2] - delta_xi_te - cls_deriv1(constraints[i][1], n1, n2) * \
                      (bp_te * bernstein(constraints[i][1], n, n)) - \
                      cls(constraints[i][1], n1, n2) * (bp_te * bernstein_deriv1(constraints[i][1], n, n))
        elif k == 2:
            b[0][i] = constraints[i][2] - cls_deriv2(constraints[i][1], n1, n2) * \
                      (bp_te * bernstein(constraints[i][1], n, n)) - \
                      2 * cls_deriv1(constraints[i][1], n1, n2) * (bp_te * bernstein_deriv1(constraints[i][1], n, n)) - \
                      cls(constraints[i][1], n1, n2) * (bp_te * bernstein_deriv2(constraints[i][1], n, n))
        else:
            Warning('iCST: Geometrical constraint type undefined')

    b = np.squeeze(b)
    # solve linear system of equations
    X = np.linalg.inv(A).dot(b)
    return X


def bpi_fuse_preint(r_cent_f: float, r_hi_hub: float, teta_te: float, l_preint: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for fuselage section
    between center section and intake.
    Parameters
    ----------
    :param r_cent_f: (float):   [m]     Fuselage center section radius
    :param r_hi_hub: (float):   [m]     Fuselage radius at highlight
    :param teta_te: (float)     [°]     TE angle of section
    :param l_preint: (float)    [m]     Length of section

    Returns
    -------
    bpi_fus_preint : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by nacelle length
    r_cent_f /= l_preint
    r_hi_hub /= l_preint
    r_le = r_cent_f
    r_te = r_hi_hub

    # calculate leading edge coefficient. first coefficient defined by leading edge radius. Needs to be normalized with nacelle length
    bp_le = -(r_te - r_le)
    # calculate trailing edge coefficent. last coefficient defined by boattail angle.
    bp_te = -np.tan(teta_te) + (r_te - r_le)
    # populate bpi array
    bpi_fus_preint = np.array([bp_le])
    # add all other coefficients by solving linear system of equations
    # create list with constraints
    constraints = [(2, 0, 0),  # curvature at fuselage FF stage outlet
                   (2, 1, 0)  # curvature at fuselage TE
                   ]
    n1 = 1
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = r_te - r_le
    bpi_fus_preint = np.append(bpi_fus_preint, bpi_solve(constraints, bp_le, bp_te, n1, n2, delta_xi_te))
    bpi_fus_preint = np.append(bpi_fus_preint, bp_te)

    return bpi_fus_preint


def bpi_fuse_int(r_hi_hub: float, r_thr_hub: float, r_12_hub: float, l_hi_f: float, l_thr_f: float,
                 l_12_f: float, teta_f_aft: float, teta_ff_in: float, l_int: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for fuselage section
    at intake.
    Parameters
    ----------
    :param r_hi_hub: (float):   [m]     Fuselage radius at highlight
    :param r_thr_hub: (float):  [m]     Fuselage radius at throat
    :param r_12_hub: (float):   [m]     Fuselage radius at station 12 (FF rotor inlet)
    :param l_hi_f: (float)      [m]     Absolute x-position of highlight
    :param l_thr_f: (float)     [m]     Absolute x-position of throat
    :param l_12_f: (float)      [m]     Absolute x-position of station 12
    :param teta_f_aft: (float)  [°]     Angle of fuselage aft section
    :param teta_ff_in: (float)  [°]     Angle of FF stage at fuselage
    :param l_int: (float)       [m]     Length of section

    Returns
    -------
    bpi_fuse_front : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by fuselage aft section length
    r_hi_hub /= l_int
    r_thr_hub /= l_int
    r_12_hub /= l_int
    l_hi_f /= l_int
    l_thr_f /= l_int
    l_12_f /= l_int
    r_le = r_hi_hub
    r_te = r_12_hub

    # calculate leading edge coefficient. first coefficient defined by leading edge gradient.
    bp_le = np.tan(teta_f_aft) - (r_te - r_le)
    # calculate trailing edge coefficent.
    bp_te = -np.tan(teta_ff_in) + (r_te - r_le)
    # populate bpi array
    bpi_fuse_int = np.array([bp_le])
    # add all other coefficients by solving linear system of equations
    # create list with constraints
    constraints = [(2, 0, 0),  # LE curvature
                   (0, l_thr_f - l_hi_f, r_thr_hub),  # Throat position
                   # (2, 1, 0)                      # TE curvature, optional. Might want to use it, if geometries behave weird
                   ]
    n1 = 1
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = r_te - r_le
    bpi_fuse_int = np.append(bpi_fuse_int, bpi_solve(constraints, bp_le, bp_te, n1, n2, delta_xi_te))
    bpi_fuse_int = np.append(bpi_fuse_int, bp_te)

    return bpi_fuse_int


def bpi_fuse_int2(r_hi_hub: float, r_thr_hub: float, r_12_hub: float, l_hi_f: float, l_thr_f: float,
                  l_12_f: float, teta_f_aft: float, teta_ff_in: float, l_int: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for fuselage section
    at intake.
    Parameters
    ----------
    :param r_hi_hub: (float):   [m]     Fuselage radius at highlight
    :param r_thr_hub: (float):  [m]     Fuselage radius at throat
    :param r_12_hub: (float):   [m]     Fuselage radius at station 12 (FF rotor inlet)
    :param l_hi_f: (float)      [m]     Absolute x-position of highlight
    :param l_thr_f: (float)     [m]     Absolute x-position of throat
    :param l_12_f: (float)      [m]     Absolute x-position of station 12
    :param teta_f_aft: (float)  [°]     Angle of fuselage aft section
    :param teta_ff_in: (float)  [°]     Angle of FF stage at fuselage
    :param l_int: (float)       [m]     Length of section

    Returns
    -------
    bpi_fuse_front : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by fuselage aft section length
    r_hi_hub /= l_int
    r_thr_hub /= l_int
    r_12_hub /= l_int
    l_hi_f /= l_int
    l_thr_f /= l_int
    l_12_f /= l_int
    r_le = r_hi_hub
    r_te = r_12_hub

    # calculate leading edge coefficient. first coefficient defined by leading edge gradient.
    bp_le = np.tan(teta_f_aft) - (r_te - r_le)
    # calculate trailing edge coefficent.
    bp_te = -np.tan(teta_ff_in) + (r_te - r_le)
    # populate bpi array
    bpi_fuse_int = np.array([bp_le])
    # add all other coefficients by solving linear system of equations
    # create list with constraints
    constraints = [(0, l_thr_f - l_hi_f, r_thr_hub),  # Throat position
                   ]
    n1 = 1
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = r_te - r_le
    bpi_fuse_int = np.append(bpi_fuse_int, bpi_solve(constraints, bp_le, bp_te, n1, n2, delta_xi_te))
    bpi_fuse_int = np.append(bpi_fuse_int, bp_te)

    return bpi_fuse_int


def bpi_fuse_noz1(r_13_hub: float, r_18_hub: float, l_13: float, l_18: float, teta_ff_out: float, l_noz_f: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for fuselage nozzle
    inside duct.
    Parameters
    ----------
    :param r_13_hub: (float):   [m]     Station 13 fuselage radius
    :param r_18_hub: (float):   [m]     Station 18 fuselage radius
    :param l_13: (float)        [m]     Length highlight to station 13
    :param l_18: (float)        [m]     Length highlight to station 18
    :param teta_ff_out: (float) [°]     TE angle of section
    :param l_noz_f: (float)     [m]     Length of section

    Returns
    -------
    bpi_nac_cowl : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by nacelle length
    r_13_hub /= l_noz_f
    r_18_hub /= l_noz_f
    l_18 /= l_noz_f
    l_13 /= l_noz_f
    r_le = r_13_hub
    r_te = r_18_hub

    # calculate leading edge coefficient. first coefficient defined by leading edge radius. Needs to be normalized with nacelle length
    bp_le = np.tan(teta_ff_out) - (r_te - r_le)
    # calculate trailing edge coefficent. last coefficient defined by boattail angle.
    bp_te = (r_te - r_le)
    # populate bpi array
    bpi_fus_noz = np.array([bp_le])
    # add all other coefficients by solving linear system of equations
    # create list with constraints
    constraints = [(2, 0, 0),  # curvature at fuselage FF stage outlet
                   (2, 1, 0)  # curvature at nozzle exit
                   ]
    n1 = 1
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = r_te - r_le
    bpi_fus_noz = np.append(bpi_fus_noz, bpi_solve(constraints, bp_le, bp_te, n1, n2, delta_xi_te))
    bpi_fus_noz = np.append(bpi_fus_noz, bp_te)

    return bpi_fus_noz


def bpi_fuse_noz2(r_18_hub: float, r_f_te: float, teta_f_cone: float, l_noz_f: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for fuselage cone
    angle.
    Parameters
    ----------
    :param r_18_hub: (float):   [m]     Section 13 fuselage radius
    :param r_f_te: (float):     [m]     Fuselage TE radius
    :param teta_f_cone: (float) [°]     TE angle of section
    :param l_noz_f: (float)     [m]     Length of section

    Returns
    -------
    bpi_nac_cowl : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by nacelle length
    r_18_hub /= l_noz_f
    r_f_te /= l_noz_f
    r_le = r_18_hub
    r_te = r_f_te

    # calculate leading edge coefficient. first coefficient defined by leading edge radius. Needs to be normalized with nacelle length
    bp_le = -(r_te - r_le)
    # calculate trailing edge coefficent. last coefficient defined by boattail angle.
    bp_te = -np.tan(teta_f_cone) + (r_te - r_le)
    # populate bpi array
    bpi_fus_noz = np.array([bp_le])
    # add all other coefficients by solving linear system of equations
    # create list with constraints
    constraints = [  # (2, 0 , 0),         # curvature at fuselage FF stage outlet
        (2, 1, 0)  # curvature at fuselage TE
    ]
    n1 = 1
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = r_te - r_le
    bpi_fus_noz = np.append(bpi_fus_noz, bpi_solve(constraints, bp_le, bp_te, n1, n2, delta_xi_te))
    bpi_fus_noz = np.append(bpi_fus_noz, bp_te)

    return bpi_fus_noz


def bpi_nac_cowl(r_le: float, r_te: float, r_max: float, f_max: float, rho_le: float, beta_te: float, l_nac: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for outer cowl nacelle
    Parameters
    ----------
    :param r_le: (float):   [m]     LE radius
    :param r_te: (float):   [m]     TE radius
    :param r_max: (float):  [m]     Radius of max. nacelle position
    :param f_max: (float):  [-]     Rel. location of max. nacelle position
    :param rho_le: (float)  [-]     Leading edge curvature radius
    :param beta_te: (float) [°]     TE angle of section
    :param l_nac: (float)   [m]     Length of section

    Returns
    -------
    bpi_nac_cowl : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by nacelle length
    r_le /= l_nac
    r_te /= l_nac
    r_max /= l_nac
    f_max /= l_nac
    # beta_te /= l_nac
    rho_le /= l_nac

    # calculate leading edge coefficient. first coefficient defined by leading edge radius. Needs to be normalized with nacelle length
    bp_le = np.sqrt(2 * rho_le)
    # calculate trailing edge coefficent. last coefficient defined by boattail angle.
    bp_te = np.tan(beta_te) + (r_te - r_le)
    # populate bpi array
    bpi_nac_cowl = np.array([bp_le])
    # add all other coefficients by solving linear system of equations
    # create list with constraints
    constraints = [(0, f_max, r_max),  # position of max. thickness
                   (1, f_max, 0),  # gradient of max. thickness
                   (2, 1, 0)  # curvature at trailing edge
                   ]
    n1 = 0.5
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = r_te - r_le
    bpi_nac_cowl = np.append(bpi_nac_cowl, bpi_solve(constraints, bp_le, bp_te, n1, n2, delta_xi_te))
    bpi_nac_cowl = np.append(bpi_nac_cowl, bp_te)

    return bpi_nac_cowl


def bpi_nac_cowl_not_max(r_le: float, r_te: float, r_max: float, f_max: float, rho_le: float, beta_te: float,
                         l_nac: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for outer cowl nacelle
    Parameters
    ----------
    :param r_le: (float):   [m]     LE radius
    :param r_te: (float):   [m]     TE radius
    :param r_max: (float):  [m]     Radius of max. nacelle position
    :param f_max: (float):  [-]     Rel. location of max. nacelle position
    :param rho_le: (float)  [-]     Leading edge curvature radius
    :param beta_te: (float) [°]     TE angle of section
    :param l_nac: (float)   [m]     Length of section

    Returns
    -------
    bpi_nac_cowl : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by nacelle length
    r_le /= l_nac
    r_te /= l_nac
    r_max /= l_nac
    f_max /= l_nac
    # beta_te /= l_nac
    rho_le /= l_nac

    # calculate leading edge coefficient. first coefficient defined by leading edge radius. Needs to be normalized with nacelle length
    bp_le = np.sqrt(2 * rho_le)
    # calculate trailing edge coefficent. last coefficient defined by boattail angle.
    bp_te = np.tan(beta_te) + (r_te - r_le)
    # populate bpi array
    bpi_nac_cowl = np.array([bp_le])
    # add all other coefficients by solving linear system of equations
    # create list with constraints
    constraints = [(0, f_max, r_max),  # position of max. thickness
                   # (1, f_max,0),              # gradient of max. thickness
                   (2, 1, 0)  # curvature at trailing edge
                   ]
    n1 = 0.5
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = r_te - r_le
    bpi_nac_cowl = np.append(bpi_nac_cowl, bpi_solve(constraints, bp_le, bp_te, n1, n2, delta_xi_te))
    bpi_nac_cowl = np.append(bpi_nac_cowl, bp_te)

    return bpi_nac_cowl


def bpi_nac_int(r_le: float, r_thr_tip: float, l_thr: float, r_12_tip: float, l_12: float, beta_ff_in: float,
                l_int: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for nacelle
    intake-like shape.
    Parameters
    ----------
    :param r_le: (float):       [m]     LE radius
    :param r_thr_tip: (float):  [m]     TE radius (throat)
    :param l_thr: (float)       [m]     Length highlight to throat
    :param r_12_tip: (float)    [m]     Station 12 tip radius
    :param l_12: (float)        [m]     Length highlight to station 12
    :param beta_ff_in: (float)  [°]     TE angle of section
    :param l_int: (float)       [m]     Length of section

    Returns
    -------
    bpi_nac_int: np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by fuselage aft section length
    r_le /= l_int
    r_thr_tip /= l_int
    l_thr /= l_int
    r_12_tip /= l_int
    l_12 /= l_int
    r_te = r_12_tip

    # Leading edge coefficient used to ensure direction of leading edge
    bp_le = 0
    # Trailing edge coefficient not calculated analytically.
    bp_te = np.tan(beta_ff_in) + (r_te - r_le)
    # create list with constraints
    constraints = [(0, l_thr, r_thr_tip),  # throat position
                   # (1, l_thr, 0),                   # throat gradient
                   (2, 1, 0)  # FF rotor inlet curvature
                   ]
    n1 = 0.5
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = r_te - r_le
    bpi_nac_int = bpi_solve_nac_int(constraints, bp_te, n1, n2, delta_xi_te)
    bpi_nac_int = np.append(bpi_nac_int, bp_te)

    return bpi_nac_int


def bpi_nac_noz(r_13_tip: float, r_te: float, l_13: float, l_noz: float, beta_ff_out: float, beta_te_noz: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for inner nacelle
    nozzle.
    Parameters
    ----------
    :param r_13_tip: (float)    [m]     Station 13 tip radius
    :param r_te: (float):       [m]     TE radius
    :param l_13: (float)        [m]     Length highlight to station 13
    :param l_noz: (float)       [m]     Length of section
    :param beta_ff_out: (float) [°]     LE angle of section
    :param beta_te_noz: (float) [°]     TE angle of section

    Returns
    -------
    bpi_nac_cowl : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by nacelle length
    r_13_tip /= l_noz
    r_le = r_13_tip
    r_te /= l_noz
    l_13 /= l_noz
    # beta_ff_out /= l_noz
    # beta_te_noz /= l_noz

    # trailing edge gap
    delta_z_te = r_te - r_le

    # Leading edge coefficient not calculated analytically.
    bp_le = np.tan(beta_ff_out) - delta_z_te
    # calculate trailing edge coefficent. last coefficient defined by boattail angle.
    bp_te = -np.tan(beta_te_noz) + delta_z_te
    # populate bpi array
    bpi_nac_noz = np.array([bp_le])
    # add all other coefficients by solving linear system of equations
    # create list with constraints
    constraints = [(2, 0, 0),  # FF stator outlet curvature
                   (2, 1, 0)  # curvature at trailing edge
                   ]
    n1 = 1
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = r_le - r_13_tip
    bpi_nac_noz = np.append(bpi_nac_noz, bpi_solve(constraints, bp_le, bp_te, n1, n2, delta_xi_te))
    bpi_nac_noz = np.append(bpi_nac_noz, bp_te)

    return bpi_nac_noz


def bpi_ff_simple(r_hi_hub: float, r_18_hub: float, teta_le: float, l_ff_stage: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for fuselage section
    between center section and intake.
    Parameters
    ----------
    :param r_cent_f: (float):   [m]     Fuselage center section radius
    :param r_hi_hub: (float):   [m]     Fuselage radius at highlight
    :param teta_te: (float)     [°]     TE angle of section
    :param l_preint: (float)    [m]     Length of section

    Returns
    -------
    bpi_fus_preint : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by nacelle length
    r_hi_hub /= l_ff_stage
    r_18_hub /= l_ff_stage
    r_le = r_hi_hub
    r_te = r_18_hub

    # calculate leading edge coefficient. first coefficient defined by leading edge radius. Needs to be normalized with nacelle length
    bp_le = np.tan(teta_le) - (r_te - r_le)
    # calculate trailing edge coefficent. last coefficient defined by boattail angle.
    bp_te = +(r_te - r_le)
    # populate bpi array
    bpi_ff_simple = np.array([bp_le])
    # add all other coefficients by solving linear system of equations
    # create list with constraints
    constraints = [(2, 0, 0),  # curvature at fuselage FF stage outlet
                   (2, 1, 0)  # curvature at fuselage TE
                   ]
    n1 = 1
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = -(r_te - r_le)
    bpi_ff_simple = np.append(bpi_ff_simple, bpi_solve(constraints, bp_le, bp_te, n1, n2, delta_xi_te))
    bpi_ff_simple = np.append(bpi_ff_simple, bp_te)

    return bpi_ff_simple


def bpi_ff_simple_2(r_hi_hub: float, r_18_hub: float, teta_te: float, l_ff_stage: float):
    """Compute Bernstein polynomial coefficients, i.e. weighting coefficients of shape function, for fuselage section
    between center section and intake.
    Parameters
    ----------
    :param r_cent_f: (float):   [m]     Fuselage center section radius
    :param r_hi_hub: (float):   [m]     Fuselage radius at highlight
    :param teta_te: (float)     [°]     TE angle of section
    :param l_preint: (float)    [m]     Length of section

    Returns
    -------
    bpi_fus_preint : np.array
        Bernstein polynomial coefficients for given set of constraints
    Notes
    -----
    References
    ----------
    [1] & [2]
    """

    # normalize all variables by nacelle length
    r_hi_hub /= l_ff_stage
    r_18_hub /= l_ff_stage
    r_le = r_hi_hub
    r_te = r_18_hub

    # calculate leading edge coefficient. first coefficient defined by leading edge radius. Needs to be normalized with nacelle length
    bp_le = -(r_te - r_le)
    # calculate trailing edge coefficent. last coefficient defined by boattail angle.
    bp_te = np.tan(teta_te) - (r_te - r_le)
    # populate bpi array
    bpi_ff_simple = np.array([bp_le])
    # add all other coefficients by solving linear system of equations
    # create list with constraints
    constraints = [(2, 0, 0),  # curvature at fuselage FF stage outlet
                   (2, 1, 0)  # curvature at fuselage TE
                   ]
    n1 = 1
    n2 = 1
    # calculate trailing edge position delta
    delta_xi_te = -(r_te - r_le)
    bpi_ff_simple = np.append(bpi_ff_simple, bpi_solve(constraints, bp_le, bp_te, n1, n2, delta_xi_te))
    bpi_ff_simple = np.append(bpi_ff_simple, bp_te)

    return bpi_ff_simple
