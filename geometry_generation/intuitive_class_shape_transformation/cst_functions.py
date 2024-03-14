""" Source: https://github.com/daniel-de-vries/cst [adapted]
Functions required for Class/Shape Transformation.
"""

import numpy as np
import scipy.optimize as opt

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import binom
from typing import Union, List, Tuple, Optional


def cls(
        x: Union[float, List[float], np.ndarray], n1: float, n2: float, norm: bool = False
) -> np.ndarray:
    """
    Compute class function.
    Parameters
    ----------
    x : array_like
        Points to evaluate class function for
    n1, n2 : int
        Class function parameters
    norm : bool, optional
        True (default) if the class function should be normalized
    Returns
    -------
    np.array
        Class function value for each given point
    Notes
    -----
    It is assumed that 0 <= x <= 1.
    """
    c = (x ** n1) * ((1.0 - x) ** n2)
    c /= (
        1.0
        if not norm or n1 == n2 == 0
        else (((n1 / (n1 + n2)) ** n1) * ((n2 / (n1 + n2)) ** n2))
    )
    return c


def cls_deriv1(
        x: Union[float, List[float], np.ndarray], n1: float, n2: float, norm: bool = False
) -> np.ndarray:
    """
    Compute class function's first derivative
    Parameters
    ----------
    x : array_like
        Points to evaluate class function for
    n1, n2 : int
        Class function parameters
    norm : bool, optional
        True (default) if the class function should be normalized
    Returns
    -------
    np.array
        Class function 1st derivative value for each given point
    Notes
    -----
    It is assumed that 0 <= x <= 1.
    """
    if n1 == 0.5 and n2 == 1.0:
        c1 = (1 - 3 * x) / (2 * np.sqrt(x))
    elif n1 == 1.0 and n2 == 1.0:
        c1 = 1 - 2 * x
    else:
        c1 = (x ** (n1 - 1)) * (-(1 - x) ** (n2 - 1)) * (n1 * (x - 1) + n2 * x)
    c1 /= (
        1.0
        if not norm or n1 == n2 == 0
        else (((n1 / (n1 + n2)) ** n1) * ((n2 / (n1 + n2)) ** n2))
    )

    return c1


def cls_deriv2(
        x: Union[float, List[float], np.ndarray], n1: float, n2: float, norm: bool = False
) -> np.ndarray:
    """
    Compute class function's second derivative (n1=0.5, n2=1)
    Parameters
    ----------
    x : array_like
        Points to evaluate class function for
    n1, n2 : int
        Class function parameters
    norm : bool, optional
        True (default) if the class function should be normalized
    Returns
    -------
    np.array
        Class function 2nd derivative value for each given point
    Notes
    -----
    It is assumed that 0 <= x <= 1.
    """
    if n1 == 0.5 and n2 == 1.0:
        c2 = (-3 * x - 1) / (4 * x ** (3 / 2))
    elif n1 == 1.0 and n2 == 1.0:
        c2 = -2
    else:
        c2 = ((n1 - 1) * n1 * (x ** (n1 - 2)) * ((1 - x) ** n2) - 2 * n1 * n2 * (x ** (n1 - 1)) * (
                (1 - x) ** (n2 - 1))) + ((n2 - 1) * n2 * (x ** n1) * ((1 - x) ** (n2 - 2)))
    c2 /= (
        1.0
        if not norm or n1 == n2 == 0
        else (((n1 / (n1 + n2)) ** n1) * ((n2 / (n1 + n2)) ** n2))
    )
    return c2


def bernstein(x: Union[float, List[float], np.ndarray], r: int, n: int) -> np.array:
    """
    Compute Bernstein basis polynomial.
    Parameters
    ----------
    x : array_like
        Points to evaluate the Bernstein polynomial at
    r, n : int
        Bernstein polynomial index and degree
    Returns
    -------
    np.array
        Values of the Bernstein polynomial at the given points
    Notes
    -----
    It is assumed that 0 <= x < 1.
    It is assumed that r <= n.
    """
    bern = (
        0.0
        if (r < 0 or r > n)  # important for calculation of 1st and 2nd derivative
        else binom(n, r) * (x ** r) * ((1.0 - x) ** (n - r))
    )

    return bern


def bernstein_deriv1(x: Union[float, List[float], np.ndarray], r: int, n: int) -> np.array:
    """
    Compute Bernstein basis polynomial first derivative.
    Parameters
    ----------
    x : array_like
        Points to evaluate the Bernstein polynomial at
    r, n : int
        Bernstein polynomial index and degree
    Returns
    -------
    np.array
        Values of the Bernstein polynomial 1st derivative at the given points
    Notes
    -----
    It is assumed that 0 <= x < 1.
    It is assumed that r <= n.
    """
    return n * (bernstein(x, r - 1, n - 1) - bernstein(x, r, n - 1))


def bernstein_deriv2(x: Union[float, List[float], np.ndarray], r: int, n: int) -> np.array:
    """
    Compute Bernstein basis polynomial second derivative.
    Parameters
    ----------
    x : array_like
        Points to evaluate the Bernstein polynomial at
    r, n : int
        Bernstein polynomial index and degree
    Returns
    -------
    np.array
        Values of the Bernstein polynomial 2nd derivative at the given points
    Notes
    -----
    It is assumed that 0 <= x < 1.
    It is assumed that r <= n.
    """
    k = 2
    import math

    # calculate 2nd derivative for x = 0
    def bern2_0(X, i, j):
        return binom(k, i) * math.factorial(j) / math.factorial(j - k) * ((-1) ** (i + k))

    if x == 0:
        bern2 = bern2_0(x, r, n)
    elif x == 1:
        bern2 = (-1) ** k * bern2_0(x, n - r, n)
    else:
        bern2 = n ** 2 * (bernstein(x, r - 2, n - 2) - 2 * bernstein(x, r - 1, n - 2) + bernstein(x, r, n - 2))
    return bern2


def cst(
        x: Union[float, List[float], np.ndarray],
        a: Union[List[float], np.ndarray],
        delta: Tuple[float, float] = (0.0, 0.0),
        n1: float = 0.5,
        n2: float = 1.0,
) -> np.ndarray:
    """Compute coordinates of a CST-decomposed curve.
    This function uses the Class-Shape Transformation (CST) method to compute the y-coordinates as a function of a given
    set of x-coordinates, `x`, and a set of coefficients, `a`. The x-coordinates can be scaled by providing a length
    scale, `c`. The starting and ending points of the curve can be displaced by providing non-zero values for `delta`.
    Finally, the class of shapes generated can be adjusted with the `n1` and `n2` parameters. By default, these are 0.5
    and 1.0 respectively, which are good values for generating airfoil shapes.
    Parameters
    ----------
    x : float or array_like
        x-coordinates.
    a : array_like
        Bernstein coefficients.
    delta : tuple of two floats
        Vertical displacements of the start- and endpoints of the curve. Default is (0., 0.).
    n1, n2 : float
        Class parameters. These determine the helpers "class" of the shape. They default to n1=0.5 and n2=1.0 for
        airfoil-like shapes.
    Returns
    -------
    y : np.ndarray
        Y-coordinates.
    Notes
    -----
    It is assumed that 0 <= x <= 1.
    References
    ----------
    [1] Brenda M. Kulfan, '"CST" Universal Parametric Geometry Representation Method With Applications to Supersonic
     Aircraft,' Fourth International Conference on Flow Dynamics, Sendai, Japan, September 2007.
    """
    # Ensure x is a numpy array
    x = np.atleast_1d(x)

    # Bernstein polynomial degree
    n = len(a) - 1

    # Compute Class and Shape functions
    _class = cls(x, n1, n2, norm=False)
    _shape = sum(a[r] * bernstein(x, r, n) for r in range(len(a)))

    # Compute y-coordinates
    y = _class * _shape + (1.0 - x) * delta[0] + x * delta[1]
    return y


def fit(
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
        n: int,
        delta: Optional[Tuple[float, float]] = None,
        n1: float = 0.5,
        n2: float = 1.0,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Fit a set of coordinates to a CST representation.
    Parameters
    ----------
    x, y : array_like
        X- and y-coordinates of a curve.
    n : int
        Number of Bernstein coefficients.
    delta : tuple of two floats, optional
        Manually set the start- and endpoint displacements.
    n1, n2 : float
        Class parameters. Default values are 0.5 and 1.0 respectively.
    Returns
    -------
    A : np.ndarray
        Bernstein coefficients describing the curve.
    delta : tuple of floats
        Displacements of the start- and endpoints of the curve.
    Notes
    -----
    It is assumed that 0 <= x <= 1.
    """
    # Ensure x and y are np.ndarrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Make sure the coordinates are sorted by x-coordinate
    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    if delta is None:
        f = InterpolatedUnivariateSpline(x, y)
        delta = (f(0.0), f(1.0))

    def f(_a):
        return np.sqrt(np.mean((y - cst(x, _a, delta=delta, n1=n1, n2=n2)) ** 2))

    # Fit the curve
    res = opt.minimize(f, np.zeros(n))

    return res.x, delta
