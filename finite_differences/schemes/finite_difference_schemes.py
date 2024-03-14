"""2D finite difference schemes.

Author:  A. Habermann
"""


# First derivatives
def d1o1_left(dim, x, i, j):  # left-sided, first order
    if dim == 'xi':
        return x[i,j]-x[i-1,j]
    elif dim == 'eta':
        return x[i, j] - x[i, j-1]


def d1o2_left(dim, x, i, j):  # left-sided, second order
    if dim == 'xi':
        return (1/2)*(3*x[i,j]-4*x[i-1,j]+x[i-2,j])
    elif dim == 'eta':
        return (1/2)*(3*x[i,j]-4*x[i,j-1]+x[i,j-2])


def d1o1_right(dim, x, i, j):  # right-sided, first order
    if dim == 'xi':
        return x[i+1,j]-x[i,j]
    elif dim == 'eta':
        return x[i, j+1] - x[i, j]


def d1o2_right(dim, x, i, j):  # right-sided, second order
    if dim == 'xi':
        return (1/2)*(-3*x[i,j]+4*x[i+1,j]-x[i+2,j])
    elif dim == 'eta':
        return (1/2)*(-3*x[i,j]+4*x[i,j+1]-x[i,j+2])


def d1o2_cent(dim, x, i, j):  # central, second order
    if dim == 'xi':
        return (1/2)*(x[i + 1, j] - x[i-1, j])
    elif dim == 'eta':
        return (1/2)*(x[i, j + 1] - x[i, j-1])


def d1o4_cent(dim, x, i, j):  # central, fourth order
    if dim == 'xi':
        return (1 / 12) * (-x[i+2, j] + 8 * x[i + 1, j] - 8*x[i -1, j]+x[i-2,j])
    elif dim == 'eta':
        return (1 / 12) * (-x[i, j+2] + 8 * x[i, j+1] - 8*x[i, j-1]+x[i,j-2])


# Second derivatives
def d2o1_left(dim, x, i, j):  # left-sided, first order
    if dim == 'xi':
        return x[i,j]-2*x[i-1,j]+x[i-2,j]
    elif dim == 'eta':
        return x[i,j]-2*x[i,j-1]+x[i,j-2]


def d2o2_left(dim, x, i, j):  # left-sided, second order
    if dim == 'xi':
        return 2*x[i,j]-5*x[i-1,j]+4*x[i-2,j]-x[i-3,j]
    elif dim == 'eta':
        return 2*x[i,j]-5*x[i,j-1]+4*x[i,j-2]-x[i,j-3]


def d2o1_right(dim, x, i, j):  # right-sided, first order
    if dim == 'xi':
        return x[i+2,j]-2*x[i+1,j]+x[i,j]
    elif dim == 'eta':
        return x[i,j+2]-2*x[i,j+1]+x[i,j]


def d2o2_right(dim, x, i, j):  # right-sided, second order
    if dim == 'xi':
        return 2*x[i,j]-5*x[i+1,j]+4*x[i+2,j]-x[i+3,j]
    elif dim == 'eta':
        return 2*x[i,j]-5*x[i,j+1]+4*x[i,j+2]-x[i,j+3]


def d2o2_cent(dim, x, i, j):  # central, second order
    if dim == 'xi':
        return x[i+1,j]-2*x[i,j]+x[i-1,j]
    elif dim == 'eta':
        return x[i,j+1]-2*x[i,j]+x[i,j-1]


def d2o4_cent(dim, x, i, j):  # central, fourth order
    if dim == 'xi':
        return (1/12)*(-x[i+2,j]+16*x[i+1,j]-30*x[i,j]+16*x[i-1,j]-x[i-2,j])
    elif dim == 'eta':
        return (1/12)*(-x[i,j+2]+16*x[i,j+1]-30*x[i,j]+16*x[i,j-1]-x[i,j-2])


def d202_mix_cent(x, i, j):  # mixed, central, second order
    return (1/4)*(x[i+1,j+1]-x[i+1,j-1]+x[i-1,j-1]-x[i-1,j+1])


def d202_mix_cent_left(dim, x, i, j):  # mixed, central and left combined, second order
    if dim == 'xi':         # left-sided derivative in xi-direction, central derivative in eta-direction
        return (1/4)*((3*x[i,j+1]-4*x[i-1,j+1]+x[i-2,j+1])-(3*x[i,j-1]-4*x[i-1,j-1]+x[i-2,j-1]))
    elif dim == 'eta':         # left-sided derivative in eta-direction, central derivative in xi-direction
        return (1/4)*((3*x[i+1,j]-4*x[i+1,j-1]+x[i+1,j-2])-(3*x[i-1,j]-4*x[i-1,j-1]+x[i-1,j-2]))


def d202_mix_cent_right(dim, x, i, j):  # mixed, central and right combined, second order
    if dim == 'xi':         # right-sided derivative in xi-direction, central derivative in eta-direction
        return (1/4)*((-3*x[i,j+1]+4*x[i+1,j+1]-x[i+2,j+1])-(-3*x[i,j-1]+4*x[i+1,j-1]-x[i+2,j-1]))
    elif dim == 'eta':         # right-sided derivative in eta-direction, central derivative in xi-direction
        return (1/4)*((-3*x[i+1,j]+4*x[i+1,j+1]-x[i+1,j+2])-(-3*x[i-1,j]+4*x[i-1,j+1]-x[i-1,j+2]))


def d202_mix_right_right(x, i, j):  # mixed, right and right combined, second order
    return (1/4)*(-3*(-3*x[i,j]+4*x[i,j+1]-x[i,j+2])
                  +4*(-3*x[i+1,j]+4*x[i+1,j+1]-x[i+1,j+2])
                  -(-3*x[i+2,j]+4*x[i+2,j+1]-x[i+2,j+2]))


def d202_mix_left_left(x, i, j):  # mixed, left and left combined, second order
    return (1/4)*(3*(3*x[i,j]-4*x[i,j-1]+x[i,j-2])
                  -4*(3*x[i-1,j]-4*x[i-1,j-1]+x[i-1,j-2])
                  +(3*x[i-2,j]-4*x[i-2,j-1]+x[i-2,j-2]))


def d202_mix_left_right(dim, x, i, j):  # mixed, left and right combined, second order
    if dim == 'xi':         # right-sided derivative in xi-direction, left-sided derivative in eta-direction
        return (1/4)*(-3*(3*x[i,j]-4*x[i,j-1]+x[i,j-2])
                      +4*(3*x[i+1,j]-4*x[i+1,j-1]+x[i+1,j-2])
                      -(3*x[i+2,j]-4*x[i+2,j-1]+x[i+2,j-2]))
    elif dim == 'eta':         # right-sided derivative in eta-direction, left-sided derivative in xi-direction
        return (1/4)*(3*(-3*x[i,j]+4*x[i,j+1]-x[i,j+2])
                      -4*(-3*x[i-1,j]+4*x[i-1,j+1]-x[i-1,j+2])
                      +(-3*x[i-2,j]+4*x[i-2,j+1]-x[i-2,j+2]))
