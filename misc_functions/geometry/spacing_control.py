"""Spacing control for boundaries using minimum wall distance.

Author:  A. Habermann

"""

# Built-in/Generic Imports
import numpy as np
from scipy.optimize import fsolve


# calculate spacing of lines if  no. of lines unspecified, ratio specified
def ratio_spacing(y_min, y_max, delta_y_wall, ratio):
    """
        delta_y_wall:               wall spacing
        y_min:                      minimum y-value
        y_max:                      maximum y-value
        ratio:                      ratio between y-values
    """

    spacing = np.array([y_min])
    delta = np.array([delta_y_wall])
    spacing = np.append(spacing, spacing[-1]+delta[-1])

    while spacing[-1] < y_max:
        delta = np.append(delta, ratio*delta[-1])
        spacing = np.append(spacing, spacing[-1]+delta[-1])

    spacing[-1] = y_max
    n = len(spacing)

    return spacing, n


# calculate spacing of lines if no. of lines specified, ratio unspecified
def number_spacing(y_min: float, y_max: float, delta_y_wall: float, n_y: int, line, n_line = 0):
    """
        y_min:                      minimum y-value
        y_max:                      maximum y-value
        delta_y_wall:               wall spacing
        n_y:                        number of lines in y-direction
        line:                       y-position of line at which wall spacing is specified
        n_line:                     number of line (counted from top) at which wall spacing is specified
    """
    init = 1.5

    if line == y_min:
        def func(x):
            sum_ratio = 1
            for j in range(1,n_y-1):
                sum_ratio += x**j
            return (y_max-y_min)/delta_y_wall-sum_ratio

        ratio = fsolve(func, np.array([init]))
        while ratio == init:
            init += 0.5
            ratio = fsolve(func, np.array([init]))

        spacing = np.array([y_min])
        delta = np.array([delta_y_wall])
        spacing = np.append(spacing, spacing[-1]+delta[-1])

        for i in range(1,n_y-1):
            delta = np.append(delta, ratio*delta[-1])
            spacing = np.append(spacing, spacing[-1]+delta[-1])

        spacing[-1] = y_max

    elif line == y_max:
        def func(x):
            sum_ratio = 1
            for j in range(1,n_y-1):
                sum_ratio += x**j
            return (y_max-y_min)/delta_y_wall-sum_ratio

        ratio = fsolve(func, np.array([init]))
        while ratio == init:
            init += 0.5
            ratio = fsolve(func, np.array([init]))

        spacing = np.array([y_max])
        delta = np.array([delta_y_wall])
        spacing = np.insert(spacing, 0, spacing[0]-delta[0])

        for i in range(1,n_y-1):
            delta = np.insert(delta, 0, ratio*delta[0], axis=0)
            spacing = np.insert(spacing, 0, spacing[0]-delta[0], axis=0)

        spacing[0] = y_min

    else:
        n_y_up = n_line-1
        n_y_low = n_y-n_y_up

        def func_low(x):
            sum_ratio = 1
            for j in range(1,n_y_low-1):
                sum_ratio += x**j
            return (line-y_min)/delta_y_wall-sum_ratio

        def func_up(x):
            sum_ratio = 1
            for j in range(1,n_y_up):
                sum_ratio += x**j
            return (y_max-line)/delta_y_wall-sum_ratio

        ratio_low = fsolve(func_low, np.array([init]))
        while ratio_low == init:
            init += 0.5
            ratio_low = fsolve(func_low, np.array([init]))

        init = 1.5

        ratio_up = fsolve(func_up, np.array([init]))
        while ratio_up == init:
            init += 0.5
            ratio_up = fsolve(func_up, np.array([init]))

        spacing_low = np.array([line])
        delta_low = np.array([delta_y_wall])
        spacing_low = np.insert(spacing_low, 0, spacing_low[0]-delta_low[0])

        spacing_up = np.array([line])
        delta_up = np.array([delta_y_wall])
        spacing_up = np.append(spacing_up, spacing_up[-1]+delta_up[-1])

        for i in range(1, n_y_low-1):
            delta_low = np.insert(delta_low, 0, ratio_low*delta_low[0], axis=0)
            spacing_low = np.insert(spacing_low, 0, spacing_low[0]-delta_low[0], axis=0)

        for i in range(1, n_y_up):
            delta_up = np.append(delta_up, ratio_up*delta_up[-1])
            spacing_up = np.append(spacing_up, spacing_up[-1]+delta_up[-1])

        spacing_low[0] = y_min
        spacing_up[-1] = y_max

        spacing = np.concatenate((spacing_low[0:-1], spacing_up))
        ratio = np.concatenate((ratio_low, ratio_up))

    return spacing, ratio


# calculate spacing between two boundaries if no. of lines specified, ratio unspecified, with "bump"
def number_spacing_bump(y1, y2, delta_y_wall1, delta_y_wall2, n_y):
    """
        y1:                         y-value of wall1
        y2:                         y-value of wall2
        delta_y_wall1:              wall spacing at wall1
        delta_y_wall2:              wall spacing at wall2
        n_y:                        number of lines in y-direction
    """
    init = 1.5
    if 50 > n_y > 10:
        n_y1 = min(int(round(n_y*(delta_y_wall1/(delta_y_wall1+delta_y_wall2)))),n_y-5)
    elif n_y > 50:
        n_y1 = min(int(round(n_y*(delta_y_wall1/(delta_y_wall1+delta_y_wall2)))),n_y-10)
    else:
        n_y1 = int(round(n_y * (delta_y_wall1 / (delta_y_wall1 + delta_y_wall2))))
    n_y2 = n_y - n_y1

    def func1(x):
        sum_ratio = 1
        for j in range(1, n_y1):
            sum_ratio += x ** j
        return (delta_y_wall1/(delta_y_wall2+delta_y_wall1)*(y2 - y1)) / delta_y_wall1 - sum_ratio

    def func2(x):
        sum_ratio = 1
        for j in range(1, n_y2-1):
            sum_ratio += x ** j
        return (delta_y_wall2/(delta_y_wall2+delta_y_wall1)*(y2 - y1)) / delta_y_wall2 - sum_ratio

    ratio1 = fsolve(func1, np.array([init]))
    while ratio1 == init:
        init += 0.5
        ratio1 = fsolve(func1, np.array([init]))

    init = 1.5

    ratio2 = fsolve(func2, np.array([init]))
    while ratio2 == init:
        init += 0.5
        ratio2 = fsolve(func2, np.array([init]))

    spacing = np.array([y1, y2])
    delta = np.array([delta_y_wall1,delta_y_wall2])

    spacing = np.insert(spacing, 1, spacing[0] + delta[0], axis=0)
    spacing = np.insert(spacing, -1, spacing[-1] - delta[-1], axis=0)

    i = 1
    while i < n_y1-1:
        delta = np.insert(delta, i, ratio1*delta[i-1])
        spacing = np.insert(spacing, i+1, spacing[i] + delta[i])
        i += 1

    i = 1
    while i < n_y2-1:
        delta = np.insert(delta, -i, ratio2 * delta[-i], axis=0)
        spacing = np.insert(spacing, -i - 1, spacing[-i - 1] - delta[-i - 1])
        i += 1

    if len(spacing) != n_y:
        raise Warning('Spacing failed.')

    return spacing, ratio1, ratio2


# calculate spacing between two boundaries if no. of lines specified, ratio unspecified, with "bump"
def ratio_spacing_bump(y1, y2, delta_y_wall1, delta_y_wall2, ratio1, ratio2):
    """
        y1:                         y-value of wall1
        y2:                         y-value of wall2
        delta_y_wall1:              wall spacing at wall1
        delta_y_wall2:              wall spacing at wall2
        ratio1:                     ratio between y-values at wall1
        ratio2:                     ratio between y-values at wall2
    """

    spacing = np.array([y1,y2])
    delta = np.array([delta_y_wall1,delta_y_wall2])
    spacing = np.insert(spacing,1,delta_y_wall1+spacing[0])
    spacing = np.insert(spacing,-1,spacing[-1]-delta_y_wall2)

    i = 1
    if y1 < y2:
        while spacing[i] < spacing[-i-1]:
            delta = np.insert(delta, i, ratio1*delta[i-1], axis=0)
            delta = np.insert(delta, -i, ratio2*delta[-i], axis=0)
            spacing = np.insert(spacing, i+1, spacing[i]+delta[i], axis=0)
            spacing = np.insert(spacing, -i-1, spacing[-i-1]-delta[-i], axis=0)
            i += 1

    elif y1 > y2:
        while spacing[i] > spacing[-i-1]:
            delta = np.insert(delta, i, ratio1*delta[i-1], axis=0)
            delta = np.insert(delta, -i, ratio2*delta[-i], axis=0)
            spacing = np.insert(spacing, i+1, spacing[i]-delta[i], axis=0)
            spacing = np.insert(spacing, -i-1, spacing[-i-1]+delta[-i], axis=0)
            i += 1

    j = i-1
    spacing = np.delete(spacing, j+1)
    spacing = np.delete(spacing, -j-2)
    n = len(spacing)

    return spacing, n


if __name__ == "__main__":
    x, no = ratio_spacing_bump(1.0,2.0,0.001,0.002,1.1,1.2)
    x2, no2 = ratio_spacing_bump(2.0,1.0,0.001,0.002,1.1,1.2)
    import matplotlib.pyplot as plt
    plt.scatter(x,[0]*len(x))
    plt.scatter(x2,[1]*len(x2))
    plt.show()
