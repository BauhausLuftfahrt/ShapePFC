import numpy as np
import ctypes as C
from scipy.interpolate import griddata


def interp_cubic2d(z_orig, fillv):
    data_pointer_coords = C.cast(CAddress, C.POINTER(C.c_double))
    coords = np.ctypeslib.as_array(data_pointer_coords, shape=(SIZE, 3))
    z_cubic = griddata(tuple((x_orig, y_orig)), z_orig, (coords[:, 0], coords[:, 2]), method='cubic', fill_value=fillv)
    return z_cubic


def interp_lin2d(z_orig, fillv):
    data_pointer_coords = C.cast(CAddress, C.POINTER(C.c_double))
    coords = np.ctypeslib.as_array(data_pointer_coords, shape=(SIZE, 3))
    z_lin = griddata(tuple((x_orig, y_orig)), z_orig, (coords[:, 0], coords[:, 2]), method='linear', fill_value=fillv)
    return z_lin


def interp_nearest2d(z_orig, fillv):
    data_pointer_coords = C.cast(CAddress, C.POINTER(C.c_double))
    coords = np.ctypeslib.as_array(data_pointer_coords, shape=(SIZE, 3))
    z_nearest = griddata(tuple((x_orig, y_orig)), z_orig, (coords[:, 0], coords[:, 2]), method='nearest', fill_value=fillv)
    return z_nearest
