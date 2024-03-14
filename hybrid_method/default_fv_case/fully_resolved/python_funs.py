import numpy as np
import ctypes as C
from scipy.interpolate import griddata
import scipy.interpolate as interpolate
import csv


def interp_cubic2d(x_orig, y_orig, z_orig, fillv):
    data_pointer_coords = C.cast(CAddress, C.POINTER(C.c_double))
    coords = np.ctypeslib.as_array(data_pointer_coords, shape=(SIZE, 3))
    z_cubic = griddata(tuple((x_orig, y_orig)), z_orig, (coords[:, 0], coords[:, 2]), method='cubic', fill_value=fillv)
    return z_cubic


def interp_lin2d(x_orig, y_orig, z_orig, fillv):
    data_pointer_coords = C.cast(CAddress, C.POINTER(C.c_double))
    coords = np.ctypeslib.as_array(data_pointer_coords, shape=(SIZE, 3))
    z_lin = griddata(tuple((x_orig, y_orig)), z_orig, (coords[:, 0], coords[:, 2]), method='linear', fill_value=fillv)
    return z_lin


def interp_nearest2d(z_orig, fillv):
    data_pointer_coords = C.cast(CAddress, C.POINTER(C.c_double))
    coords = np.ctypeslib.as_array(data_pointer_coords, shape=(SIZE, 3))
    z_nearest = griddata(tuple((x_orig, y_orig)), z_orig, (coords[:, 0], coords[:, 2]), method='nearest', fill_value=fillv)
    return z_nearest


def interface_u(z_orig):
	with open('u_interface_data.csv', 'r', newline='') as f:
		csv_reader = csv.reader(f)
		header = next(csv_reader)
		data = []
		for row in csv_reader:
			record = {}
			for i, value in enumerate(row):
				record[header[i]] = value
			data.append(record)
	F_ux = interpolate.UnivariateSpline([float(i['y']) for i in data], [float(i['u_x']) for i in data], s=0)
	F_uz = interpolate.UnivariateSpline([float(i['y']) for i in data], [float(i['u_z']) for i in data], s=0)
	return [float(F_ux(z_orig)), 0.0, float(F_uz(z_orig))]


def interface_t(z_orig):
	with open('t_interface_data.csv', 'r', newline='') as f:
		csv_reader = csv.reader(f)
		header = next(csv_reader)
		data = []
		for row in csv_reader:
			record = {}
			for i, value in enumerate(row):
				record[header[i]] = value
			data.append(record)
	F_t = interpolate.UnivariateSpline([float(i['y']) for i in data], [float(i['t']) for i in data], s=0)
	return float(F_t(z_orig))
