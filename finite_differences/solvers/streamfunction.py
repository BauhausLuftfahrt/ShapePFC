"""
Author:  A. Habermann

Solve transformed stream function on boundary-fitted grid with finite differences.

Source:

[0] Thompson et al 1977: Boundary-fitted curvilinear coordinate systems for solution of partial differential equations on
    fields containing any number of arbitrary two-dimensional bodies.
"""


import numpy as np
import matplotlib.pyplot as plt

""" 
Psi         Stream function
"""


def initialize_Psi(n_eta, n_xi, Psi_init, Psi_bc):
    # Initialize solution: The grid of Psi(i, j)
    Psi = np.full((n_eta, n_xi), Psi_init)

    # Set boundary conditions
    Psi[0,:] = Psi_bc[0]        # top
    Psi[:,0] = Psi_bc[1]        # left
    Psi[-1,:] = Psi_bc[2]        # bottom
    Psi[:,-1] = Psi_bc[3]        # right

    return Psi


def calculate_slow(Psi, alpha, beta, gamma, omega, tau):
    n_i, n_j = Psi.shape
    for i in range(1, n_i-1):
        for j in range(1, n_j-1):
            Psi[i,j] = (1/(2*alpha[i,j]+2*gamma[i,j]))*\
                       ((-0.5*beta[i,j])*Psi[i-1,j-1]+(alpha[i,j]-0.5*tau[i,j])*Psi[i-1,j]+(0.5*beta[i,j])*Psi[i-1,j+1]+
                       (gamma[i,j]-0.5*omega[i,j])*Psi[i,j-1]+(gamma[i,j]+0.5*omega[i,j])*Psi[i,j+1]+
                       (0.5*beta[i,j])*Psi[i+1,j-1]+(alpha[i,j]+0.5*tau[i,j])*Psi[i+1,j]+(-0.5*beta[i,j])*Psi[i+1,j+1])
    return Psi


def kernel(i, j, alpha, beta, gamma, omega, tau):
    kernel = np.array([
        [-0.5*beta[i,j], alpha[i,j]-0.5*tau[i,j], 0.5*beta[i,j]],
        [gamma[i,j]-0.5*omega[i,j],-2*alpha[i,j]-2*gamma[i,j],gamma[i,j]+0.5*omega[i,j]],
        [0.5*beta[i,j], alpha[i,j]+0.5*tau[i,j], -0.5*beta[i,j]]
    ])
    return kernel


def calculate_fast(Psi, alpha, beta, gamma, omega, tau):
    n_i, n_j = Psi.shape
    # # we get a 4D array that contains all possible 3x3 local maps
    local_maps = np.lib.stride_tricks.sliding_window_view(Psi, (3,3))

    # sum the product of the kernel and each map
    # and sum each local map


def calculate_faster(Psi, alpha, beta, gamma, omega, tau):
    # see also: https: // towardsdatascience.com / 300 - times - faster - resolution - of - finite - difference - method - using - numpy - de28cdade4e1
    A = -0.5*beta[1:-1,1:-1]*Psi[:-2,2:]
    B = (alpha[1:-1,1:-1]-0.5*tau[1:-1,1:-1])*Psi[:-2,1:-1]
    C = (0.5*beta[1:-1,1:-1])*Psi[:-2,2:]
    D = (gamma[1:-1,1:-1]-0.5*omega[1:-1,1:-1])*Psi[1:-1,2:]
    E = 0*Psi[1:-1, 1:-1]
    F = (gamma[1:-1,1:-1]+0.5*omega[1:-1,1:-1])*Psi[1:-1, :-2]
    G = (0.5*beta[1:-1,1:-1])*Psi[2:, 2:]
    H = (alpha[1:-1,1:-1]+0.5*tau[1:-1,1:-1])*Psi[2:, 1:-1]
    J = -0.5*beta[1:-1,1:-1]*Psi[2:, :-2]

    Coeff = 1/(2*(alpha[1:-1,1:-1]+gamma[1:-1,1:-1]))

    result = Coeff*(A+B+C+D+F+G+H+J)
    Psi[1:-1, 1:-1]+result
    return Psi
