"""Test grid generation for two different kinds of grid.

Author:  A. Habermann
"""


from finite_differences.mesh.grid_generation_nacelle import gridgeneration_test

# test nacelle grid generation
x_grid, y_grid, alpha, beta, gamma, tau, omega, _, _, _, _ = gridgeneration_test("PFC nacelle", "slit", "planar")

# test akron airship grid generation
x_grid_2, y_grid_2, alpha_2, beta_2, gamma_2, tau_2, omega_2, _, _, _, _ = gridgeneration_test("Akron", "rect", "planar")
