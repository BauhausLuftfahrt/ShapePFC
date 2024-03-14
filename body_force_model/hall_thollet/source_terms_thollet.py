"""Calculate source terms, which are included in RANS equations to model rotor and stator stages according to
Hall-Thollet.

Author: A. Habermann
Date:   06.04.2022

 Args:
    rho             [kg/m^3] Local fluid density
    mu              [kg/m s] Local fluid dynamic viscosity
    sos             [m/s]   Local fluid speed of sound
    v_abs           [m/s]   Local absolute velocity
    blade_normal            Local blade profile normal [n_x, n_y, n_teta]
    blade_coordinates       x, r, teta coordinates of blade body_force_model
    blockage                Local blade metall blockage factor
    number_blades   [-]     Total number of rotor/stator blades
    rot_speed       [rad/s] Rotor rotational speed

Returns:
    f_vector        [N/kg]  Local force source term vector in x, r, teta coordinates

Sources:
    [0] Hall, D. K.; Greitzer, E.; Tan, C.: Analysis of Fan Stage Conceptual Design Attributes for Boundary Layer 
    Ingestion (2017).
    [1] Thollet, W.: Body force modeling of fan-airframe interactions, Dissertation, Universite Federale Toulouse 
    Midi-Pyrenees (2017).
    [2] Benichou, E.; Dufour, G.; Bousquet, Y.; Binder, N.; Ortolan, A.; Carbonneau, X.: Body Force Modeling of the 
    Aerodynamics of a Low-Speed Fan under Distorted Inflow, IJTPP 4(3) (2019).
    [3] Godard, B.; de Jaeghere, E.; Gourdain, N.: Efficient Design Investigation of a Turbofan in Distorted Inlet 
    Conditions, Int. Gas Turb. Inst. (2019)
    [4] Matesanz-Garcia, J.; Piovesan, T.; MacManus, D. G.: Aerodynamic optimization of the exhaust system of an 
    aft-mounted boundary layer ingestion propulsor (2022).

"""

# Built-in/Generic Imports
import numpy as np


class SourceTermsThollet:

    def __init__(self, rho, mu, sos, v_abs, blade_normal, blade_coordinates, blockage, number_blades, rot_speed):
        self.rho = rho
        self.mu = mu
        self.sos = sos
        self.v_abs = v_abs
        self.x_chord = blade_coordinates[0]
        self.blockage = blockage
        self.number_blades = number_blades
        self.r = blade_coordinates[1]  # local radius
        self.rot_speed = rot_speed
        self.blade_normal = blade_normal
        self.blade_tangent = None
        self.v_rel = None
        self.K = None
        self.delta = None
        self.pitch = None
        self.Re_chord = None
        self.c_f = None
        self.Ma_rel = None

    def run(self):
        # calculate required parameters
        self.v_rel = calc_rel_velocity(self)
        self.Ma_rel = calc_Mach_rel(self)
        self.K = calc_K_mach(self)
        self.delta = calc_delta(self)
        self.pitch = calc_pitch(self)
        self.Re_chord = calc_Re_local(self)
        self.c_f = calc_C_f(self)

        # calculate local force vector normal and parallel to the flow
        f_n = calc_f_n(self)
        f_p = calc_f_p(self)

        # calculate tangent vector
        self.blade_tangent = np.cross(self.blade_normal, [0, -1, 0])

        # transform f_n and f_p to x, r, teta coordinate system
        f_x = self.blade_tangent[0] * (-np.cos(self.delta) * f_p + np.sin(self.delta) * f_n) + \
              self.blade_normal[0] * (-np.sin(self.delta) * f_p - np.cos(self.delta) * f_n)
        f_r = 0
        f_teta = self.blade_tangent[2] * (-np.cos(self.delta) * f_p + np.sin(self.delta) * f_n) + \
                 self.blade_normal[2] * (-np.sin(self.delta) * f_p - np.cos(self.delta) * f_n)

        f_vector = [f_x, f_r, f_teta]

        return f_vector


def calc_f_n(self):
    """Compute local force vector normal to the relative flow [N/kg] [2], equ. (8)
    Author:  Anais Habermann
     Args:
        self
    Returns:
        f_n         local force vector normal to the relative flow
    """

    return self.K * np.pi * self.delta * np.linalg.norm(self.v_rel) ** 2 / (self.pitch * self.blockage *
                                                                            np.abs(self.blade_normal[2]))


def calc_f_p(self):
    """Compute local force vector parallel to the relative flow [N/kg] [2], equ. (9)
    Author:  Anais Habermann
     Args:
        self
    Returns:
        f_p         local force vector normal to the relative flow
    """
    delta_ref = self.delta  # Assumption: Only considering design points for which there is no
    # deviation from the local deviation angle delta (no off-design conditions)
    return 0.5 * np.linalg.norm(self.v_rel) ** 2 / (self.pitch * self.blockage * np.abs(self.blade_normal[2])) * \
           (2 * self.c_f + 2 * np.pi * self.K * (self.delta - delta_ref) ** 2)


def calc_Mach_rel(self):
    """Calculate relative Mach number
    Author:  Anais Habermann
     Args:
        self
    Returns:
        Ma_rel         [-] relative Mach number
    """

    return np.linalg.norm(self.v_rel) / self.sos


def calc_K_mach(self):
    """Compute correction factor to account for compressibility, [2] equ. (9)
    Author:  Anais Habermann
     Args:
        self
    Returns:
        K_mach         [-] local force vector normal to the relative flow
    """

    if self.Ma_rel < 1:
        K_mach = min(1 / (np.sqrt(1 - self.Ma_rel ** 2)), 3)
    else:
        K_mach = min(4 / (np.sqrt(self.Ma_rel ** 2 - 1)), 3)

    return K_mach


def calc_Re_local(self):
    """Calculate local chordwise Reynolds number, [2] equ. (11)
    Author:  Anais Habermann
     Args:
        self
    Returns:
        Re_local         [-] local chordwise Reynolds number
    """

    return self.rho * self.x_chord * np.linalg.norm(self.v_rel) / self.mu


def calc_C_f(self):
    """Calculate skin friction contribution, [2] equ. (10)
    Author:  Anais Habermann
     Args:
        self
    Returns:
        C_f         [-] skin friction contribution
    """
    if self.Re_chord == 0:
        cf = 0.01
    else:
        cf = 0.0592 * self.Re_chord ** (-0.2)

    return cf


def calc_delta(self):
    """Calculate local deviation angle, [2] equ. (6)
    Author:  Anais Habermann
     Args:
        self
    Returns:
        delta      [rad] local deviation angle
    """

    return np.arcsin(np.dot(self.v_rel, self.blade_normal) / np.linalg.norm(self.v_rel))


def calc_rel_velocity(self):
    """Calculate blade relative velocity vector, [0]
    Author:  Anais Habermann
     Args:
        self
    Returns:
        v_rel      [m/s] relative velocity vector [v_rel_p, v_rel_n, v_rel_teta]
    """

    teta_unit = [0, 0, 1]  # unit vector for teta
    v_rel = np.add(self.v_abs, -np.dot((self.rot_speed * self.r), teta_unit))

    return v_rel


def calc_pitch(self):
    """Calculate local blade pitch, [3]
    Author:  Anais Habermann
     Args:
        self
    Returns:
        blade_pitch      [m]
    """

    return 2 * np.pi * self.r / self.number_blades
