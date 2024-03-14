"""Validation of panel method using a number of validation cases. Most validation studies (including references to
original validation data) can be found in

    [1] Santa Cruz Mendoza, C.E.R.: Numerical Prediction of Turbulent Boundary Layer Characteristics to Feed
        the Design of Propulsive Fuselages. Master’s thesis. University of Stuttgart. 2020.
    [2] Romanow, N.: Extension of a Numerical Model for the Prediction of Turbulent Boundary Layer Characteristics
        on a Propulsive Fuselage Aircraft. Master’s thesis. Technical University Munich. 2021.

Author:  A. Habermann
"""

from panel.main_panel_method import Main

# Potential flow validation
Spheroid_6_1 = Main(['Spheroid_6_1', 'Spheroid_6_1'], [100], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0],
                    visc=0, calc_opt=2, alt=1000,
                    Mach=0.0365, discret=1, weight=0, interact=0, turb=0, trans=0.05, compr=0, case=1)
Spheroid_6_1.run()

Ellipsoid_8_1 = Main(['Ellipsoid_8_1', 'Ellipsoid_8_1'], [90], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0],
                     visc=0, calc_opt=2, alt=1000,
                     Mach=0.0365, discret=1, weight=0, interact=0, turb=0, trans=0.05, compr=0, case=2)
Ellipsoid_8_1.run()

LewisBody = Main(['LewisBody', 'LewisBody'], [90], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0], visc=0,
                 calc_opt=2, alt=10100,
                 Mach=0.75, discret=1, weight=0, interact=0, turb=0, trans=0.05, compr=0, case=3)
LewisBody.run()

NACA0012 = Main(['NACA0012_4', 'NACA0012'], [40, 40], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], visc=0,
                calc_opt=2, alt=0,
                Mach=0.2, discret=1, weight=0, interact=0, turb=0, trans=1, compr=0, case=0, sing_type=4)
NACA0012.run()

NACA2412 = Main(['NACA2412', 'NACA2412'], [50, 50], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], visc=0,
                calc_opt=2, alt=0,
                Mach=0.2, discret=1, weight=0, interact=0, turb=0, trans=0.05, compr=0, case=0, sing_type=4)
NACA2412.run()

# Turbulence model validation
Akron_Patel = Main(['Akron_Patel', 'Akron_Patel'], [90], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], visc=1,
                   calc_opt=2, alt=2000,
                   Mach=0.15, discret=1, weight=0, interact=1, turb=0, trans=0.06, compr=0, case=4)
Akron_Patel.run()

Akron_Green = Main(['Akron_Green', 'Akron_Green'], [90], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], visc=0,
                   calc_opt=2, alt=2000,
                   Mach=0.15, discret=1, weight=0, interact=1, turb=1, trans=0.06, compr=0, case=4)
Akron_Green.run()

F57_Patel = Main(['F57_Patel', 'F57_Patel'], [90], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], visc=0,
                 calc_opt=2, alt=500,
                 Mach=0.045, discret=1, weight=0, interact=1, turb=0, trans=0.475, compr=0, case=6)
F57_Patel.run()

F57_Green = Main(['F57_Green', 'F57_Green'], [90], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], visc=0,
                 calc_opt=2, alt=500,
                 Mach=0.045, discret=1, weight=0, interact=1, turb=1, trans=0.475, compr=0, case=6)
F57_Green.run()

ModSpheroid_6_1_Patel = Main(['ModSpheroid_6_1_Patel', 'ModSpheroid_6_1_Patel'], [90],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], calc_opt=2, visc=1, alt=1000,
                             Mach=0.3, discret=1, weight=0, interact=1, turb=0, trans=0.05, compr=0, case=5)
ModSpheroid_6_1_Patel.run()

ModSpheroid_6_1_Green = Main(['ModSpheroid_6_1_Green', 'ModSpheroid_6_1_Green'], [90],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], calc_opt=2, visc=0, alt=1000,
                             Mach=0.0365, discret=1, weight=0, interact=1, turb=1, trans=0.05, compr=0, case=5)
ModSpheroid_6_1_Green.run()

ESDU = Main(['ESDU', 'ESDU'], [80], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], calc_opt=2, visc=0, alt=500,
            Mach=0.045, discret=1, weight=0, interact=1, turb=0, trans=0.475, compr=0, case=7)
ESDU.run()

# Compressible flow validation
WaistedBody_Patel = Main(['WaistedBody_Patel', 'WaistedBody_Patel'], [90],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0], calc_opt=2, visc=1, alt=7680,
                         Mach=0.597, discret=1, weight=0, interact=2, turb=0, trans=0.1, compr=1, case=8)
WaistedBody_Patel.run()

# Compressible flow validation
WaistedBody_Patel = Main(['WaistedBody_Patel', 'WaistedBody_Patel'], [90],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0], calc_opt=2, visc=0, alt=7680,
                         Mach=0.597, discret=1, weight=0, interact=1, turb=0, trans=0.05, compr=2, case=8)
WaistedBody_Patel.run()
# Compressible flow validation
WaistedBody_Patel = Main(['WaistedBody_Patel', 'WaistedBody_Patel'], [90],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0], calc_opt=2, visc=0, alt=7680,
                         Mach=0.597, discret=1, weight=0, interact=1, turb=0, trans=0.05, compr=4, case=8)
WaistedBody_Patel.run()
WaistedBody_Green = Main(['WaistedBody_Green', 'WaistedBody_Green'], [90],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], calc_opt=2, visc=1, alt=7680,
                         Mach=0.597, discret=1, weight=0, interact=1, turb=1, trans=0.05, compr=1, case=8)
WaistedBody_Green.run()

NASAfuselage_Patel = Main(['NASAfuselage_Patel', 'NASAfuselage_Patel'], [90],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], calc_opt=2, visc=0, alt=10100,
                          Mach=0.75, discret=1, weight=0, interact=1, turb=0, trans=0.05, compr=1, case=9)
NASAfuselage_Patel.run()

NASAfuselage_Green = Main(['NASAfuselage_Green', 'NASAfuselage_Green'], [90],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0], calc_opt=2, visc=0, alt=10100,
                          Mach=0.75, discret=1, weight=0, interact=1, turb=1, trans=0.05, compr=1, case=9)
NASAfuselage_Green.run()
