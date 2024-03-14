"""
Prepare the OpenFOAM case before running the simulation as part of the automated HPFVM method.

Author:  A. Habermann
"""

import os

from misc_functions.air_properties.calculate_freestream import calc_freestream


class PrepareOFCase:

    def __init__(self, atmosphere, geo_parameters, l_fuse_tot, rotor_coeffs, stator_coeffs, casepath):
        self.stator_coeffs = stator_coeffs
        self.rotor_coeffs = rotor_coeffs
        self.l_fuse_tot = l_fuse_tot
        self.casepath = casepath
        self.gp = geo_parameters
        self.atmos = atmosphere

    def prepare_files(self):
        freestreampath = os.path.join(self.casepath, '0//include')
        if not os.path.exists(freestreampath):
            os.makedirs(freestreampath)
        k, omega, alphat, nut = calc_freestream(self.atmos, self.l_fuse_tot, 'yplus')
        with open(f'{freestreampath}//freestreamConditions', 'w') as f:
            f.write(f'U ({self.atmos.ext_props["u"][0]} 0.0 0.0);\n')
            f.write(f'p {self.atmos.pressure[0]};\n')
            f.write(f'T {self.atmos.temperature[0]};\n')
            f.write(f'k {k[0]};\n')
            f.write(f'omega {omega[0]};\n')
            f.write(f'gamma 1.4;\n')
            f.write(f'intensity 0.001;\n')
            f.write(f'mixingLength 1e-05;\n')
            f.write(f'nue_t {nut[0]};\n')

        with open(f'{self.casepath}//controlvars1', 'w') as c1:
            c1.write(f'start latestTime;\n')
            c1.write(f'starttime 0;\n')
            c1.write(f'endtime 4000;\n')
            c1.write(f'writeinterval 100;\n')
            c1.write(f'rho {self.atmos.ext_props["rho"][0]};\n')
            c1.write(f'p {self.atmos.pressure[0]};\n')
            c1.write(f'U {self.atmos.ext_props["u"][0]};\n')

        with open(f'{self.casepath}//controlvars2', 'w') as c2:
            c2.write(f'start latestTime;\n')
            c2.write(f'starttime 4000;\n')
            c2.write(f'endtime 4001;\n')
            c2.write(f'writeinterval 1;\n')
            c2.write(f'rho {self.atmos.ext_props["rho"][0]};\n')
            c2.write(f'p {self.atmos.pressure[0]};\n')
            c2.write(f'U {self.atmos.ext_props["u"][0]};\n')

        with open(f'{self.casepath}//controlvars3', 'w') as c3:
            c3.write(f'start latestTime;\n')
            c3.write(f'starttime 4001;\n')
            c3.write(f'endtime 6000;\n')
            c3.write(f'writeinterval 100;\n')
            c3.write(f'rho {self.atmos.ext_props["rho"][0]};\n')
            c3.write(f'p {self.atmos.pressure[0]};\n')
            c3.write(f'U {self.atmos.ext_props["u"][0]};\n')

        with open(f'{self.casepath}//controlvars4', 'w') as c4:
            c4.write(f'start latestTime;\n')
            c4.write(f'starttime 6000;\n')
            c4.write(f'endtime 12000;\n')
            c4.write(f'writeinterval 100;\n')
            c4.write(f'rho {self.atmos.ext_props["rho"][0]};\n')
            c4.write(f'p {self.atmos.pressure[0]};\n')
            c4.write(f'U {self.atmos.ext_props["u"][0]};\n')

        # adapt fvoptions file with prepare_body_force_model parameters
        adapt_fvoptions(self)


def adapt_fvoptions(self):
    fvoptionsPath = os.path.join(self.casepath, "fvOptions")
    with open(fvoptionsPath, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'define constants' in line:
                lines[i + 1] = f"const scalar OMEGAROT = {self.gp['omega_rot']};\n"
                lines[i + 2] = f"const scalar sos = {self.atmos.ext_props['sos'][0]};\n"
                lines[i + 3] = f"std::vector<double> rotor_le_coeffs_vec{{{self.rotor_coeffs[0]}," \
                               f"{self.rotor_coeffs[1]},{self.rotor_coeffs[2]},{self.rotor_coeffs[3]}}};\n"
                lines[i + 4] = f"std::vector<double> stator_le_coeffs_vec{{{self.stator_coeffs[0]}," \
                               f"{self.stator_coeffs[1]},{self.stator_coeffs[2]},{self.stator_coeffs[3]}}};\n"
            else:
                pass
    with open(fvoptionsPath, 'w') as f:
        for line in lines:
            f.write(line)
