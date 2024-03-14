"""
Run an OpenFOAM simulation as part of the automated HPFVM method.

Author:  A. Habermann
"""

import os
import subprocess


def run_sim(casepath, sim_idx):
    os.chdir(casepath)
    # run simulation
    logfile = open(f"{casepath}//log", 'w')
    subprocess.call(["./Allrun"], stdout=logfile, stderr=subprocess.STDOUT)
    with open(f"{casepath}//log", 'r+') as f:
        lines = f.readlines()

        if not (lines[-1].split() == ['Finalising', 'parallel', 'run'] or lines[-2].split() ==
                ['Finalising', 'parallel', 'run'] or lines[-3].split() == ['Finalising', 'parallel', 'run']):
            raise Exception(f'No FV solution for index {sim_idx}')

    # clean up folder
    os.system("./CleanSim")
