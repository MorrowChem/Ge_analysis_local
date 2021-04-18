import traceback
from ase.io.trajectory import Trajectory
from ase.io import read
from ase.units import kJ
from ase.eos import EquationOfState, birchmurnaghan
from scipy.optimize import root
from copy import deepcopy

from ase.io.extxyz import read_xyz
import numpy as np
from ase.atoms import Atoms
from ase.io.cfg import read_cfg
from mtp import *
import os
from matplotlib import pyplot as plt
from quippy.potential import Potential
import pickle
from Ge_analysis import *
from Ge_calculation import *
import matplotlib.pyplot as plt
from matscipy.rings import ring_statistics
from datetime import datetime
import pymatgen.ext.matproj as mp
import pymatgen.io.ase as pase
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from numpy.random import normal
from ase.io.extxyz import write_xyz
from ase.io.lammpsdata import write_lammps_data
import time, sys

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 30 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:5.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def BM_p(v, v0, b0, bp):
    return 3*b0/2 * ( (v0/v)**(7/3) - (v0/v)**(5/3) ) * ( 1 + 3/4*(bp-4)*((v0/v)**(2/3) - 1) )


def V_p_BM(P, xtal, calc, traj_name='tmp', n=20):
    xtal.set_calculator(None)
    at = deepcopy(xtal)
    at.set_calculator(calc)
    cell = at.get_cell()
    traj = Trajectory('{}.traj'.format(traj_name), 'w')
    
    for x in np.linspace(0.95, 1.05, n):
        at.set_cell(cell * x, scale_atoms=True)
        at.get_potential_energy()
        traj.write(at)
        
    configs = read('{}.traj@0:{}'.format(traj_name, n))  # read configurations
    # Extract volumes and energies:
    volumes = [i.get_volume() for i in configs]
    energies = [i.get_potential_energy() for i in configs]
    eos = EquationOfState(volumes, energies, 'birchmurnaghan')
    v0, e0, B = eos.fit()
    BP = eos.eos_parameters[2]
#     print(B / kJ * 1.0e24, 'GPa')
#     eos.plot()
    
    if not isinstance(P, np.ndarray):
        try:
            P = np.array(P)
        except:
            raise TypeError('P must be a list or numpy array')
            
    p = P * kJ / 1.0e24 # convert to atomic units
    prev_v = xtal.get_volume()
#     print('initial volume guess: {:5.2f}'.format(prev_v))
    bm_volumes = []
    
    for i in p:
        sol = root(lambda vol: BM_p(vol, v0, B, BP) - i, prev_v)
        prev_v = sol.x[0]
#         print(sol.x)
        bm_volumes.append(sol.x[0])
    
    bm_volumes = np.array(bm_volumes)
    
    bm_energies = birchmurnaghan(bm_volumes, e0, B, BP, v0)
        
    return bm_volumes, bm_energies




def GAP_calc(r, ind, pot, local_variance=True):
    try:
        if local_variance:
            pot.calculate(r.df['Configs'][ind], properties=['energy', 'energies', 'forces', 'stress'],
                                args_str='local_gap_variance', copy_all_results=True)
        else:
            pot.calculate(r.df['Configs'][ind], properties=['energy', 'energies', 'forces', 'stress'],
                                copy_all_results=True)
        r.df['Configs'][ind].info['{}_energy'.format('GAP18')] = pot.results['energy']
        r.df['Configs'][ind].info['{}_virial'.format('GAP18')] = pot.results['stress']
        r.df['Configs'][ind].arrays['{}_force'.format('GAP18')] = pot.results['forces']
        r.df['Configs'][ind].arrays['{}_energies'.format('GAP18')] = pot.results['energies']

        if local_variance:
            r.df['Configs'][ind].arrays['{}_variance'.format('GAP18')] = pot.extra_results['atoms']['local_gap_variance']
    except:
        e = sys.exc_info()[0]
        print('calculation of config {} failed for some reason\n{}\n'.format(ind, e))
        traceback.print_exc()

    r.df['Configs'][ind].set_calculator(None)
    

def MTP_calc(r, ind, pot, grade=True, debug=False, method='mlp'):

    try:
        if grade:
            if method=='lammps':
                raise AttributeError('grade not implemented for lammps')
            pot.calculate(r.df['Configs'][ind], properties=['energy', 'forces', 'stress', 'grade'], timeout=3600)
        else:
            pot.calculate(r.df['Configs'][ind], properties=['energy', 'forces', 'stress'],
            timeout=36000, method=method)
    
        r.df['Configs'][ind].info['{}_energy'.format(pot.name)] = pot.results['energy']
        r.df['Configs'][ind].info['{}_virial'.format(pot.name)] = pot.results['stress']
        r.df['Configs'][ind].arrays['{}_force'.format(pot.name)] = pot.results['forces']

        if grade:
            r.df['Configs'][ind].info['{}_grade'.format(pot.name)] = pot.results['MV_grade']

    except:
        e = sys.exc_info()[0]
        print('calculation of config {} with {} failed for some reason\n{}'.format(ind, pot.name, str(e)))
        if debug:
            traceback.print_exc()
