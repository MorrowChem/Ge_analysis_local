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
from quippy.potential import Potential
from Ge_analysis import *
from Ge_calculation import *
from matscipy.rings import ring_statistics
from datetime import datetime
from ase.io.extxyz import write_xyz
from ase.io.lammpsdata import write_lammps_data
import time, sys
import pickle
from os.path import join

# temporary imports here
from calc_all_lib import *


### IDEA: calculate extra properties and add them to existing xyz MD_run

### Properties to have here:
# rings stats
# gap_local_variance and gradient
# local Enthalpy definition (and excess)
# mtps (with lammps calc?)
# calculate/use elasic constants instead of isotropic contraction (current definition of H_xs)
# similarities with different descriptors

init_xyz = sys.argv[1]
#init_xyz = 'min1.5_u16_fw0.1_sw0.001_1k.xyz'
if init_xyz[-4:] == '.xyz':
    lab = init_xyz[:-4]
    r = MD_run(init_xyz, label=lab, format='xyz')
else:
    r = MD_run(init_xyz, label=init_xyz, format='lammps')
    lab = init_xyz
    for i, ind in enumerate(r.df.index): # also write all MD_r data to xyz (need function for back-conversion)
        r.df['Configs'][ind].info['timestep'] = ind
        for j, val in enumerate(r.df.iloc[i][1:]):
            r.df['Configs'][ind].info[r.df.columns[j+1]] = val
'''
extra_mtps = [MTP('u16_m1.5_fw0.1_PQ1k_scratch.mtp', mtp_command=r'/u/vld/hert5155/mlip-2/bin/mlp',
              potential_name='u16_m1.5_fw0.1_PQ1k_scratch.mtp',
              train='/u/vld/hert5155/Si_research/GAP_to_MTP/PQ1k_GAP18_GAP18labelled/PQ/PQ_GAP18labelled_train.cfg')]
'''
extra_mtps = []

job_size = len(r.df.index)
print('job_size: {} configs'.format(job_size), flush=True)

GAP_18_pot = Potential(param_filename='/u/vld/hert5155/Si_research/GAP-18_ref/gp_iter6_sparse9k.xml')
for i, ind in enumerate(r.df.index):

    # calculate with GAP-18 if not already there
    if '{}_energy'.format('GAP18') in r.df['Configs'][ind].info.keys():
        print('skipping GAP-calc of {}'.format(i))
    else:
        GAP_calc(r, ind, GAP_18_pot, local_variance=False)
    
    update_progress(i/job_size)
    # calculate with extra MTPs if not already there
    for j, val in enumerate(extra_mtps):
        
        if not hasattr(val, 'name'):
            val.name = i
        
        if '{}_energy'.format(val.name) in r.df['Configs'][ind].info.keys():
            print('skipping MTP_{}-calc of {}'.format(j,i))
        else:
            MTP_calc(r, ind, val, grade=False, debug=True, method='lammps')
            update_progress(i/job_size)

# bulk grade calculation
if False:
    print('Calculating bulk MV grades')
    for j, val in enumerate(extra_mtps):
        
        grades = val.calc_grade_bulk(r.df['Configs'].tolist(), timeout=36000)
        
        for i, conf in enumerate(r.df['Configs']):
            conf.info['{}_grade'.format(val.name)] = grades[i]



with open(os.path.join('{}_label_G18.xyz'.format(lab)), 'w') as f:
    write_xyz(f, r.df['Configs'].tolist())
