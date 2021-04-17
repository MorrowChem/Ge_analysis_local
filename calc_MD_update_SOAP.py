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

# temporary imports here
from calc_all_lib import *

GAP_18_pot = Potential(param_filename='/u/vld/hert5155/Si_research/GAP-18_ref/gp_iter6_sparse9k.xml',
                        calc_args='local_gap_variance')

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
r = MD_run(init_xyz, label=init_xyz[:-4], format='xyz')

'''extra_mtps = [MTP('u16_m1.5_fw0.1_PQ1k_G18_scratch.mtp', mtp_command=r'/u/vld/hert5155/mlip-2/bin/mlp',
              potential_name='u16_m1.5_fw0.1_PQ1k_scratch.mtp',
              train='/u/vld/hert5155/Si_research/GAP_to_MTP/PQ1k_GAP18_GAP18labelled/PQ/PQ_GAP18db_GAP18labelled_train.cfg')]'''

extra_mtps = []

job_size = len(r.df.index)
print('job_size: {} configs'.format(job_size), flush=True)

for i, ind in enumerate(r.df.index):

    # calculate with GAP-18 if not already there
    if '{}_energy'.format('GAP18') in r.df['Configs'][ind].info.keys():
        print('skipping GAP-calc of {}'.format(i))
    else:
        GAP_calc(r, ind, GAP_18_pot, local_variance=True)
    
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

# kernel comparison

a = 2.65
sh = Atoms(hexagonal.Hexagonal(symbol='Si', latticeconstant={'a':a, 'c':a*0.957}))
dia = bulk('Si', crystalstructure='diamond', a=5.43,  cubic=True)
fcc = bulk('Si', crystalstructure='fcc', a=3.89, cubic=True)
bSn_fact = bSn_factory()
a = 4.8
bSn = Atoms(bSn_fact(symbol='Si', latticeconstant={'a':a, 'c':0.525*a}))

comps = [sh, bSn, dia, fcc]
comp_labels = ['sh', 'bSn', 'dia', 'fcc']
comp_fix_labels = ['sh_fix', 'bSn_fix', 'dia_fix', 'fcc_fix']
rough_pressures = np.linspace(0, 20*1e4, len(r.df.index))
if 'f_PressAve' not in r.df.columns:
    r.df['f_PressAve'] = rough_pressures
if 'Press' not in r.df.columns:
    r.df['Press'] = rough_pressures

# calculate the energies and volumes of the crystals

vols = []; es = []
for i in comps:
    v, e = V_p_BM(r.df['f_PressAve']/1e4, i, GAP_18_pot)
    vols.append(v); es.append(e)
vols = np.array(vols)
es = np.array(es)

Hs = es + np.array(r.df['Press'].tolist())/160.2176 * vols
for i in range(len(Hs)):
    Hs[i] *= 1/len(comps[i])
    base_H = [min(i) for i in Hs.T]
fac = [vols[i]/comps[i].get_volume() for i in range(len(vols))] # unit-cell scaling factor
fac_fix = [i[0] for i in fac]
print('Enthalpies and volumes of crystals calculated', flush=True)

comps_copy = deepcopy(comps)
comps_fix = deepcopy(comps)

for i, val in enumerate(comps_fix):
    val.set_cell(val.get_cell()*fac_fix[i]**(1/3), scale_atoms=True)

for i, ind in enumerate(r.df.index):

    for c, cval in enumerate(comps_copy):
        cval.set_cell(comps[c].get_cell()*fac[c][i]**(1/3), scale_atoms=True)

    try:
        kerns = kernel_compare(r.df['Configs'][ind],
                                                  comps_copy+comps_fix, similarity=True, average=True)
    except:
        e = sys.exc_info()[0]
        print('Average kernel comparison of config {} failed for some reason\n{}'.format(ind, str(e)))
        if debug:
            traceback.print_exc()

    for j, val in enumerate(kerns):
        r.df['Configs'][ind].info[(comp_labels+comp_fix_labels)[j]] = val
        
    try:
        kerns = kernel_compare(r.df['Configs'][ind],
                                                  comps_copy, similarity=True, average=False)
    except:
        e = sys.exc_info()[0]
        print('Kernel comparison of config {} failed for some reason\n{}'.format(ind, str(e)))
        if debug:
            traceback.print_exc()

    for j, val in enumerate(kerns):
        r.df['Configs'][ind].arrays[comp_labels[j]] = val
    
    update_progress(i/job_size)


# coordination counting
if 'cn' not in r.df['Configs'][r.df.index[0]].info.keys():
    print('\nbeg. cn counting')
    
    for i, val in enumerate(r.df['Configs']):
        g_cn, i_cn = cn_count(val, r=2.7)
#        print(g_cn, len(i_cn), i)
        val.info['cn'] = g_cn
        if len(i_cn) == len(val):
            val.arrays['cn'] = i_cn
        else:
            print('Problem with cn counting for config {}'.format(i))


with open(os.path.join('{}_relabel_SOAP.xyz'.format(init_xyz[:-4])), 'w') as f:
    write_xyz(f, r.df['Configs'].tolist())
