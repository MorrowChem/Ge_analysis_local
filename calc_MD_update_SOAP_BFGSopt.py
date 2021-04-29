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
############### Run to run parameters #######################################
inplace=True
GAP=True; calc_args=''
local_variance=True


init_xyz = sys.argv[1]
'''extra_mtps = [MTP(sys.argv[2], mtp_command=r'/u/vld/hert5155/mlip-2/bin/mlp',
              potential_name=sys.argv[2],
              train='/u/vld/hert5155/Si_research/GAP_to_MTP/PQ1k_GAP18_GAP18labelled/PQ/PQ_GAP18labelled_train.cfg')]'''
extra_mtps = []

print('inplace: ',inplace,
      '\nGAP: ', GAP,
      '\nlv: ', local_variance,
      '\nxyz: ', init_xyz,
      '\nmtps: ', extra_mtps, flush=True)
############ should be able to set with arguments or input file #############

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


job_size = len(r.df.index)
print('job_size: {} configs'.format(job_size), flush=True)

if GAP:
    if local_variance:
        calc_args='local_gap_variance'

    GAP_18_pot = Potential(param_filename='/u/vld/hert5155/Si_research/GAP-18_ref/gp_iter6_sparse9k.xml',
                          calc_args=calc_args)

for i, ind in enumerate(r.df.index):
    
    if GAP:
    # calculate with GAP-18 if not already there
        if '{}_energy'.format('GAP18') in r.df['Configs'][ind].info.keys() \
           and (not local_variance or '{}_variance'.format('GAP18') in r.df['Configs'][ind].info.keys()):
            
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
fit_folder = '/u/vld/hert5155/Ge_analysis_local'
with open(join(fit_folder, 'xtals_for_SOAP_fits_abc.pickle'), 'rb') as f:
    fits_abc = pickle.load(f)
with open(join(fit_folder, 'xtals_for_SOAP_fits_e.pickle'), 'rb') as f:
    fits_e = pickle.load(f)
with open(join(fit_folder, 'xtals_for_SOAP.pickle'), 'rb') as f:
    ats = pickle.load(f)

comps = [i[0] for i in ats]
comp_labels = ['fcc', 'dia', 'hcp', 'bcc', 'sc', 'sh', 'bSn', 'Imma', 'Cmca']
comp_fix_labels = ['{}_fix_B'.format(i) for i in comp_labels]
rough_pressures = np.linspace(0, 20*1e4, len(r.df.index)) ## change this!
if 'f_PressAve' not in r.df.columns:
    r.df['f_PressAve'] = rough_pressures
if 'Press' not in r.df.columns:
    r.df['Press'] = rough_pressures

# calculate the energies and lattice parameters of the crystals
Ps = np.array(r.df['Press'].tolist())/(160.2176*1e4)
#Ps = np.linspace(0,20,len(r.df['Configs']))/160.2176
vols = []; es = []; abcs = []
for ct, i in enumerate(comps):
    abc = fits_abc[ct](Ps)
    v = np.abs(np.array([np.linalg.multi_dot((i[0], np.cross(i[1], i[2]))) for i in abc.T]))/len(i)
    e = fits_e[ct](Ps)
    vols.append(v); es.append(e); abcs.append(abc)

vols = np.array(vols)
es = np.array(es)
# calculate lowest-enthalpy crystal
Hs = es + Ps * vols
for i in range(len(Hs)):
    base_H = [min(i) for i in Hs.T]
    stable_xtal = [comp_labels[np.argmin(i)] for i in Hs.T]
print('Enthalpies and volumes of crystals calculated\nBeginning kernel evaluations', flush=True)

comps_copy = deepcopy(comps)
for i, ind in enumerate(r.df.index):

    for c, cval in enumerate(comps_copy):
        cval.set_cell(abcs[c].T[i], scale_atoms=True)
    try:
        kerns = kernel_compare(r.df['Configs'][ind],
                               comps_copy+comps, similarity=True, average=True)
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

if inplace:
    with open(os.path.join('{}.xyz'.format(lab)), 'w') as f:
        write_xyz(f, r.df['Configs'].tolist())
else:
    with open(os.path.join('{}_relabel_SOAP_B.xyz'.format(lab)), 'w') as f:
        write_xyz(f, r.df['Configs'].tolist())


for j, val in enumerate(r.df['Configs'][r.df.index[0]].info.keys()):
    r.df[val] = [struct.info[val] for struct in r.df['Configs']]

write_df = r.df.drop(columns='Configs')

if inplace:
    with open(os.path.join('{}.json'.format(lab)), 'w') as f:
        write_df.to_json(f)
else:
    with open(os.path.join('{}_relabel_SOAP_B.json'.format(lab)), 'w') as f:
        write_df.to_json(f)
