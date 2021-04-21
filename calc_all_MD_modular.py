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


def kernel_compare(cfgs, comp,
                         desc=None,
                         zeta=4, similarity=False, average=True):
    '''calculates the average/std dev similarity kernel between a set of
    configs and a reference (or set of references).
    '''
    if desc is None:
        desc_str = 'soap l_max=6 n_max=12 \
                   atom_sigma=0.5 cutoff=5.0 \
                   cutoff_transition_width=1.0 central_weight=1.0'
        if average:
            desc_str += ' average=T'
        else:
            desc_str += ' average=F'
    else:
        desc_str = desc
    desc = Descriptor(desc_str)
    
    if not isinstance(comp, list):
        comp = [comp]

    descs = np.array(desc.calc_descriptor(cfgs))
    comp_desc = np.array([i[0] for i in desc.calc_descriptor(comp)]) # currently only looks at first atom of xtal
#     print(descs.shape, comp_desc.shape)

    if similarity:
        k = np.einsum('ij,kj', descs, comp_desc)**zeta
    else:
        k = np.array(2 - 2*np.einsum('ij,j', descs, comp_desc)**zeta)


    return k.T


def GAP_18_calc(r, ind, local_variance=True):
    try:
        if local_variance:
            GAP_18_pot.calculate(r.df['Configs'][ind], properties=['energy', 'energies', 'forces', 'stress'],
                                args_str='local_gap_variance', copy_all_results=True)
        else:
            GAP_18_pot.calculate(r.df['Configs'][ind], properties=['energy', 'energies', 'forces', 'stress'],
                                copy_all_results=True)
        r.df['Configs'][ind].info['{}_energy'.format('GAP18')] = GAP_18_pot.results['energy']
        r.df['Configs'][ind].info['{}_virial'.format('GAP18')] = GAP_18_pot.results['stress']
        r.df['Configs'][ind].arrays['{}_force'.format('GAP18')] = GAP_18_pot.results['forces']
        r.df['Configs'][ind].arrays['{}_energies'.format('GAP18')] = GAP_18_pot.results['energies']

        if local_variance:
            r.df['Configs'][ind].arrays['{}_variance'.format('GAP18')] = GAP_18_pot.extra_results['atoms']['local_gap_variance']
    except:
        e = sys.exc_info()[0]
        print('calculation of config {} failed for some reason\n{}\n'.format(i, e))

    r.df['Configs'][ind].set_calculator(None)
    

def MTP_calc(r, ind, pot, grade=True, debug=False):

    try:
        if grade:
            pot.calculate(r.df['Configs'][ind], properties=['energy', 'forces', 'stress', 'grade'], timeout=3600)
        else:
            pot.calculate(r.df['Configs'][ind], properties=['energy', 'forces', 'stress'], timeout=36000)
    
        r.df['Configs'][ind].info['{}_energy'.format(val.name)] = pot.results['energy']
        r.df['Configs'][ind].info['{}_virial'.format(val.name)] = pot.results['stress']
        r.df['Configs'][ind].arrays['{}_force'.format(val.name)] = pot.results['forces']

        if grade:
            r.df['Configs'][ind].info['{}_grade'.format(val.name)] = pot.results['MV_grade']

    except:
        e = sys.exc_info()[0]
        print('calculation of config {} with {} failed for some reason\n{}'.format(i, val.name, str(e)))
        if debug:
            traceback.print_exc()


# set up the crystal structures against which to compare (lattice parameters need to be within +-5% of equilibrium value.
# TODO: improve the BM procedure to iterate to better lattice parameters automatically
start = time.time()

a = 2.65
sh = Atoms(hexagonal.Hexagonal(symbol='Si', latticeconstant={'a':a, 'c':a*0.957}))
dia = bulk('Si', crystalstructure='diamond', a=5.43,  cubic=True)
fcc = bulk('Si', crystalstructure='fcc', a=3.89, cubic=True)
bSn_fact = bSn_factory()
a = 4.8
bSn = Atoms(bSn_fact(symbol='Si', latticeconstant={'a':a, 'c':0.525*a}))

comps = [sh, bSn, dia, fcc]
comp_labels = ['sh', 'bSn', 'dia', 'fcc']

#set up the MDs to be analysed and calculators to calculate
print(sys.argv)
print('MD_run of {}, extra pot is {}'.format(sys.argv[1], sys.argv[2]))
if sys.argv[1][-4:] == '.xyz':
    form = 'xyz'
else:
    form = 'lammps'

run = MD_run(sys.argv[1], label=sys.argv[2], format=form)
#run.pot = MTP(sys.argv[3], potential_name=sys.argv[3].split('/')[-1], mtp_command=r'/u/vld/hert5155/mlip-2/bin/mlp',
#              train='/u/vld/hert5155/Si_research/GAP_to_MTP/GAP_18_db_GAPlabelled.cfg')
runs = [run]

#GAP_18_pot = Potential(param_filename='/u/vld/hert5155/Si_research/GAP-18_ref/gp_iter6_sparse9k.xml',
#                       calc_args='local_gap_variance')
GAP_18_pot = Potential(param_filename='/u/vld/hert5155/Si_research/GAP-18_ref/gp_iter6_sparse9k.xml'
                       )


# main procedures from here

for r in runs:

    if sys.argv[1][-4:] == '.xyz':
        out_path = '/'.join(sys.argv[1].split('/')[:-1])
    else:
        out_path = sys.argv[1]

    if form == 'lammps':
        for i, ind in enumerate(r.df.index): # also write all MD_r data to xyz (need function for back-conversion)
            r.df['Configs'][ind].info['timestep'] = ind
            for j, val in enumerate(r.df.iloc[i][1:]):
                r.df['Configs'][ind].info[r.df.columns[j+1]] = val

        with open(os.path.join(sys.argv[1],'tmp.xyz'), 'w') as f:
            write_xyz(f, r.df['Configs'].tolist())


    vols = []; es = []

'''# calculate the energies and volumes of the crystals

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
    print('Enthalpies and volumes of crystals calculated', flush=True)'''

# expensive step: calculate all the energies and kernels (memory intensive)
# calculate and add SOAP kernel information to 
# TODO: add squared exponential kernel to kernel_compare 
# TODO: make calculation of each attribute a function that can be easily turned on and off
    job_size = len(r.df.index)
    print('job_size: {} configs'.format(job_size), flush=True)
    #extra_pots = [r.pot]
    extra_pots = []

    for i, ind in enumerate(r.df.index):

        # calculate with GAP-18 if not already there
        if '{}_energy'.format('GAP18') in r.df['Configs'][ind].info.keys():
            print('skipping GAP-calc of {}'.format(i))
        else:
            GAP_18_calc(r, ind, local_variance=True)
        
        # calculate with extra MTPs if not already there
        for j, val in enumerate(extra_pots):
            
            if not hasattr(val, 'name'):
                val.name = i
            
            if '{}_energy'.format(val.name) in r.df['Configs'][ind].info.keys():
                print('skipping MTP_{}-calc of {}'.format(j,i))
            else:
                MTP_calc(r, ind, val, grade=False, debug=True)
            
        
        update_progress(i/job_size)
        
        # Excess enthalpies
        r.df['Configs'][ind].info['H_xs'] = (r.df['Configs'][ind].info['GAP18_energy'] + \
        r.df['Press'][ind]/160.2176 * r.df['Configs'][ind].get_volume())/len(r.df['Configs'][ind]) - base_H[i]

        r.df['Configs'][ind].arrays['H_xs'] = r.df['Configs'][ind].arrays['GAP18_energies'] + \
        r.df['Press'][ind]/160.2176 * r.df['Configs'][ind].get_volume()/len(r.df['Configs'][ind]) - base_H[i]
        

        # similarities
        comps_copy = deepcopy(comps)
        for c, cval in enumerate(comps_copy):
            cval.set_cell(cval.get_cell()*fac[c][i]**(1/3), scale_atoms=True)
        
        try:
            kerns = kernel_compare(r.df['Configs'][ind],
                                                      comps_copy, similarity=True, average=True)
        except:
            print('Average kernel comparison of {} failed'.format(ind))

        for j, val in enumerate(kerns):
            r.df['Configs'][ind].info[comp_labels[j]] = val
            
        try:
            kerns = kernel_compare(r.df['Configs'][ind],
                                                      comps_copy, similarity=True, average=False)
        except:
            print('Average kernel comparison of {} failed'.format(ind))

        for j, val in enumerate(kerns):
            r.df['Configs'][ind].arrays[comp_labels[j]] = val
            
        update_progress((2*i+1)/(job_size*2))

        with open(os.path.join(out_path, 'tmp.xyz'), 'w') as f:
            write_xyz(f, r.df['Configs'].tolist())

    #for j, val in enumerate(comp_labels):
    #    r.df[val] = [struct.info[val] for struct in r.df['Configs']]
# add the H_exs etc. to df as well (only important for notebook version of this analysis script)

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
    
# bulk grade calculation
    print('Calculating bulk MV grades')
    for j, val in enumerate(extra_pots):
        
        if not hasattr(val, 'name'):
            val.name = i
        
        grades = val.calc_grade_bulk(r.df['Configs'].tolist(), timeout=36000)
        
        for i, conf in enumerate(r.df['Configs']):
            conf.info['{}_grade'.format(val.name)] = grades[i]

#all-important: write the output xyz
    print('writing {}.xyz'.format(run.label))
    with open(os.path.join(out_path, '{}.xyz'.format(run.label)), 'w') as f:
        write_xyz(f, r.df['Configs'].tolist())

end = time.time()
print('Done in {:3.2f} hrs'.format(float((end - start)/3600)))
