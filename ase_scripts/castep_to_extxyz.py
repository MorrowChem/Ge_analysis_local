''' A script to convert a directory full of
.castep output into an extxyz format'''
from ase.io.castep import read_castep_castep, read_param
from ase.io.extxyz import write_extxyz
#from ase.neighborlist import neighbor_list
import numpy as np
import re
from sys import argv
import os

def extract_float(a):
    '''Extracts the last float in a string'''
    for t in a.split():
        try:
            E = float(t)
        except ValueError:
            pass
    return E

def get_castep_E(file):
    '''Gets the basis-set corrected total energy from .castep'''
    with open(file) as f:
        for a in f.readlines():
            if re.search(" Total energy", a):
                return extract_float(a)
        print('Warning: no energy found')
        return 1

def get_config_type(file, resc=False):
    'makes a config_type string by mapping from timestep in .cell file'
    if resc:
        t120 = 'liq_rescale_120'
    else:
        t120 = 'liq'
    timestep_map = {'20000.0'  : 'hiT_liq',
                    '80000.0'  : 'liq_rescale_80',
                    '120000.0' : t120,
                    '160000.0' : 'inter',
                    '180000.0' : 'hiT_amorph',
                    '240000.0' : 'amorph'}
    with open(file) as f:
        for a in f.readlines():
            if re.search("time step", a):
                return timestep_map[str(extract_float(a))]

def construct_xyz_entry(file, config_type=None):
    '''Constructs an ase.Atoms object with additional information required
    by gap_fit in extended xyz file'''
    c = read_castep_castep(file)[0]
    p = read_param(file[:-6] + 'param').todict()['param']
    try:
        c.get_total_energy()
    except:
        print('%s is bad' % file)
        return 1
    #cutoff = 5.5
    #nl = neighbor_list('i', c, cutoff, self_interaction=False)
    #nn = np.bincount(nl)
    #if nn.size == 0:
    #    nn = np.array([0])
    c.info['dft_energy'] = get_castep_E(file)
    #c.info['cutoff'] = cutoff
    c.info['dft_virial'] = -c.get_stress(voigt=False)*c.get_volume()
    c.info['castep_file'] = file
    c.info.update(p)
    if not config_type:
        cell_file = file[:-6] + '.cell'
        c.info['config_type'] = get_config_type(cell_file, resc=('liquid_rescale' in file))
    else:
        c.info['config_type'] = config_type
    c.set_array('dft_forces', c.get_forces())
    c.set_array('Z', c.get_atomic_numbers())
    return c

#castep_directory = argv[1]
#castep_directory = ['/data/chem-amais/hert5155/Ge/run_64_atoms/cells/',
#                    '/data/chem-amais/hert5155/Ge/run_64_atoms/cells_20_160_180/',
#                    '/data/chem-amais/hert5155/Ge/run_125_atoms/cells_20_120_160_180_240/',
#                    '/data/chem-amais/hert5155/Ge/run_216_atoms/cells_20_120_160_180_240/',
#                    '/data/chem-amais/hert5155/Ge/run_64_atoms/liquid_rescale/']
castep_directory = ['/data/chem-amais/hert5155/Ge/run_64_atoms/Si_DFT_64/']
extras = ['/data/chem-amais/hert5155/Ge/crystal_structures/hcp_opt.castep']
file_ranges = [[1,81], [1,121], [1, 41], [1, 41], [1, 81]] # choose train/val balance
#file_ranges = [[1,201]]
selection = [[2], [3], [4,5], [4,5], [1,2]] # choose which timesteps - based on modular arithmetic on run index
#types = [2, 3, 5, 5, 2] # number of config_types in each castep_directory, set to 1 to ignore selection and include all
types = [1 for i in castep_directory]
with open('./train.xyz', 'w') as fp:
    pass
with open('./validate.xyz', 'w') as fp:
    pass
tracking = [0, [], []]
#c = construct_xyz_entry('/data/chem-amais/hert5155/Ge/isolated.castep', config_type='isol')
c = construct_xyz_entry('/data/chem-amais/hert5155/Ge/Si_isol.castep', config_type='isol')
write_extxyz('./train.xyz', c, columns=['symbols', 'positions',
                                          'Z', 'n_neighb',
                                          'dft_forces'],
             write_info=True, append=False)
if c != 1:
    print('\nIsolated atom added, energy = {0:6.4f}'.format(c.info['dft_energy']))

for i in extras:
    c = construct_xyz_entry(i, config_type='cryst')
    if c != 1:
        write_extxyz('./train.xyz', c, columns=['symbols', 'positions',
                                                     'Z', 'n_neighb',
                                                     'dft_forces'],
                     write_info=True, append=True)
        tracking[0] += 1
print('\nextra configuration count: {}'.format(tracking[0]))

for ct, direc in enumerate(castep_directory):
    tracking[1].append(0); tracking[2].append(0)
    for i in os.listdir(direc):
        if re.search(r"\.castep", i) and 0 in [(int(i.split('.')[0]) - j) % types[ct] for j in selection[ct]]:
            c = construct_xyz_entry(direc + i)
            if int(i.split('.')[0]) < file_ranges[ct][1] and c != 1:
                write_extxyz('./train.xyz', c, columns=['symbols', 'positions',
                                                             'Z', 'n_neighb',
                                                             'dft_forces'],
                             write_info=True, append=True)
                tracking[1][-1] += 1
            elif c != 1:
                write_extxyz('./validate.xyz', c, columns=['symbols', 'positions',
                                                            'Z', 'n_neighb',
                                                            'dft_forces'],
                             write_info=True, append=True)
                tracking[2][-1] += 1
print('\nTraining configs (by directory):')
for i in range(len(castep_directory)):
    print('{0:40s} : {1}'.format(castep_directory[i], tracking[1][i]))
print('\nValidation configs (by directory):')
for i in range(len(castep_directory)):
    print('{0:40s} : {1}'.format(castep_directory[i], tracking[2][i]))
