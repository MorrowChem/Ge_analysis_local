from Ge_calculation import *
from sys import argv
from os import system, getcwd
import numpy as np

def gap_fit_create(*args):
    
    gap_args = {'at_file': 'train_216_125_64_v.xyz',
                'gap': {'distance_Nb': {'order': '2',
                        'cutoff': '5.0',
                        'covariance_type': 'ARD_SE',
                        'theta_uniform': '2.0',
                        'n_sparse': '15',
                        'delta': '2.0',
                        'sparse_method': 'uniform',
                        'compact_clusters': 'T'},
                        'soap': {'l_max': '6',
                         'n_max': '12',
                         'atom_sigma': '0.5',
                         'zeta': '4',
                         'cutoff': '5.0',
                         'cutoff_transition_width': '1.0',
                         'central_weight': '1.0',
                         'n_sparse': '5000',
                         'delta': '0.155',
                         'f0': '0.0',
                         'covariance_type': 'dot_product',
                         'sparse_method': 'CUR_POINTS'}},
                'energy_parameter_name': 'dft_energy',
                'force_parameter_name': 'dft_forces',
                'virial_parameter_name': 'dft_virial',
                'sparse_jitter': '1.0e-8',
                'default_sigma': '{0.002 0.2 0.2 0.0}',
                'do_copy_at_file':'F',
                'sparse_separate_file': 'T',
                'gap_file': getcwd()+'.xml'}

    '''if len(argv) > 1:
        for i in argv[1:]:
            if 'gap=' in i:
                subarg = i[5:-1].split(':')
                subarg = [j.split() for j in subarg]
                for j in subarg:
                    for k in j[1:]:
                        gap_args['gap'][j[0]][k.split('=')[0]] = k.split('=')[1]
                continue
            gap_args[i.split('=')[0]] = i.split('=')[1]
    #print(gap_args)'''
    print(*args)
    for i in args:
        if 'gap=' in i:
            subarg = i[5:-1].split(':')
            subarg = [j.split() for j in subarg]
            for j in subarg:
                for k in j[1:]:
                    gap_args['gap'][j[0]][k.split('=')[0]] = k.split('=')[1]
            continue
        gap_args[i.split('=')[0]] = i.split('=')[1]


    gap_str = ''
    for i in gap_args.keys():
        if i == 'gap':
            temp_str = 'gap={'
            for k in gap_args[i].keys():
                temp_str += '{} '.format(k)
                for l in gap_args[i][k].keys():
                    temp_str += "{0}={1} ".format(l, gap_args[i][k][l])
                temp_str += ": "
            temp_str = temp_str[:-3]
            temp_str += "} "
            gap_str += temp_str
            continue
        gap_str += "{0}={1} ".format(i, gap_args[i])
    print(gap_str)

    return(gap_str)

def write_sub(gap_str, sub_file):
    with open(sub_file, 'w') as f:
        f.write('#!/bin/bash\n' +
                '#SBATCH --job-name=gap_fit\n' +
                '#SBATCH --time=24:00:00\n' +
                '#SBATCH --nodes=1\n' +
                '#SBATCH --ntasks-per-node=16\n' +
                '#SBATCH --account=chem-amais\n' +
                #'#SBATCH --partition=htc\n\n' +
                'module load gcc/7.3.0\n' +
                '{}'.format(gap_str))

#system('sbatch gap.sh')
