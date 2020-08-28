from Ge_calculation import *
from sys import argv
from os import system, getcwd
import numpy as np
from copy import deepcopy

def gap_fit_create(*args, no_soap=False, gap_args=None):

    if not gap_args:
        gap_args = {'atoms_filename': 'train_216_125_64_v.xyz',
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

    if no_soap:
        gap_args['gap'].pop('soap')
    gap_str = gap_args_to_str(gap_args)

    print(gap_args)
    return gap_str, gap_args

def gap_args_to_str(gap_args):
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

    return gap_str

def write_sub(gap_str, sub_file):
    with open(sub_file, 'w') as f:
        f.write('#!/bin/bash\n' +
                '#SBATCH --job-name=gap_fit\n' +
                '#SBATCH --time=24:00:00\n' +
                '#SBATCH --nodes=1\n' +
                '#SBATCH --ntasks-per-node=16\n' +
                '#SBATCH --account=chem-amais\n' +
                #'#SBATCH --partition=htc\n\n' +
                'module load gcc/6.4.0\n' +
                'gap_fit {}'.format(gap_str))

def train_2b(gap_args, train_file, gap_file='2b'):
    '''trains a 2b-only potential and returns the energy rmse to set the
    delta for a SOAP fit'''

    if 'soap' in gap_args['gap']:
        print('SOAP removed\n')
        soap = gap_args['gap']['soap']
        gap_args['gap'].pop('soap')

    gap_str = gap_args_to_str(gap_args)

    system('gap_fit {}'.format(gap_str))

    db_2b = GAP(str(train_file), pot='{}'.format(gap_args['gap_file']))
    db_2b.calc(val=False)
    db_2b.analyse()

    with open('{}.out'.format(gap_args['gap_file'].split('.')[0]), 'w') as f:
        f.write('Energy RMSE / eV:'.ljust(40, ' ') + '{: 5.6f}\n'.format(
            np.average([i['rmse'] for i in db_2b.data_dict['E_rmse_t']])))
        f.write('Forces RMSE / eVA^-1:'.ljust(40, ' ') + '{: 5.6f}\n'.format(
            np.average([i['rmse'] for i in db_2b.data_dict['F_rmse_t']])))
        f.write('Configs:'.ljust(40, ' ') +
                ('{:>11s} '*len(db_2b.config_labels)+'\n').format(
                    *db_2b.config_labels))
        f.write('Energy RMSE / eV by config:'.ljust(40, ' ') +
                ('{: 5.6f} '*len(db_2b.T_configs)+'\n').format(
            *[i['rmse'] for i in db_2b.data_dict['E_rmse_t']]))
        f.write('Forces RMSE / eVA^-1 by config:'.ljust(40, ' ') +
                ('{: 5.6f} '*len(db_2b.T_configs)+'\n').format(
                    *[i['rmse'] for i in db_2b.data_dict['F_rmse_t']]))

    delta = np.average([i['rmse'] for i in db_2b.data_dict['E_rmse_t']])
    gap_args['gap']['soap'] = soap
    return delta

#system('sbatch gap.sh')
