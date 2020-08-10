from Ge_calculation import *
from sys import argv
from os import system, getcwd
import numpy as np

# Train the 2b part potential
gap_args = {'at_file': '../train_216_125_64_v.xyz',
            'gap': {'distance_Nb': {'order': '2',
                                    'cutoff': '5.0',
                                    'covariance_type': 'ARD_SE',
                                    'theta_uniform': '2.0',
                                    'n_sparse': '15',
                                    'delta': '2.0',
                                    'sparse_method': 'uniform',
                                    'compact_clusters': 'T'}},
            'energy_parameter_name': 'dft_energy',
            'force_parameter_name': 'dft_forces',
            'virial_parameter_name': 'dft_virial',
            'sparse_jitter': '1.0e-8',
            'default_sigma': '{0.002 0.2 0.2 0.0}',
            'do_copy_at_file':'F',
            'sparse_separate_file': 'T',
            'gap_file': getcwd()+'.xml'}

if len(argv) > 1:
    for i in argv[1:]:
        if 'gap=' in i:
            subarg = i[5:-1].split(':')
            subarg = [j.split() for j in subarg]
            for j in subarg:
                for k in j[1:]:
                    gap_args['gap'][j[0]][k.split('=')[0]] = k.split('=')[1]
            continuei
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

system('gap_fit {}'.format(gap_str))

db_2b = GAP('../../train_216_125_64_v.xyz', pot=gap_args['gap_file'])
db_2b.calc(val=False)
db_2b.analyse()
with open('2b_analysis.out', 'w') as f:
    f.write('Energy RMSE error / eV:\n{}\n'.format(
        np.average([i['rmse'] for i in db_2b.data_dict['E_rmse_t']])))
