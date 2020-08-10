from Ge_calculation import *
from gap_fit_sub import *
from sys import argv
from os import system, getcwd
import numpy as np

# Train the 2b part potential
settings = [['gap={soap atom_sigma=0.75}', 'gap={soap atom_sigma=0.5}'],
            [['default_sigma={0.01 0.4 2.0 0.0}'],
             ['default_sigma={0.01 0.4 0.0 0.0}', 'virial_parameter_name=NOT_USED'],
             ['default_sigma={0.002 0.2 0.4 0.0}'],
             ['default_sigma={0.002 0.2 0.0 0.0}', 'virial_parameter_name=NOT_USED']]]

for i in settings[1]:
    if len(i) == 2:
        v='F'
    else:
        v='T'
    gap_file='gap_file=ds{0}_v{1}.xml'.format(i[0].split('.')[1].split()[0],
                                                v)
    gap_str, gap_args = gap_fit_create(*i, gap_file, no_soap=True)
    system('gap_fit {}'.format(gap_str))

    db_2b = GAP('../train_216_125_64_v.xyz', pot='{}'.format(gap_args['gap_file']))
    db_2b.calc(val=False)
    db_2b.analyse()

    with open('{}.out'.format(gap_file), 'w') as f:
        f.write('Energy RMSE error / eV:\n{}\n'.format(
            np.average([i['rmse'] for i in db_2b.data_dict['E_rmse_t']])))
