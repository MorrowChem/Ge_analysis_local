from gap_fit_sub import *
import os

settings = [['gap={soap atom_sigma=0.75}', 'gap={soap atom_sigma=0.5}'],
            [['default_sigma={0.01 0.4 2.0 0.0}'],
            ['default_sigma={0.01 0.4 0.0 0.0}', 'virial_parameter_name=NOT_USED'],
            ['default_sigma={0.002 0.2 0.4 0.0}'],
            ['default_sigma={0.002 0.2 0.0 0.0}', 'virial_parameter_name=NOT_USED']]]
ct = 0
for i in settings[0]:
    for j in settings[1]:
        print('\n\n')
        if len(j) == 2:
            v = 'F'
        else:
            v = 'T'
        gap_file='gap_file=as{0}_ds{1}_v{2}.xml'.format(i.split('=')[-1][2:-1],
                                                    j[0].split('.')[1].split()[0],
                                                    v)
        ct += 1
        write_sub(gap_fit_create(i, *j, gap_file)[0], str(ct)+'.sh')

