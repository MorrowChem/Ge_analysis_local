from gap_fit_sub import *
import os
import numpy as np
universal = ['sparse_separate_file=F']
settings = [['gap={soap atom_sigma=0.6}', 'gap={soap atom_sigma=0.5}'],
            [['default_sigma={0.01 0.4 2.0 0.0}'],
             ['default_sigma={0.002 0.2 0.4 0.0}'],
             ['default_sigma={0.002 0.2 0.4 0.0}', 'config_type_sigma={'+
              'amorph:0.01:0.2:0.4:0.0:'+
              'hiT_amorph:0.006:0.2:0.3:0.0:'+
              'inter:0.002:0.2:0.2:0.0:'+
              'liq:0.002:0.2:0.2:0.0:'+
              'hiT_liq:0.002:0.2:0.2:0.0}'
              ]]]
ct = 0
for i in settings[0]:
    for j in settings[1]:
        '''print('\n\n')
        if len(j) == 2:
            v = 'F'
        else:
            v = 'T'''''
        v='T'
        gap_file='gap_file=as{0}_ds{1}_v{2}.xml'.format(i.split('=')[-1][2:-1],
                                                    j[0].split('.')[1].split()[0],
                                                    v)
        ct += 1
        gap_args = gap_fit_create(i, *j, gap_file, *universal)[1]
        delta = np.around(train_2b(gap_args, 'train_216_125_64_v.xyz'), decimals=3)
        #delta=0.145
        gap_str = gap_fit_create('gap={{soap delta={}}}'.format(delta), gap_args=gap_args)[0]
        write_sub(gap_str, str(ct)+'.sh')

