from ase.io.extxyz import read_xyz
from ase.io.castep import write_castep_cell
from ase.io import read
import numpy as np
import os
import sys

def make_castep_from_xyz(this_file, dump_dir, param, kpt_spacing=0.03, index=slice(0,None)):
    #writefile = str(ct) + '_' +  this_file.split('/')[0] + '_' + str(target) + '.cell'
    a = list(read_xyz(this_file, index=index))
    writefiles = [dump_dir + str(i) + '.cell' for i in range(len(a))]

    # randomise slightly
    r = 0.99 + 0.02 * np.random.random_sample(len(a))
    for i, b in enumerate(a):
        b.set_atomic_numbers([32]*len(b))
    # scale by ratio of Ge-Ge / Si-Si bond lengths (https://doi.org/10.1063/1.1702202) 2.45/2.352
    # new ratio 1.105 for liquid only: https://doi.org/10.1143/JJAP.34.4124 and https://doi.org/10.1088/0305-4608/18/11/007
        b.set_cell(a[i].get_cell() * 2.45/2.352 * r[i], scale_atoms=True)
        #b.set_cell(a[i].get_cell() * (1.105) * r, scale_atoms=True) ####### ratio
 
        if 'config_type' not in b.info.keys():
            b.info['config_type'] = 'None'

        with open(writefiles[i], 'w') as f:
            write_castep_cell(f, b, precision=8) # set to a/b for original/modified cells
            f.seek(0, 0)
            f.write('# xyz index: {0}\n# config_type: {1}\n'
            .format(i, b.info['config_type']))
            f.seek(0, 2)
            f.write('kpoints_mp_spacing : ' + str(kpt_spacing) + '\n')

        os.system('cp {0} {1}'.format(param, writefiles[i][:-4] + 'param'))
        if i % 10 == 0:
            print('{} is done'.format(i))

    return

#dump_dir = 'Si_supercell_crystals/'
#os.mkdir(dump_dir[:-1])
#make_castep_from_xyz(sys.argv[1], dump_dir, 'template.param', kpt_spacing=0.03)

def make_castep(atoms, dump_dir, param, kpt_spacing=0.03, index=slice(0,None), filetype=None):
    #writefile = str(ct) + '_' +  this_file.split('/')[0] + '_' + str(target) + '.cell'
    if not isinstance(atoms, list) and filetype is None:
        atoms = [atoms]
    elif filetype is not None:
        atoms = list(read(atoms, index=index, format=filetype))

    if dump_dir[-1] != '/':
        dump_dir += '/'
    if not os.path.isdir(dump_dir):
        os.makedirs(dump_dir, exist_ok=True)
    writefiles = [dump_dir + str(i) + '.cell' for i in range(len(atoms))]
    # randomise slightly
    r = 0.99 + 0.02 * np.random.random_sample(len(atoms))
    for i, b in enumerate(atoms):
        b.set_atomic_numbers([32]*len(b))
        # scale by ratio of Ge-Ge / Si-Si bond lengths (https://doi.org/10.1063/1.1702202) 2.45/2.352
        # new ratio 1.105 for liquid only: https://doi.org/10.1143/JJAP.34.4124 and https://doi.org/10.1088/0305-4608/18/11/007
        b.set_cell(atoms[i].get_cell() * 2.45/2.352 * r[i], scale_atoms=True)
        #b.set_cell(atoms[i].get_cell() * (1.105) * r, scale_atoms=True) ####### ratio

        if 'config_type' not in b.info.keys():
            b.info['config_type'] = 'None'

        with open(writefiles[i], 'w+') as f:
            for j in b.info.keys():
                if isinstance(b.info[j], np.ndarray):
                    b.info[j] = b.info[j].tolist() 
                f.write('# {0}: {1}\n'.format(j, b.info[j]))
            f.write('# config_type: {0}\n'
                    .format(b.info['config_type']))
            write_castep_cell(f, b, precision=8)
            f.write('kpoints_mp_spacing : ' + str(kpt_spacing) + '\n')

        os.system('cp {0} {1}'.format(param, writefiles[i][:-4] + 'param'))
        if i % 10 == 0:
            print('{} is done'.format(i))

    return
