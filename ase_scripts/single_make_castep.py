# coding: utf-8

# This generates CASTEP cell files from xyz coordinates, in which the Si -> Ge
# substitution is made. NB the cell files will additionally need information
# about k-point sampling, e.g. by appending "kpoints_mp_spacing : 0.03"

from ase.io import read, write
import numpy as np
from sys import argv

def make_castep(this_file, kpt_spacing=0.03):
    a = read(this_file)
    b = a.copy()
    b.set_atomic_numbers([32]*len(b))
    
    # randomise slightly
    r = 0.99 + 0.02 * np.random.random_sample()
    
    # scale by ratio of Ge-Ge / Si-Si bond lengths (https://doi.org/10.1063/1.1702202)
    b.set_cell(a.get_cell() * (2.450/2.352) * r, scale_atoms=True)
    
    write(this_file[:-4]+'.cell', b, precision=8)
    
    with open(this_file[:-4]+'.cell', 'a+') as f:
        f.write('kpoints_mp_spacing : ' + str(kpt_spacing) + '\n')
    return
    
make_castep(argv[1])
