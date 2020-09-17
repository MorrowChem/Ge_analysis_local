# coding: utf-8

# This generates CASTEP cell files from xyz coordinates, in which the Si -> Ge
# substitution is made. NB the cell files will additionally need information
# about k-point sampling, e.g. by appending "kpoints_mp_spacing : 0.03"

from ase.io import read, write
import numpy as np
import sys
import os
from glob import glob

def make_castep(this_file, ct=0, kpt_spacing=0.03, target=0):
    #writefile = str(ct) + '_' +  this_file.split('/')[0] + '_' + str(target) + '.cell'
    writefile = str(ct) + '.cell'
    a = read(this_file)
    b = a.copy()
    b.set_atomic_numbers([32]*len(b))
    
    # randomise slightly
    r = 0.99 + 0.02 * np.random.random_sample()
    
    # scale by ratio of Ge-Ge / Si-Si bond lengths (https://doi.org/10.1063/1.1702202) 2.45/2.352
    # new ratio for liquid only: https://doi.org/10.1143/JJAP.34.4124 and https://doi.org/10.1088/0305-4608/18/11/007
    b.set_cell(a.get_cell() * (1.105) * r, scale_atoms=True) ####### ratio
    
    write(writefile, a, precision=8) # set to a/b for original/modified cells

    with open(writefile, 'r+') as f:
        f.seek(0, 0)
        f.write('# run number: {0}\n# time step / fs: '
        '{1}\n'.format(this_file.split('/')[0], target))
        f.seek(0, 2)
        f.write('kpoints_mp_spacing : ' + str(kpt_spacing) + '\n')

    return writefile
    
# Run on a folder with MD output subfolders
dump_dir = 'Si_DFT_64' # change the target directory
os.mkdir(dump_dir) #  new folder for castep files
targets = list(sys.argv[1:]) # script arguments are req. time steps (int)
print(targets)
ct = 1 # counter for run index

for i in os.listdir():
    if os.path.isdir(i):
        for j in targets:
            reg = (i + '/NPT/' + '*.' + j + '.*') # change if data varies
            cfg = glob(reg) # bash-style * globbing
            print(cfg)
            if cfg:
                writefile = make_castep(cfg[0], ct, 0.03, j)
                os.system('cp template.param ' + writefile[:-5] + '.param')
                #  could easily change this last step to a custom .param,
                #  perhaps hard-coded or read by pymatgen
                ct += 1

os.system('mv *.cell ' + dump_dir)
os.system('mv *.param ' + dump_dir)
os.system('mv ' + dump_dir + '/template.param .')
