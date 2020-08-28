''' A script to convert a directory full of
.cfg output into an extxyz format for rings'''
from ase.io.cfg import read_cfg
from ase.io.proteindatabank import write_proteindatabank
from ase.neighborlist import neighbor_list
import numpy as np
import re
from sys import argv
import os

cfg_directory = [argv[1]]
a=0
c=[]
for ct, direc in enumerate(cfg_directory):
    for i in os.listdir(direc):
        if re.search(r"\.cfg", i):
            a +=1
            c.append(read_cfg(direc + i))
if len(argv) == 2:
    write_proteindatabank(cfg_directory[0]+'run.pdb', c[-1])
else:
    write_proteindatabank(str(argv[])[:-4]+'.pdb', read_cfg(argv[1]+ '/' + argv[2]))
print(a)