''' A script to convert a directory full of
.cfg output into an extxyz format for rings'''
from ase.io.cfg import read_cfg
from ase.io.proteindatabank import write_proteindatabank
from ase.neighborlist import neighbor_list
import numpy as np
import re
from sys import argv
import os

cfg_directory = ['/Users/Moji/Documents/Summer20/Ge/MD_runs/2bSOAP_5000_125_216_d155/run_64001/NPT/']
a=0
c=[]
for ct, direc in enumerate(cfg_directory):
    for i in os.listdir(direc):
        if re.search(r"\.cfg", i):
            a +=1
            c.append(read_cfg(direc + i))
write_proteindatabank(cfg_directory[0]+'run.pdb', c[-1])

print(a)