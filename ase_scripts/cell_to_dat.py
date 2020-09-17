from ase.io.castep import read_castep_cell
from ase.io.lammpsdata import write_lammps_data
from sys import argv

with open(argv[1]) as f:
    struc = read_castep_cell(f)
write_lammps_data("{}.dat".format(argv[1][:-5]), struc)
