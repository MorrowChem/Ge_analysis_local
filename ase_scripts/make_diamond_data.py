from ase.io import write
from ase.build import bulk, make_supercell

prim = bulk('Ge', crystalstructure='diamond', a=5.666, cubic=True)
scell = make_supercell(prim, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

write('diamond.dat', scell, format='lammps-data')
