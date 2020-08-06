from sys import path
path.append('/Users/Moji/Applications/QUIP/build/darwin_x86_64_gfortran')
from ase.build import bulk
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.calculators import castep
from ase.io import read, write
from ase.eos import EquationOfState
import os
import numpy as np

calc = castep.Castep(label='dia',
                     castep_command='mpirun $MPI_HOSTS castep.mpi --dryrun',
                     kpts={'spacing' : 0.03})
print(calc)
a = 5.6858
dia = bulk('Ge', crystalstructure='diamond', a=a, cubic=True)
dia.set_calculator(calc)
cell = dia.get_cell()
print(cell)

traj = Trajectory('Ge_d_castep.traj', 'w')

for i in np.linspace(0.9, 1.1, 10):
    dia.set_cell(cell*i, scale_atoms=True)
    dia.get_potential_energy()
    traj.write(dia)

configs = read('Ge.traj@0:10')
volumes = [dia.get_volume() for dia in configs]
energies = [dia.get_potential_energy() for dia in configs]
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
eos.plot('dia-eos.png', show=False)
