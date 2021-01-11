from ase.build import bulk
from ase.lattice import tetragonal, orthorhombic, hexagonal
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.calculators import castep
from ase.io import castep as iocastep
import os
import numpy as np
from copy import deepcopy

element='Si'

class bSn_factory(tetragonal.CenteredTetragonalFactory):
    xtal_name='bSn'
    bravais_basis=[[0, 0.5, 0.25], [0.0, 0.0, 0.5]]

class Imma_factory(orthorhombic.BodyCenteredOrthorhombicFactory):
    xtal_name='Imma'
    bravais_basis=[[0.0641, 0.0000, 0.7500], [0.4359, 0.5000, 0.7500]]

bSn_fact = bSn_factory()
bSn = Atoms(bSn_fact(symbol=element, latticeconstant={'a':5.2, 'c':2.87}))

Imma_fact = Imma_factory()
Imma = Imma_fact(symbol=element, latticeconstant={'a':2.87, 'b':4.99, 'c':5.30})

# structs = [bulk(element, crystalstructure='fcc', a=3.0, cubic=True),
#            bulk(element, crystalstructure='diamond', a=5.0,  cubic=True),
#            bulk(element, crystalstructure='hcp', a=3, c=5),
#            bulk(element, crystalstructure='bcc', a=3.0, cubic=True),
#            bulk(element, crystalstructure='sc', a=3.0),
#            Atoms(hexagonal.Hexagonal(symbol=element, latticeconstant={'a':3.0, 'c':3.0})),
#            bSn,
#            Imma]
structs = [bulk(element, crystalstructure='diamond', a=5.0,  cubic=True)]

# labels = ['ccp',
#           'cd',
#           'hcp', 'bcc', 'sc', 'sh', 'bSn', 'Imma']
labels = ['diamond']

directory = './geom'

for i, val in enumerate(structs):
    calc = castep.Castep(label=labels[i],
                         castep_command='mpirun $MPI_HOSTS castep.mpi')

    val.calc = calc

    calc.set_kpts({'spacing' : 0.08})
    calc.merge_param('/data/chem-amais/hert5155/Ge/template.param')
    calc.param.task = 'GeometryOptimization'
    calc.param.elec_energy_tol = 1e-5
    calc.param.fine_grid_scale = 1

    calc._directory = directory
    calc._rename_existing_dir = False
    calc._label = '{}_geom'.format(labels[i])
    print('\n{}\n'.format(calc._label))
    calc._set_atoms = True

    cell = deepcopy(val.get_cell())
    out = [None]

    if calc._label + '.castep' not in os.listdir('geom'):
        try:
            out = iocastep.read_seed(os.path.join('geom', calc._label + '.castep'))
            print('Existing .castep found and read. Warnings associated: {}'.format(out[0].calc._warnings))
        except:
            print('.castep found, but can\'t be read correctly so reoptimising geometry')

    if calc.dryrun_ok() and out[0] is None:
        print('%s : %s ' % (val.calc._label, val.get_potential_energy()))
        out = iocastep.read_seed(os.path.join('geom', calc._label))
    else:
        print("Found error in input")
        print(calc._error)

    out.calc.set_pspot('OTF')

    traj = Trajectory('{}_castep.traj'.format(labels[i]), 'w')
    calc.param.task = 'SinglePoint'
    calc.set_kpts({'spacing' : 0.03})
    calc.param.elec_energy_tol = 1e-6
    calc.param.fine_grid_scale = 2

    for ct, i in enumerate(np.linspace(0.95, 1.05, 10)):
        calc._label = '{}_sp_{}'.format(labels[i], ct)
        val.set_cell(cell*i, scale_atoms=True)
        e = val.get_potential_energy()
        traj.write(val)
        print(e)
