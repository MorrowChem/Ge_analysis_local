from ase.build import bulk
from ase.lattice import tetragonal, orthorhombic, hexagonal
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.calculators import castep
from ase.io import castep as iocastep
import os
import numpy as np
from copy import deepcopy
import pickle

file = 'xtals_for_SOAP.pickle'
with open(file, 'rb') as f:
    inits = pickle.load(f)

labels = ['fcc', 'dia', 'hcp', 'bcc', 'sc', 'sh', 'bSn', 'Imma', 'Cmca']

directory = './fcc_geom_fine'
ats = []
val = inits[0][0] # fcc 0-pressure GAP-18 opt
i = 0

for ct, p in enumerate(np.linspace(-1,25,50)):
    calc = castep.Castep(label=labels[i],
                         castep_command='mpirun -np $NMPI /usr/local/CASTEP-20/castep.mpi')

    val.calc = calc
    val.calc.cell.external_pressure = 'GPa ! Units optional\n'\
    '{} 0.0 0.0 ! Isotropic presssure of 4GPa\n'\
    '{} 0.0 ! Only upper half of 3x3 matrix specified\n'\
    '{} ! Castep makes symetric lower half\n'.format(p, p, p)

    calc.set_kpts({'spacing' : 0.08})
    calc.merge_param('/u/vld/hert5155/scripts/castep/template.param')
    calc.param.task = 'GeometryOptimization'
    calc.param.elec_energy_tol = 1e-4

    calc._directory = directory
    calc._rename_existing_dir = False
    calc._label = '{}_geom_{}'.format(labels[i], ct)
    print('\n{}\n'.format(calc._label))
    calc._set_atoms = True

    cell = deepcopy(val.get_cell())
    out = None
    
    if os.path.isdir(directory):
        if calc._label + '.castep' in os.listdir(directory):
            try:
                out = iocastep.read_seed(os.path.join(directory, calc._label))
                print('Existing .castep found and read. Warnings associated: {}'.format(out.calc._warnings))
            except:
                print('.castep found, but can\'t be read correctly so reoptimising geometry')

    if out is None:
        if calc.dryrun_ok():
            print('%s : %s ' % (val.calc._label, val.get_potential_energy()))
            out = iocastep.read_seed(os.path.join(directory, calc._label))
        else:
            print("Found error in input")
            print(calc._error)
    
    with open(os.path.join(directory, calc._label + '.geom'),'r') as f:
        out_geom = iocastep.read_castep_geom(f, index=-1)

    new_cell = out_geom.get_cell()
    new_cell[np.abs(new_cell) < 1e-6] = 0
    val.set_cell(new_cell.round(5), scale_atoms=True)
    ats.append(val.copy())
    #cells[ct].append(new_cell.round(5))
    val.set_calculator(None)
    
    with open('castep_opt_xtals_tmp.pickle', 'wb') as f:
            pickle.dump(ats, f)

with open('castep_opt_xtals.pickle', 'wb') as f:
    pickle.dump(ats, f)
