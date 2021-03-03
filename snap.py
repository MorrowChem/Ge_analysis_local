from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.calculator import Parameters, equal, PropertyNotImplementedError
from ase.data import atomic_numbers, atomic_names, atomic_masses
import os
import numpy as np
import json

class SNAP(LAMMPSlib):
    r"""
    Wrapper to the LAMMPSlib interface for SNAPs

    joe.morrow@queens.ox.ac.uk

    Getting Started:
    ================

    Running the Calculator:
    =======================

    """

    def __init__(self, atom_types, potential_seed='pot', potential_name=None,
                 zblcutinner=4.0,
                 zblcutouter=4.8,
                 **lmpkwargs):
        '''atom_types - dict: mapping of atom symbols to snapparam lammps types e.g. {'Si:1, Ge:2, O:3, O:4}
           potential_seed -  path-like str: basename of the SNAP potential
           potential name - str: optional name for the potential for use in e.g. plot legends
           zblcutinner - float: parameter for the nuclear-repulsion zbl potential overlayed on the SNAP
           zblcutouter - float: ...
        '''

        if not os.path.isfile(potential_seed+'.snapcoeff') or not os.path.isfile(potential_seed+'.snapparam'):
            raise FileNotFoundError('SNAP potential \'{}\' does not exist'.format(potential_seed))

        self.potential_seed = potential_seed
        self.name = potential_name
        zblz = [atomic_numbers[i] for i in atom_types.keys()]


        lmpcmds = [
                      "pair_style hybrid/overlay zbl {} {} snap".format(zblcutinner, zblcutouter),
                      "pair_coeff * * zbl 0.0 0.0"] + \
                  ["pair_coeff {0} {1} zbl {2} {3}".format(atom_types[i], atom_types[j],
                                                           atomic_numbers[i], atomic_numbers[j])
                   for i in atom_types.keys() for j in atom_types.keys() if atom_types[i] >= atom_types[j]] + \
                  [("pair_coeff * * snap {0}.snapcoeff {0}.snapparam " + \
                    " ".join(["{{{}}}".format(i+1) for i in range(len(atom_types.keys()))])).format(
                      potential_seed, *atom_types.keys())
                  ]

        LAMMPSlib.__init__(self, lmpcmds=lmpcmds, atom_types=atom_types, keep_alive=True, **lmpkwargs) # TODO: leaks a process

    # def set_atoms(self, atoms):
    #     # self.atoms.calc = self
    #     self.restart_lammps(atoms)
    #     self.atoms = atoms
    #     self.__dict__['atoms'] = atoms.copy

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError('{} property not implemented'
                                              .format(name))

        if atoms is None:
            atoms = self.atoms
            system_changes = []
        else:
            system_changes = self.check_state(atoms)
            if system_changes:
                self.reset()
                self.restart_lammps(atoms)
        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms, [name], system_changes)

        if name == 'magmom' and 'magmom' not in self.results:
            return 0.0

        if name == 'magmoms' and 'magmoms' not in self.results:
            return np.zeros(len(atoms))

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError('{} not present in this '
                                              'calculation'.format(name))

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

def write_json_db(a, d="", force_name='dft_force', virial_name='dft_virial', energy_name='dft_energy',
                  weights_dict={}, weights_default=[100, 1, 1e-8]):
    """Writes a list of atoms to the MTP mlip-2 cfg database file format
    WARNING: incurs loss of array information other than positions and forces of atoms.
    Atoms.info dict is stored as features
    Params: set the key words for the forces, energy, and virials if they differ from the defaults above
    """
    d = os.path.join(d, "JSON")
    if not os.path.isdir(d):
        os.makedirs(d)

    # divide DB into list of lists based on config_type
    configs = []; types_list = []
    for i in a:
        if 'config_type' not in i.info.keys():
            i.info['config_type'] = 'default'
        if i.info['config_type'] not in types_list:
            types_list.append(i.info['config_type'])
            configs.append([])
            configs[-1].append(i)
        else:
            configs[types_list.index(i.info['config_type'])].append(i)

    type_lengths = [len(i) for i in configs]
    for i in types_list:
        if i not in weights_dict.keys():
            weights_dict.update({i: weights_default})

    for ci, i in enumerate(configs):
        path = os.path.join(d, types_list[ci])
        os.mkdir(path)
        no_stress_indices = []

        for ct, val in enumerate(i):
            Lattice = val.get_cell()
            det = np.linalg.det(Lattice)
            if det < 0:
                print('Warning, left-handed axes detected, skipping this config:')
                print(det, types_list[ci], 'No.', ct)
                type_lengths[ci] -= 1
                continue
            NumAtoms = len(val)
            AtomTypes = val.get_chemical_symbols()
            Positions = val.get_positions(wrap=True)

            if virial_name in val.info.keys() and 'no_stress' not in val.info['config_type']:
                if isinstance(val.info[virial_name], np.ndarray):
                    Stress = -1e4*np.reshape(val.info[virial_name], (3,3))/val.get_volume()
                    # Need to check that this is the correct form for the stress:
                    # currently as Castep outputs (i.e. non-virial (mechanical) stress, converted to Bar
            elif 'no_stress' not in val.info['config_type']:
                if (td := types_list[ci] + '_no_stress') not in types_list: # move to the relevant *no_stress folder
                    configs.append([])
                    type_lengths.append(0)
                    types_list.append(td)
                    weights_dict.update({td: weights_dict[val.info['config_type']][:-1] + [0]}) # set stress weight to 0
                val.info['config_type'] = td
                configs[(ti := types_list.index(td))].append(val)
                type_lengths[ci] -= 1
                type_lengths[ti] += 1
                continue
            else:
                Stress = np.zeros((3,3))

            Forces = val.get_array(force_name)
            Energy = val.info[energy_name]

            data = {"Dataset": {"Data": [{
                "Stress": Stress.tolist(),
                "Positions": Positions.tolist(),
                "Energy": Energy.tolist(),
                "AtomTypes": AtomTypes,
                "Lattice": Lattice.tolist(),
                "NumAtoms": NumAtoms,
                "Forces": Forces.tolist(),
            }],
                "PositionsStyle": "angstrom", "AtomTypeStyle": "chemicalsymbol",
                "Label": types_list[ci], "StressStyle": "bar",
                "LatticeStyle": "angstrom", "EnergyStyle": "electronvolt",
                "ForcesStyle": "electronvoltperangstrom"
            }}

            with open(os.path.join(d, types_list[ci], val.info['config_type'] + '_{}.json'.format(str(ct))), 'w') as f:
                f.write('# Comment line\n')
                json.dump(data, f)

    with open(os.path.join(d[:-4] + 'grouplist'), 'w') as f:
        f.write('# name size eweight fweight vweight\n')
        for i, val in enumerate(types_list):
            if type_lengths[i] > 0:
                f.write('{:40s}     {:<6d}     {:<7.10f}     {:<7.10f}    {:<7.10f}\n'.format(val, type_lengths[i], *weights_dict[val]))
