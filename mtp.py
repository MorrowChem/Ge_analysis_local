'''File containing my burgeoning ASE-support for MTPs
TODO:
formalise the read and write functions so that they align with the recommended coding style of ase (inhereting
stuff properly, naming etc.

add in all the documentation for the calculator

share with the rest of the group first'''
from ase.io.extxyz import read_xyz
import numpy as np
from ase.atoms import Atoms
# from itertools import islice
# import re
# import warnings
# import json
# import numbers
# from ase.calculators.calculator import all_properties, Calculator
# from ase.calculators.singlepoint import SinglePointCalculator
# from ase.spacegroup.spacegroup import Spacegroup
# from ase.parallel import paropen
# from ase.constraints import FixAtoms, FixCartesian
from ase.io.formats import index2range
from io import StringIO, UnsupportedOperation
from copy import deepcopy


atom_data_dtype_map = {
    'id': int,
    'type': int,
    'cartes_x': float,
    'cartes_y': float,
    'cartes_z': float,
    'fx': float,
    'fy': float,
    'fz': float,
}


class Error(Exception):
    """Base class"""
    pass


class CFGError(Error):

    def __init__(self, message):
        self.message = message


def write_cfg_db(f, a, force_name='dft_force', virial_name='dft_virial', energy_name='dft_energy'):
    """Writes a list of atoms to the MTP mlip-2 cfg database file format
    WARNING: incurs loss of array information other than positions and forces of atoms.
    Atoms.info dict is stored as features
    Params: set the key words for the forces, energy, and virials if they differ from the defaults above
    """
    if not hasattr(f, 'read'):
        f = open(f, 'w')
        close = True
    else:
        close = False

    if not isinstance(a, list):
        a = [a]

    for ct, val in enumerate(a):
        f.write('BEGIN_CFG\n Size\n')
        f.write('%i\n' % len(val))
        f.write('Supercell\n')
        cell = val.get_cell(complete=True)
        for i in range(3):
            f.write(('{:>10.6f}'*3 + '\n').format(*cell[i]))

        numbers, types = np.unique(val.get_atomic_numbers(), return_inverse=True)
        cart = val.get_positions(wrap=True)
        force = val.get_array(force_name)

        f.write('AtomData: id type cartes_x cartes_y cartes_z fx fy fz\n')
        for ict, ival in enumerate(val):
            f.write(('          {:<5d} {:<5d} ' + '{:<10.8f}   '*6 + '\n').format(
                ict+1, types[ict], *cart[ict], *force[ict]))

        f.write('Energy\n{}\n'.format(val.info[energy_name]))

        if virial_name in val.info.keys() and isinstance(val.info[virial_name], np.ndarray):

            if val.info[virial_name].shape in [(6,), (6, 1)]:
                stress = val.info[virial_name]
            else:
                stress = np.reshape(val.info[virial_name], (3,3))

                stress = np.concatenate((stress.diagonal(),
                                         stress[1,2], stress[0,2], stress[0,1]),
                                        axis=None)

            f.write('PlusStress: xx     yy     zz     yz     xz     xy\n')
            f.write(('          '+'{:<10.8f}   '*6 + '\n').format(*stress))


            f.write(('Feature    {} ' + '{} '*len(numbers) + '\n').format('Type_Ar_map',
                                                                          *list(['{}:{}'.format(str(i),str(j)) for i, j in enumerate(numbers)])))
        for i,j in val.info.items():
            if i != energy_name and i != virial_name:
                f.write('Feature    {}    {}\n'.format(str(i).replace('\n', ''),
                                                       str(j).replace('\n', '')))
        f.write('END_CFG\n\n')
    f.seek(0)

    if close:
        f.close()


def read_cfg_db(fileobj, index=slice(0,None), Type_Ar_map=None,
                    energy_label='dft_energy',
                    force_label='dft_force',
                    stress_label='dft_virial',
                temp_file=False):

    if isinstance(fileobj, str):
        fileobj = open(fileobj)
        close = True
    else:
        close = False

    if not isinstance(index, int) and not isinstance(index, slice):
        raise TypeError('Index argument is neither slice nor integer!')

    # If possible, build a partial index up to the last frame required
    last_frame = None
    if isinstance(index, int) and index >= 0:
        last_frame = index
    elif isinstance(index, slice):
        if index.stop is not None and index.stop >= 0:
            last_frame = index.stop

    # scan through file to find where the frames start
    try:
        fileobj.seek(0)
    except UnsupportedOperation:
        fileobj = StringIO(fileobj.read())
        fileobj.seek(0)
    frames = []
    line = fileobj.readline()
    while line:
        frame_pos = fileobj.tell()
        try:
            assert line.strip() == 'BEGIN_CFG', 'Malformed CFG'
            fileobj.readline()
            line = fileobj.readline()
            natoms = int(line)
            while line != 'END_CFG':
                line = fileobj.readline().strip()
            fileobj.readline()
            line = fileobj.readline()

        except ValueError as err:
            raise CFGError('cfg: Expected cfg header but got: {}'
                           .format(err))

        frames.append((frame_pos, natoms))

        if last_frame is not None and len(frames) > last_frame:
            break

    trbl = index2range(index, len(frames))

    for index in trbl:
        frame_pos, natoms = frames[index]
        fileobj.seek(frame_pos)
        fileobj.readline()
        # check for consistency with frame index table
        assert int(fileobj.readline()) == natoms
        yield _read_cfg_frame(fileobj, natoms, Type_Ar_map,
                              energy_label, force_label, stress_label)

    if close and not temp_file:
        fileobj.close()


def _read_cfg_frame(lines, natoms, Type_Ar_map,
                    energy_label, force_label, stress_label):
    # comment line
    features = {}

    line = next(lines).strip()
    if line == 'Supercell':
        pbc = True
        cell = []
        for i in range(3):
            line = next(lines).strip()
            cell.append(line.split())
        cell = np.array(cell, dtype=np.float64)
        line = next(lines).strip()
    else:
        cell = None
        pbc = False

    atom_data = line.split()
    if atom_data[0] != 'AtomData:':
        raise CFGError('AtomData line missing: {}'.format(line))
    else:
        atom_data = atom_data[1:]
        nvec = len(atom_data)
    convs = [atom_data_dtype_map[i] for i in atom_data]

    data = []
    for i in range(natoms):
        try:
            line = next(lines)
        except StopIteration:
            raise CFGError('read_cfg_db: Frame has {} atoms, expected {}'
                           .format(len(data), natoms))
        vals = line.split()
        row = tuple([conv(val) for conv, val in zip(convs, vals)])
        data.append(row)

    try:
        data = np.array(data)
        # print('data is:',data)
    except TypeError:
        raise CFGError('Badly formatted data')

    arrays = {}
    for i, name in enumerate(atom_data):
        arrays.update({name: data[:, i]})

    line = next(lines)

    if line.strip() == 'Energy':
        line = next(lines)
        try:
            Energy = float(line)
            features[energy_label] = Energy
            line = next(lines)
        except:
            'Energy specified, but next line has no single float'
    else:
        Energy = None

    if 'PlusStress' in line:
        line = next(lines)
        voigt = line.split()
        if len(voigt) != 6:
            raise CFGError('Stresses not in 6-vector format (voronoi)')
        Stresses = np.array(
            [[voigt[0], voigt[5], voigt[4]],
             [voigt[5], voigt[1], voigt[3]],
             [voigt[4], voigt[3], voigt[2]]],
            np.float64)

        features[stress_label] = Stresses
        line = next(lines)

    while line.strip().split()[0] =='Feature':
        ln = line.strip().split()
        ln.pop(0)
        if ln[0] == 'Type_Ar_map':
            Type_Ar_map = {}
            for i in ln[1:]:
                type, num = i.split(':')
                Type_Ar_map[int(type)] = int(num)
        else:
            features.update({ln[0] : ' '.join(ln[1:])})
        line = next(lines)

    if line.strip() != 'END_CFG':
        raise CFGError('Bad CFG, reading frame end failed')

    numbers = None
    if 'type' in arrays and Type_Ar_map is not None:
        numbers = [Type_Ar_map[i] for i in arrays['type']]
        del arrays['type']

    charges = None
    if 'charges' in arrays:
        charges = arrays['charges']
        del arrays['charges']

    positions = None
    if 'cartes_x' in arrays:
        x = arrays['cartes_x']
        del arrays['cartes_x']
    else:
        raise CFGError('Bad CFG: no positions')
    if 'cartes_y' in arrays:
        y = arrays['cartes_y']
        del arrays['cartes_y']
    if 'cartes_z' in arrays:
        z = arrays['cartes_z']
        del arrays['cartes_z']
    positions = np.stack((x,y,z), axis=1)

    Forces = None
    if 'fx' in arrays:
        x = arrays['fx']
        del arrays['fx']
        try:
            y = arrays['fy']
            z = arrays['fz']
            del arrays['fy']
            del arrays['fz']
        except KeyError:
            raise CFGError('x-force specified, but y or z missing')
    Forces = np.stack((x,y,z), axis=1)
    arrays[force_label] = Forces

    atoms = Atoms(positions=positions,
                  numbers=numbers,
                  charges=charges,
                  cell=cell,
                  pbc=pbc,
                  info=features)
    #
    # # Read and set constraints
    # if 'move_mask' in arrays:
    #     if properties['move_mask'][1] == 3:
    #         cons = []
    #         for a in range(natoms):
    #             cons.append(FixCartesian(a, mask=arrays['move_mask'][a, :]))
    #         atoms.set_constraint(cons)
    #     elif properties['move_mask'][1] == 1:
    #         atoms.set_constraint(FixAtoms(mask=~arrays['move_mask']))
    #     else:
    #         raise XYZError('Not implemented constraint')
    #     del arrays['move_mask']
    #
    for name, array in arrays.items():
        atoms.new_array(name, array)
    #
    # # Load results of previous calculations into SinglePointCalculator
    # results = {}
    # for key in list(atoms.info.keys()):
    #     if key in per_config_properties:
    #         results[key] = atoms.info[key]
    #         # special case for stress- convert to Voigt 6-element form
    #         if key == 'stress' and results[key].shape == (3, 3):
    #             stress = results[key]
    #             stress = np.array([stress[0, 0],
    #                                stress[1, 1],
    #                                stress[2, 2],
    #                                stress[1, 2],
    #                                stress[0, 2],
    #                                stress[0, 1]])
    #             results[key] = stress
    # for key in list(atoms.arrays.keys()):
    #     if (key in per_atom_properties and len(value.shape) >= 1
    #         and value.shape[0] == len(atoms)):
    #         results[key] = atoms.arrays[key]
    # if results != {}:
    #     calculator = SinglePointCalculator(atoms, **results)
    #     atoms.calc = calculator
    return atoms


import numpy as np
import os
import subprocess
from tempfile import NamedTemporaryFile

from ase.calculators.calculator import FileIOCalculator, all_changes, compare_atoms, PropertyNotImplementedError
from ase.parallel import paropen


def get_mtp_command(mtp_command=''):
    '''Abstracts the quest for a mlp command'''
    if mtp_command:
        return mtp_command
    elif 'MTP_COMMAND' in os.environ:
        return os.environ['MTP_COMMAND']
    else:
        return 'mlp'


class MTP(FileIOCalculator):
    r"""
    MTP Interface Documentation

    Format inspired by the ASE CASTEP calculator

    joe.morrow@queens.ox.ac.uk

    Getting Started:
    ================

    Set the environment variable
    >>> export MTP_COMMAND=' ... '

    Running the Calculator:
    =======================


    TODO: A faster implementation for lists of atoms would be to write a single cfg for all of them
    """

    def __init__(self, potential_file='pot.mtp', potential_name=None, mtp_command=None, train=None):

        #self.__name__ = 'MTP'
        if not os.path.isfile(potential_file):
            raise FileNotFoundError('MTP potential \'{}\' does not exist'.format(potential_file))

        self.potential_file = potential_file
        self.name = potential_name
        self.train = train

        if mtp_command is None:
            mtp_command = get_mtp_command()

        FileIOCalculator.__init__(self, command=mtp_command,
                                 discard_results_on_any_change=True)
        self.implemented_properties = ['energy', 'forces', 'stress', 'grade']
        self._calls = 0
        # self._input = NamedTemporaryFile(mode='w+', suffix='.cfg')
        # self._output = NamedTemporaryFile(mode='w+', suffix='.cfg')

    def calculate(self, atoms, properties=['energy', 'forces', 'stress'],
                  system_changes=all_changes, train=None, timeout=10):
        if atoms is not None:
            self.atoms = atoms.copy()
        elif self.atoms is None:
            raise AttributeError('No atoms provided to calculator')
        if not os.path.isdir(self._directory):
            os.makedirs(self._directory)

        for i in properties:
            if i not in self.implemented_properties:
                return PropertyNotImplementedError('{} not implemented in MTP code'.format(i))

        if system_changes != [] and system_changes != None:
            input = NamedTemporaryFile(mode='w+', suffix='.cfg')
            output = NamedTemporaryFile(mode='w+', suffix='.cfg')
            self.write_input(input)
            self.run(input, output, timeout)
            if 'grade' in properties:
                if train is None:
                    if self.train is None:
                        raise AttributeError('No training database supplied for grade calculation')
                    else:
                        train = self.train

                if not isinstance(train, str) or isinstance(train, list):
                    self.write_input(train)

                output_grade = NamedTemporaryFile(mode='w+', suffix='.cfg')
                als_grade = NamedTemporaryFile(mode='w+', suffix='.als')
                self.calc_grade(train, input, output_grade, als_grade, timeout)

                self.read_results(output, properties=['energy', 'force', 'stress', 'grade'], grade_file=output_grade)

                output_grade.close()
                als_grade.close()
            else:
                self.read_results(output)


            input.close()
            output.close()
        # self.results = {'energy': 0.0,
        #         'forces': np.zeros((len(atoms), 3)),
        #         'stress': np.zeros(6)}
        # 'dipole': np.zeros(3),
        # 'charges': np.zeros(len(atoms)),
        # 'magmom': 0.0,
        # 'magmoms': np.zeros(len(atoms))}

    def run(self, input, output, timeout):

        self._calls += 1
        efs_command = [self.command, 'calc-efs', self.potential_file, input.name, output.name]
        out = subprocess.run(efs_command, capture_output=True, text=True, timeout=timeout)

        if out.stdout:
            print('mlp call stdout:\n{}'.format(out.stdout))
        if out.stderr:
            print('mlp call stderr:\n{}'.format(out.stderr))
        out.check_returncode()

    def calc_grade(self, train, input, output, als_output, timeout, silence=True):

        calc_grade_command = [self.command, 'calc-grade', self.potential_file,
                              train, input.name, output.name, '--als-filename={}'.format(als_output.name)]
        out = subprocess.run(calc_grade_command, capture_output=True, text=True, timeout=timeout)

        if out.stdout and not silence:
            print('mlp call stdout:\n{}'.format(out.stdout))
        if out.stderr and not silence:
            print('mlp call stderr:\n{}'.format(out.stderr))
        out.check_returncode()

    def write_input(self, input, atoms_list=None):

        # null_results = {'energy': 0.0,
        #                 'forces': np.zeros((len(self.atoms), 3)),
        #                 'stress': np.zeros(6)}

        if atoms_list is None:
            atoms = [self.atoms.copy()]
        else:
            atoms = deepcopy(atoms_list)

        for i in atoms:
            i.info['energy'] = 0.0
            i.arrays['forces'] = np.zeros((len(i), 3))
            i.info['stress'] = np.zeros(6)

        write_cfg_db(input.file, atoms,
                     force_name='forces', virial_name='stress', energy_name='energy')

    def read_results(self, output, properties=['energy', 'forces', 'stress'], grade_file=None):

        temp_atoms = next(
            read_cfg_db(output.file, index=0,
                        energy_label='energy',
                        force_label='forces',
                        stress_label='stress')
        )
        self.results['energy'] = temp_atoms.info['energy']
        self.results['forces'] = temp_atoms.arrays['forces']
        self.results['stress'] = temp_atoms.info['stress']

        if 'grade' in properties:
            temp_atoms = next(
                read_cfg_db(grade_file.file, index=0,
                            energy_label='energy',
                            force_label='forces',
                            stress_label='stress')
            )
            self.results['MV_grade'] = float(temp_atoms.info['MV_grade'])

    def calc_grade_bulk(self, at_list, train=None, timeout=600):


        if train is None:
            if self.train is None:
                raise AttributeError('No training database supplied for grade calculation')
            else:
                train = self.train
        else:
            if not isinstance(train, str) or isinstance(train, list):
                self.write_input(train)

        input = NamedTemporaryFile(mode='w+', suffix='.cfg')
        output_grade = NamedTemporaryFile(mode='w+', suffix='.cfg')
        als_grade = NamedTemporaryFile(mode='w+', suffix='.als')

        self.write_input(input, atoms_list=at_list)
        self.calc_grade(train, input, output_grade, als_grade, timeout)

        temp_atoms =  list(read_cfg_db(output_grade.file,
                        energy_label='energy',
                        force_label='forces',
                        stress_label='stress',
                        temp_file=True)
        )
        output_grade.close()
        als_grade.close()
        input.close()

        grades = [float(i.info['MV_grade']) for i in temp_atoms]

        return grades