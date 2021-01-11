from lammps import PyLammps, lammps
from mpi4py import MPI
import os
import numpy as np
from ase.data import atomic_masses, atomic_numbers, atomic_names

env = os.environ
env['rundir'] = 'run_1000_mpi_test'
env['system'] = 'Si'
env['model'] = 'MTP'
env['units'] = 'metal'
env['restart_from'] = 'data'
env['data_file'] = '/home/joe/scripts/lammps/rnd_64.data'
env['pair_style'] = 'mlip {}/mlip.ini'.format(env['rundir'])
env['pair_coeff'] = "* *"
env['p'] = '0'
env['pot'] = 'Si_myDB_liqamocryst_train_24_rb_12.mtp'
comm = MPI.COMM_WORLD

if comm.rank == 0:
    os.makedirs(os.path.join(os.environ['rundir'],'NPT'))
    ini_str = 'mtp-filename {}\nselect FALSE'.format(env['pot'])
    with open(os.path.join(env['rundir'], 'mlip.ini'), 'w') as f:
        f.write(ini_str)

    np.random.seed()
    rand = np.random.randint(0, high=np.iinfo(np.int32).max, dtype=np.uint32)
else:
    rand = None

rand = comm.bcast(rand, root=0)
print(rand)

lmp = lammps()
L = PyLammps(ptr=lmp)

L.log(os.path.join(env['rundir'], 'log_npt_{}_{}.dat'.format(env['system'], env['model'])), 'append')

L.units(env['units'])
L.atom_style('atomic')
if env['restart_from'] == 'continuation':
    L.read_restart(env['rundir'], 'restart_npt_{}_{}.*'.format(env['system'], env['model']))
elif 'data_file' in env.keys():
    L.read_data(env['data_file'])
else:
    L.read_data(env['data_file'])
    if comm.rank == 0:
        print('System size not specified, defaulting to 64 atoms')

L.mass(1, atomic_masses[atomic_numbers[env['system']]])

L.pair_style(env['pair_style'])
L.pair_coeff(env['pair_coeff'])
L.neighbor(2.0, 'bin')
L.neigh_modify('every', 1, 'delay', 0, 'check', 'yes')

Nfreq = 1000; Nevery = 100; Nprint = 100; Nrepeat = Nfreq//Nevery; Ndump = 1000
L.variable('Nfreq', 'equal', Nfreq)
L.variable('Nprint', 'equal', Nprint)
L.variable('Nevery equal', Nevery)
L.variable('Nrepeat', 'equal', Nrepeat)
L.variable('Ndump', 'equal', 1000)

L.group(atomic_names[atomic_numbers[env['system']]], 'type', 1)
L.timestep(0.001)


L.variable('nAtoms equal atoms')
L.fix('removeMomentum', 'all', 'momentum', 1, 'linear', 1, 1, 1)

L.compute('T', 'all', 'temp')
L.fix('TempAve', 'all', 'ave/time', Nevery, Nrepeat, Nfreq, 'c_T')

L.variable('P equal press')
L.fix('PressAve', 'all', 'ave/time', Nevery, Nrepeat, Nfreq, 'v_P')

L.variable('v equal vol')
L.fix('vAve', 'all', 'ave/time', Nevery, Nrepeat, Nfreq, 'v_v')

L.compute('PE', 'all', 'pe', 'pair')
L.variable('PE_atom equal c_PE/v_nAtoms')
L.fix('PEAve_Atom', 'all', 'ave/time', Nevery, Nrepeat, Nfreq, 'v_PE_atom')

L.compute('MSD', 'all', 'msd')

# L.compute('force', 'all', 'property/atom' 'fx')

L.thermo_style('custom', 'step', 'cpu',
               'temp', 'f_TempAve',
               'press', 'f_PressAve',
               'f_PEAve_Atom',
               'vol', 'f_vAve',
               'c_MSD[4]', 'fmax')

L.thermo(Nfreq)
L.thermo_modify('flush', 'yes')

L.dump('trj_movie all cfg {} {}/NPT/dump_npt_{}_{}.*.cfg mass type xs ys zs'.format(
    Nfreq, env['rundir'], env['system'], env['model']))
L.dump_modify('trj_movie element {}'.format(env['system']))

L.restart(Ndump, '{}/restart_npt_{}_{}.*'.format(env['rundir'], env['system'], env['model']))


env['stages'] = 'melt liquid super_liquid quench amo'
# env['steps'] = '2e4 1e5 1e5 1e5 2e4'
env['steps'] = '3e3 3e3 3e3 5e3 3e3'
env['temps'] = '2500 2500  1800 1800  1500 1500 1500 300  300 300'
stages = env['stages'].split()
steps = [int(float(i)) for i in env['steps'].split()]
temps = [float(i) for i in env['temps'].split()]
p_bar = [float(i)*1e4 for i in env['p'].split()]
if len(p_bar) != len(temps):
    print('Global constant pressure set')
    p_bar = [p_bar[0] for i in range(len(temps))]

for i in range(len(stages)):
    L.variable(stages[i], 'equal', sum(steps[:i+1]))

# if comm.rank == 0:
#     step = L.eval('step')
# if step == 0 and steps[0] > 0:
if env['restart_from'] == 'data' and steps[0] > 0:
    L.velocity('all', 'create', temps[0], rand)
L.run(0)

# if comm.rank == 0:
#     step = L.eval('step')
# else:
#     step = None
#
# comm.bcast(step, root=0)
# print('step is ', step)
step = 0

for i in range(len(stages)):
    L.fix('integrate all npt',
          'temp', temps[2*i], temps[2*i+1], 0.1,
          'iso', p_bar[2*i], p_bar[2*i+1], 1.0,
          'nreset', 1000)
    # if step < steps[i]:
    # L.run(sum(steps[:i+1]), 'upto', 'start', sum(steps[:i]), 'stop', sum(steps[:i+1]), 'pre', 'no', 'post', 'no')
    L.run(sum(steps[:i+1]), 'upto')
    step += steps[i]
    L.unfix('integrate')
    L.write_data('{}/post_{}_data_npt_{}_{}'.format(env['rundir'], stages[i], env['system'], env['model']))

MPI.Finalize()