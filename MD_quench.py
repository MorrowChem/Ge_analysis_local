
from ase.md.npt import NPT
from ase import units, Atoms, build
from ase.io import Trajectory
from ase.io.lammpsdata import read_lammps_data
from ase.md import MDLogger
from quippy.potential import Potential
import time


class MDLogger_NPT:
    """Class for logging molecular dynamics simulations.

    Parameters:
    dyn:           The dynamics.  Only a weak reference is kept.

    atoms:         The atoms.

    logfile:       File name or open file, "-" meaning standard output.

    stress=False:  Include stress in log.

    peratom=False: Write energies per atom.

    mode="a":      How the file is opened if logfile is a filename.
    """
    def __init__(self, dyn, atoms, logfile, header=True, stress=False,
                 peratom=False, mode="a"):
        import ase.parallel
        import weakref
        import sys
        if ase.parallel.world.rank > 0:
            logfile = "/dev/null"  # Only log on master
        if hasattr(dyn, "get_time"):
            self.dyn = weakref.proxy(dyn)
        else:
            self.dyn = None
        self.atoms = atoms
        global_natoms = atoms.get_global_number_of_atoms()
        if logfile == "-":
            self.logfile = sys.stdout
            self.ownlogfile = False
        elif hasattr(logfile, "write"):
            self.logfile = logfile
            self.ownlogfile = False
        else:
            self.logfile = open(logfile, mode, 1)
            self.ownlogfile = True
        self.stress = stress
        self.peratom = peratom
        if self.dyn is not None:
            self.hdr = "%-9s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        else:
            self.hdr = ""
            self.fmt = ""
        if self.peratom:
            self.hdr += "%12s %12s %12s %12s %12s %12s  %6s" % ("Etot/N[eV]", "Epot/N[eV]",
                                                                "Ekin/N[eV]", "Volume[A^3]", "MSD", "P[GPa]", "T[K]")
            self.fmt += "%12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %6.1f"
        else:
            self.hdr += "%12s %12s %12s   %6s" % ("Etot[eV]", "Epot[eV]",
                                                  "Ekin[eV]", "T[K]")
            # Choose a sensible number of decimals
            if global_natoms <= 100:
                digits = 4
            elif global_natoms <= 1000:
                digits = 3
            elif global_natoms <= 10000:
                digits = 2
            else:
                digits = 1
            self.fmt += 3*("%%12.%df " % (digits,)) + " %6.1f"
        if self.stress:
            self.hdr += "      ---------------------- stress [GPa] -----------------------"
            self.fmt += 6*" %10.3f"
        self.fmt += "\n"
        if header:
            self.logfile.write(self.hdr+"\n")

    def __del__(self):
        self.close()

    def close(self):
        if self.ownlogfile:
            self.logfile.close()

    def __call__(self):
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        global_natoms = self.atoms.get_global_number_of_atoms()
        temp = ekin / (1.5 * units.kB * global_natoms)
        vol = self.atoms.get_volume()
        stress = tuple(self.atoms.get_stress(include_ideal_gas=True) / units.GPa)
        press = -sum(stress[:4])/3
        msd = 0.0
        if self.peratom:
            epot /= global_natoms
            ekin /= global_natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000*units.fs)
            dat = (t,)
        else:
            dat = ()
        dat += (epot+ekin, epot, ekin, vol, msd, press, temp)
        if self.stress:
            dat += tuple(self.atoms.get_stress(include_ideal_gas=True) / units.GPa)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()


pot = Potential(param_filename='/Users/Moji/Documents/Summer20/Ge/Potentials/as75_ds002_vF/as75_ds002_vF.xml')
#dia = Atoms(build.bulk('Ge', crystalstructure='diamond', cubic=True))
#dia_s = build.make_supercell(dia, [[2, 0, 0],
#                                   [0, 2, 0],
#                                   [0, 0, 2]])
dia_s = read_lammps_data('rnd_64001.data', style='atomic')
dia_s.set_atomic_numbers([32 for i in range(64)])
dia_s.set_calculator(pot)
sT = 2500*units.kB
fT = 300*units.kB
Ts = [sT - i*(sT-fT)/4 for i in range(5)]
print([i/units.kB for i in Ts])
dyn = NPT(dia_s, 1*units.fs, sT, 0, 25, 3375)

traj = Trajectory('Ge_NPT_test.traj', 'w', dia_s)
dyn.attach(MDLogger_NPT(dyn, dia_s, 'md.log', header=True, stress=True, peratom=True, mode="a"), interval=100)
dyn.attach(traj.write, interval=1000)
meltsteps = 20000
quenchsteps = 20000
st = time.time()
print('running melt...\n')
dyn.run(meltsteps)
for i in range(len(Ts)):
    print('running T ramp {}...\n'.format(i))
    dyn.set_temperature(Ts[i])
    dyn.run(quenchsteps//len(Ts))
print('----- run time = %6.2f' % (time.time() - st))
