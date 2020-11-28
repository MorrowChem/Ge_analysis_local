'''Objects to store the training/validation databases and associated
calculations for analysis purposes. Also an object to store MD runs
and facilitate efficient calculation of structural information (via wrapper
to R.I.N.G.S.)

Parallelisation of GAP QUIP calculations would be good (speed up object creation)
Memory conservation is a must
'''

import numpy as np
from re import match, compile
import pickle
import quippy
from quippy.potential import Potential
from quippy.descriptors import Descriptor
from ase.atoms import Atoms
from ase.io import write, read
import pickle
import matplotlib.pyplot as plt
from sklearn import decomposition
import time
from glob import glob
from ase.io.cfg import read_cfg
from ase.io.proteindatabank import write_proteindatabank
from Ge_analysis import *
import os
from shutil import rmtree
import pandas as pd


class GAP:


    def __init__(self, train_file, val_file=None, pot=None, parameter_names=('dft_energy', 'dft_forces', 'dft_virial'),
                 sorted_order=None):
        self.dft_energy, self.dft_forces, self.dft_virial = parameter_names
        self.data_dict = {}
        if isinstance(pot, str):
            self.pot = Potential(param_filename=pot)
        else:
            self.pot = pot
        self.T_ct = 0
        self.V_ct = 0
        pattern = compile(r'^[1-9]+$')
        with open(train_file) as f:
            for i in f.readlines():
                if match(pattern, i):
                    self.T_ct += 1
        print('Training set structure count:', self.T_ct)
        if val_file:
            with open(val_file) as f:
                for i in f.readlines():
                    if match(pattern, i):
                        self.V_ct += 1
            print('Validation set structure count:', self.V_ct)
            V_set = read(val_file, format='extxyz', index=slice(None, None, None))
            for i in V_set:
                if self.dft_virial not in i.info:
                    i.info[self.dft_virial] = np.ones(9) * np.NaN
        print('Reading xyz file (may take a while)')
        # skip the first entry (reference atomic energy)
        T_set = read(train_file, format='extxyz', index=slice(1, None, None))
        self.zero_e = read(train_file, format='extxyz', index=0).info[self.dft_energy]
        
        self.config_labels = []
        print('Read configs, now fixing virials')
        for i in T_set:
            if not i.info['config_type'] in self.config_labels:
                self.config_labels.append(i.info['config_type'])
            if self.dft_virial not in i.info:
                i.info[self.dft_virial] = np.ones(9)*np.NaN
        print('Config labels:', self.config_labels)
        self.T_configs = [[i for i in T_set if i.info['config_type'] == j] for j in self.config_labels]
        if val_file:
            self.V_configs = [[i for i in V_set if i.info['config_type'] == j] for j in self.config_labels]
        else:
            self.V_configs = []

        if sorted_order:
            self.sorted_order = sorted_order
            self.data_dict.update({'sorted_order': self.sorted_order})
            self.config_labels = self.sort_by_config_type(self.config_labels, sorted_order)
            self.T_configs = self.sort_by_config_type(self.T_configs, sorted_order)
            self.V_configs = self.sort_by_config_type(self.V_configs, sorted_order)
            print('New order: ', self.config_labels)

        self.qm_t = [[[at.info[self.dft_energy]/len(at) for at in j] for j in self.T_configs],
                [[i for at in j for i in at.get_array(self.dft_forces).flatten()] for j in self.T_configs],
                [[at.info[self.dft_virial] for at in j] for j in self.T_configs]]

        if val_file:
            self.qm_v = [[[at.info[self.dft_energy]/len(at) for at in j] for j in self.V_configs],
                         [[i for at in j for i in at.get_array(self.dft_forces).flatten()] for j in self.V_configs],
                         [[at.info[self.dft_virial] for at in j] for j in self.V_configs]]
        else:
            self.qm_v = [[] for i in range(len(self.qm_t))]

        self.data_dict.update({'QM_E_t':self.qm_t[0],  'QM_F_t':self.qm_t[1],  'QM_V_t':self.qm_t[2],
                          'QM_E_v':self.qm_v[0],  'QM_F_v':self.qm_v[1],  'QM_V_v':self.qm_v[2],
                          'T_configs':self.T_configs, 'V_configs':self.V_configs})

        self.cfi_i_T = [0]
        self.cfi_i_V = [0]
        for i in self.T_configs:
            self.cfi_i_T.append(len(i) + self.cfi_i_T[-1])
        for i in self.V_configs:
            self.cfi_i_V.append(len(i) + self.cfi_i_V[-1])
        self.cfi_i_T.append(None)
        self.cfi_i_V.append(None)
        

    def save(self, outfile):
        f = open(outfile, 'wb')
        pickle.dump(self.data_dict, f)
        f.close()

    def load(self, infile):
        f = open(infile, 'rb')
        self.data_dict = pickle.load(f)
        self.gap_v = [self.data_dict['GAP_E_v'], self.data_dict['GAP_F_v'], self.data_dict['GAP_V_v']]
        self.gap_t = [self.data_dict['GAP_E_t'], self.data_dict['GAP_F_t'], self.data_dict['GAP_V_t']]
        self.qm_v = [self.data_dict['QM_E_v'], self.data_dict['QM_F_v'], self.data_dict['QM_V_v']]
        self.qm_t = [self.data_dict['QM_E_t'], self.data_dict['QM_F_t'], self.data_dict['QM_V_t']]
        print('Load successful\ndata_dict: ', self.data_dict.keys())
        if 'sorted_order' in self.data_dict.keys():
            self.config_labels = self.sort_by_config_type(self.config_labels, self.data_dict['sorted_order'])
            self.T_configs = self.sort_by_config_type(self.T_configs, self.data_dict['sorted_order'])
            self.V_configs = self.sort_by_config_type(self.V_configs, self.data_dict['sorted_order'])
        f.close()

    def calc_gap_observables(self, configs, virials=True):
        '''Calculates GAP forces, energies and virials by config_type'''
        E = [[] for i in configs]
        F = [[] for i in configs]
        V = [[] for i in configs]
        for i, val in enumerate(configs):
            for dba in val:
                a = dba.copy()
                a.set_calculator(self.pot)
                E[i].append(a.get_total_energy()/len(a))
                F[i].extend(a.get_forces().flatten())
                if virials:
                    V[i].append((-a.get_stress(voigt=False)*a.get_volume()).ravel())
            print('Config %s done' % self.config_labels[i])
        return E, F, V

    def basic_calc(self, configs, virials=True):
        E, F, V = [], [], []
        for i, val in enumerate(configs):
            val.set_calculator(self.pot)
            E[i].append(val.get_total_energy()/len(val))
            F[i].extend(val.get_forces().flatten())
            if virials:
                V[i].append((-val.get_stress(voigt=False)*val.get_volume()))
        return E, F, V

    def rms_dict(self, x_ref, x_pred):
        """ Takes two datasets of the same shape and returns a dictionary containing RMS error data"""

        x_ref = np.array(x_ref)
        x_pred = np.array(x_pred)

        if np.shape(x_pred) != np.shape(x_ref):
            raise ValueError('WARNING: not matching shapes in rms. Shapes: {0}, {1}'
                             .format(np.shape(x_ref), np.shape(x_pred)))

        error_2 = (x_ref - x_pred) ** 2

        average = np.sqrt(np.average(error_2))
        std_ = np.sqrt(np.var(error_2))

        return {'rmse': average, 'std': std_}

    def flatten(self, o):
        return [item for sublist in o for item in sublist]


    def sort_by_config_type(self, d, sorted_order):
        return [d[i] for i in sorted_order]


    def calc(self, virials=True, train=True, val=True):
        st = time.time()
        if val:
            self.gap_v = list(self.calc_gap_observables(self.V_configs, virials=virials))
        else:
            self.gap_v = None
        if train:
            self.gap_t = list(self.calc_gap_observables(self.T_configs, virials=virials))
        else:
            self.gap_t = None
        print("--- %s seconds ---" % (time.time() - st))

    def analyse(self, sorted_order=None, virials=True, train=True):
        n = len(self.qm_t); m = len(self.config_labels)
        if self.gap_v:
            err_v = [[[(np.array(i) - np.array(j)).tolist() for i, j in zip(self.qm_v[k][l],
                                                                           self.gap_v[k][l])] for l in range(m)] for k in range(n)]
            rms_v = [[self.rms_dict(self.qm_v[k][l], self.gap_v[k][l]) for l in range(m)] for k in range(n)]
        else:
            self.gap_v, err_v, rms_v = [[[] for i in range(n)] for i in range(3)]

        if self.gap_t and train:
            err_t = [[(np.array(i) - np.array(j)).tolist() for i,j in zip(self.qm_t[k], self.gap_t[k])] for k in range(n)]
            rms_t = [[self.rms_dict(self.qm_t[k][l], self.gap_t[k][l]) for l in range(m)] for k in range(n)]
        else:
            self.gap_t, err_t, rms_t = [[[] for i in range(n)] for i in range(3)]

        self.data_dict.update({
                          'GAP_E_t':self.gap_t[0], 'GAP_F_t':self.gap_t[1], 'GAP_V_t':self.gap_t[2],
                          'E_err_t':err_t[0], 'F_err_t':err_t[1], 'V_err_t':err_t[2],
                          'E_rmse_t':rms_t[0],'F_rmse_t':rms_t[1],'V_rmse_t':rms_t[2],
                          'GAP_E_v':self.gap_v[0], 'GAP_F_v':self.gap_v[1], 'GAP_V_v':self.gap_v[2],
                          'E_err_v':err_v[0], 'F_err_v':err_v[1], 'V_err_v':err_v[2],
                          'E_rmse_v':rms_v[0],'F_rmse_v':rms_v[1],'V_rmse_v':rms_v[2]})

        if sorted_order:
            for i in self.data_dict.keys():
                if self.data_dict[i]:
                    self.data_dict[i] = self.sort_by_config_type(self.data_dict[i], sorted_order)
            self.config_labels = self.sort_by_config_type(self.config_labels, sorted_order)
            self.T_configs = self.sort_by_config_type(self.T_configs, sorted_order)
            self.V_configs = self.sort_by_config_type(self.V_configs, sorted_order)
            for i in range(len(self.qm_t)):
                self.gap_v[i] = self.sort_by_config_type(self.gap_v[i], sorted_order)
                self.gap_t[i] = self.sort_by_config_type(self.gap_t[i], sorted_order)
                self.qm_v[i] = self.sort_by_config_type(self.qm_v[i], sorted_order)
                self.qm_t[i] = self.sort_by_config_type(self.qm_t[i], sorted_order)
            print('New order: ', self.config_labels)


    def calc_similarity(self, descriptor=None, zeta=4):
        '''Could read zeta from descriptor specified and use this
        in kernel construction. If construction of the outer product
        is slow, could speed up by taking advantage of symmetry and
        broadcasting'''
        if not descriptor:
            descriptor = quippy.descriptors.Descriptor(
                             'soap average=T l_max=6 n_max=12 atom_sigma=0.5 \
                              cutoff=5.0 cutoff_transition_width=1.0 \
                              central_weight=1.0')
        descs = np.array([descriptor.calc_descriptor(i) for i in self.flatten(self.T_configs)])
        k_mat = np.array([[2 - 2*np.dot(i[0]**zeta, j[0]**zeta) for j in descs] for i in descs])
        pca = decomposition.PCA(n_components=2)
        pca.fit(k_mat)
        self.red = pca.fit_transform(k_mat)


class MD_run:

    def __init__(self, run_dir, label=None):
        self.run_dir = run_dir
        with open(glob(run_dir + '/log*')[0]) as f:
            out = f.readlines()
        flag = 0
        for i, val in enumerate(out):
            test = val.split()
            test.append('')
            if test[0] == 'Step':
                if not hasattr(self, 'dat'):
                    self.dat = [[] for j in range(len(out[i+1].split()))]
                    self.dat_head = val.split()
                flag = 1
                continue
            if flag:
                try:
                    for j, num in enumerate(val.split()):
                        self.dat[j].append(float(num))
                except:
                    flag = 0
        self.dat = np.array(self.dat) # turn into a DataFrame with header
        self.configs = []
        temp = []
        self.timesteps = []
        for i in glob(run_dir + '/NPT/*.cfg'):
            temp.append(read_cfg(i))
            temp[-1].info['file'] = i
            self.timesteps.append(int(i.split('/')[-1].split('.')[1]))
        self.configs = [i for _, i in sorted(zip(self.timesteps, temp))]
        self.timesteps.sort()
        self.df = pd.DataFrame(self.dat[1:].T, columns=self.dat_head[1:], index=self.dat[0].astype(int))
        # should drop dupes based on index (more independent of log setup)
        self.df.drop_duplicates(subset=self.df.columns[-1], inplace=True)
        self.df.insert(0, 'Configs', self.configs)
        self.df.drop(index=0, inplace=True)

        if label:
            self.label = label
        else:
            self.label = run_dir.split('/')[-2]
        return


    def structure_factors(self, selection=None, rings_dir='',
                          discard=False, read_only=False,
                          opts={}, rings_in={},
                          do_bin_fit=False, bin_args=None, rings_command='rings'):
        '''Calculates structure factors and PDFs for selection of MD run, by calling rings
        function from Ge_analysis.py
        Parameters:
            selection: iterable, indices of configurations from MD run (could implement T range etc.)
            rings_dir: str, directory in which to dump the rings outputs
            discard: bool, remove the rings output at the end (wasteful storage-wise)
            read_only: bool, if true, read output from existing rings_dir
            opts: dict, extra options for the rings calc
            rings_in: dict, parameters for the rings calc
        Returns:
            Sq_x, but also sets Sq_x(ray), Sq_n(eutron), gr objects
            of the MD_run'''
        if selection is None:
            selection = tuple(i for i in range(len(self.configs)))
        if not os.path.isdir(rings_dir):
            os.mkdir(rings_dir)
        os.chdir(rings_dir)
        self.rings_dir = os.getcwd()
        self.Sq_n = []; self.Sq_n_av = []; self.Sq_n_std = []
        self.Sq_x = []; self.Sq_x_av = []; self.Sq_x_std = []
        self.Sk_n = []
        self.Sk_x = []
        self.gr = []; self.gr_av = []; self.gr_std = []
        rings_opts = {'S(q)':True,
                      #'S(k)':True,
                      'g(r)':True}
        rings_opts.update(opts)
        if not read_only:
            for i in selection:
                flag = rings(str(i), atoms=self.configs[i], opts=rings_opts, rings_in=rings_in, rings_command=rings_command)
            if not flag:
                print('R.I.N.G.S ran successfully')
        dats = sorted(os.listdir(), key=int)
        self.Sq_timesteps = dats
        self.Sq_x = [read_dat(str(i) + '/sq/sq-xrays.dat') for i in dats]
        self.Sq_n = [read_dat(str(i) + '/sq/sq-neutrons.dat') for i in dats]
        #self.Sk_x = [read_dat(str(i) + '/sk/sk-xrays.dat') for i in dats]
        #self.Sk_n = [read_dat(str(i) + '/sk/sk-neutrons.dat') for i in dats]
        self.gr =   [read_dat(str(i) + '/gr/gr.dat') for i in dats]
        self.rings_list = [str(i) for i in dats]
        os.chdir('../')
        if discard:
            rmtree(rings_dir)
        if do_bin_fit:
            self.bin_fit()
            if 'Angles' in opts:
                self.bond_angle()
        return

    def bin_fit(self, nbins=100, s_selection=None, q_selection=None, ret=False):
        '''s_selection: list of ints, the configs you want to include
        for the averaging of the structure factor (start from 0, could
        change to match the timesteps)
        q_selection: range of q over which you want to average (often want
        to ignore longer-range q, 1,12 is good bet)'''

        # Need to clean up and implement the df properly
        if s_selection:
            a = [self.Sq_x[i] for i in s_selection]
            b = [self.Sq_n[i] for i in s_selection]
            self.Sq_av_T = np.average([self.dat[3][int(self.Sq_timesteps[i])] for i in s_selection])
        else:
            a = self.Sq_x
            b = self.Sq_n
            self.Sq_av_T = np.average([self.dat[3][int(i)] for i in self.Sq_timesteps])
        tmp = np.concatenate([i for i in a], axis=1)
        tmp2 = np.concatenate([i for i in b], axis=1)
        order = np.argsort([i for i in tmp[0]])
        tot_Sq_x = np.array([[tmp[0][i] for i in order],
                                 [tmp[1][i] for i in order]])
        tot_Sq_n = np.array([[tmp2[0][i] for i in order],
                                  [tmp2[1][i] for i in order]])
        x, y = tot_Sq_x[0], tot_Sq_x[1]
        xx, yy = tot_Sq_n[0], tot_Sq_n[1]
        if q_selection:
            mi = np.argmax(x > q_selection[0])
            ma = np.argmax(x > q_selection[1]) + 1
            x, y = x[mi:ma], y[mi:ma]
            xx, yy = xx[mi:ma], yy[mi:ma]
        bins = np.linspace(np.amin(x), np.amax(x), nbins)
        dig = np.digitize(x, bins)
        dig2 = np.digitize(xx, bins)
        self.Sq_x_av.append(np.array([[x[dig == i].mean() for i in range(1, len(bins))],
                              [y[dig == i].mean() for i in range(1, len(bins))]]))
        self.Sq_x_std.append(np.array([[x[dig == i].std() for i in range(1, len(bins))],
                            [y[dig == i].std() for i in range(1, len(bins))]]))
        self.Sq_n_av.append(np.array([[xx[dig2 == i].mean() for i in range(1, len(bins)+1)],
                                 [yy[dig2 == i].mean() for i in range(1, len(bins)+1)]]))
        self.Sq_n_std.append(np.array([[xx[dig2 == i].std() for i in range(1, len(bins)+1)],
                                  [yy[dig2 == i].std() for i in range(1, len(bins)+1)]]))

        if ret:
            return self.Sq_x_av, self.Sq_x_std
        else:
            return


    def bin_fit_g(self, nbins=100, s_selection=None, r_selection=None):
        if s_selection:
            a = [self.gr[i] for i in s_selection]
            self.gr_av_T = np.average([self.dat[3][int(self.Sq_timesteps[i])] for i in s_selection])
        else:
            a = self.gr
            self.gr_av_T = np.average([self.dat[3][int(i)] for i in self.Sq_timesteps])
        tmp = np.concatenate([i for i in a], axis=1)
        order = np.argsort([i for i in tmp[0]])
        self.tot_gr = np.array([[tmp[0][i] for i in order],
                                  [tmp[1][i] for i in order]])
        x, y = self.tot_gr[0], self.tot_gr[1]
        if r_selection:
            mi = np.argmax(x > r_selection[0])
            ma = np.argmax(x > r_selection[1]) + 1
            x, y = x[mi:ma], y[mi:ma]
        bins = np.linspace(np.amin(x), np.amax(x), nbins)
        dig = np.digitize(x, bins)
        self.gr_av.append(np.array([[x[dig == i].mean() for i in range(1, len(bins))],
                                 [y[dig == i].mean() + 1 for i in range(1, len(bins))]]))
        self.gr_std.append(np.array([[x[dig == i].std() for i in range(1, len(bins))],
                                  [y[dig == i].std() for i in range(1, len(bins))]]))

        return

    def bin_bond_angle(self, nbins=100, s_selection=None):
        '''function to calculate (and maybe average) bond angle distributions
        using rings'''
        if hasattr(self, 'bond_angle'):
            self.bond_angle.append(np.array([[], []]))
        else:
            self.bond_angle = [np.array([[], []])]
            self.bond_angle_av = []
            self.bond_angle_std = []
        if s_selection:
            for i in s_selection:
                a = np.genfromtxt(glob(self.rings_dir+'/'+self.Sq_timesteps[i]+'/angles/'+
                                                                          'angle_*.dat')[0])
                self.bond_angle[-1] = np.concatenate([self.bond_angle[-1], a.T], axis=1)
            #self.gr_av_T = np.average([self.dat[3][int(self.Sq_timesteps[i])] for i in s_selection])
        else:
            for i in os.listdir(self.rings_dir):
                a = np.genfromtxt(glob(self.rings_dir+'/'+i+'/angles/'+
                                  'angle_*.dat')[0])
                self.bond_angle[-1] = np.concatenate([self.bond_angle[-1], a.T], axis=1)
            #self.gr_av_T = np.average([self.dat[3][int(self.Sq_timesteps[i])] for i in s_selection])
        order = np.argsort([i for i in self.bond_angle[-1][0]])
        self.bond_angle[-1] = np.array([[self.bond_angle[-1][0][i] for i in order],
                                    [self.bond_angle[-1][1][i] for i in order]])
        x, y = self.bond_angle[-1][0], self.bond_angle[-1][1]
        # if r_selection:
        #     mi = np.argmax(x > r_selection[0])
        #     ma = np.argmax(x > r_selection[1]) + 1
        #     x, y = x[mi:ma], y[mi:ma]
        bins = np.linspace(np.amin(x), np.amax(x), nbins)
        dig = np.digitize(x, bins)
        self.bond_angle_av.append(np.array([[x[dig == i].mean() for i in range(1, len(bins))],
                               [y[dig == i].mean() + 1 for i in range(1, len(bins))]]))
        self.bond_angle_std.append(np.array([[x[dig == i].std() for i in range(1, len(bins))],
                                [y[dig == i].std() for i in range(1, len(bins))]]))

        return

class GAP_pd:


    def __init__(self, train_file, val_file=None, pot=None, parameter_names=('dft_energy', 'dft_forces', 'dft_virial'),
                 sorted_order=None):
        self.dft_energy, self.dft_forces, self.dft_virial = parameter_names
        data_dict = {}
        if isinstance(pot, str):
            self.pot = Potential(param_filename=pot)
        else:
            self.pot = pot
        self.T_ct = 0
        self.V_ct = 0
        pattern = compile(r'^[1-9]+$')
        with open(train_file) as f:
            for i in f.readlines():
                if match(pattern, i):
                    self.T_ct += 1
        print('Training set structure count:', self.T_ct)
        if val_file:
            with open(val_file) as f:
                for i in f.readlines():
                    if match(pattern, i):
                        self.V_ct += 1
            print('Validation set structure count:', self.V_ct)
            V_set = read(val_file, format='extxyz', index=slice(None, None, None))
        print('Reading xyz file (may take a while)')
        # skip the first entry (reference atomic energy)
        T_set = read(train_file, format='extxyz', index=slice(1, None, None))
        self.zero_e = read(train_file, format='extxyz', index=0).info[self.dft_energy]

        self.config_labels = []
        print('Read configs, now fixing virials')
        for i in T_set:
            if i.info['config_type'] not in self.config_labels:
                self.config_labels.append(i.info['config_type'])
            if self.dft_virial not in i.info:
                i.info[self.dft_virial] = None
        print('Config labels:', self.config_labels)
        T_configs = [[i for i in T_set if i.info['config_type'] == j] for j in self.config_labels]
        if val_file:
            V_configs = [[i for i in V_set if i.info['config_type'] == j] for j in self.config_labels]
        else:
            V_configs = []

        if sorted_order:
            self.sorted_order = sorted_order
            self.data_dict.update({'sorted_order': self.sorted_order})
            self.config_labels = self.sort_by_config_type(self.config_labels, sorted_order)
            self.T_configs = self.sort_by_config_type(self.T_configs, sorted_order)
            self.V_configs = self.sort_by_config_type(self.V_configs, sorted_order)
            print('New order: ', self.config_labels)

        self.qm_t = [[[at.info[self.dft_energy]/len(at) for at in j] for j in self.T_configs],
                     [[i for at in j for i in at.get_array(self.dft_forces).flatten()] for j in self.T_configs],
                     [[at.info[self.dft_virial] for at in j] for j in self.T_configs]]

        if val_file:
            self.qm_v = [[[at.info[self.dft_energy]/len(at) for at in j] for j in self.V_configs],
                         [[i for at in j for i in at.get_array(self.dft_forces).flatten()] for j in self.V_configs],
                         [[at.info[self.dft_virial] for at in j] for j in self.V_configs]]
        else:
            self.qm_v = [[] for i in range(len(self.qm_t))]

        self.data_dict.update({'QM_E_t':self.qm_t[0],  'QM_F_t':self.qm_t[1],  'QM_V_t':self.qm_t[2],
                               'QM_E_v':self.qm_v[0],  'QM_F_v':self.qm_v[1],  'QM_V_v':self.qm_v[2],
                               'T_configs':self.T_configs, 'V_configs':self.V_configs})

        self.cfi_i_T = [0]
        self.cfi_i_V = [0]
        for i in self.T_configs:
            self.cfi_i_T.append(len(i) + self.cfi_i_T[-1])
        for i in self.V_configs:
            self.cfi_i_V.append(len(i) + self.cfi_i_V[-1])
        self.cfi_i_T.append(None)
        self.cfi_i_V.append(None)


    def save(self, outfile):
        f = open(outfile, 'wb')
        pickle.dump(self.data_dict, f)
        f.close()

    def load(self, infile):
        f = open(infile, 'rb')
        self.data_dict = pickle.load(f)
        self.gap_v = [self.data_dict['GAP_E_v'], self.data_dict['GAP_F_v'], self.data_dict['GAP_V_v']]
        self.gap_t = [self.data_dict['GAP_E_t'], self.data_dict['GAP_F_t'], self.data_dict['GAP_V_t']]
        self.qm_v = [self.data_dict['QM_E_v'], self.data_dict['QM_F_v'], self.data_dict['QM_V_v']]
        self.qm_t = [self.data_dict['QM_E_t'], self.data_dict['QM_F_t'], self.data_dict['QM_V_t']]
        print('Load successful\ndata_dict: ', self.data_dict.keys())
        if 'sorted_order' in self.data_dict.keys():
            self.config_labels = self.sort_by_config_type(self.config_labels, self.data_dict['sorted_order'])
            self.T_configs = self.sort_by_config_type(self.T_configs, self.data_dict['sorted_order'])
            self.V_configs = self.sort_by_config_type(self.V_configs, self.data_dict['sorted_order'])
        f.close()

    def calc_gap_observables(self, configs, virials=True):
        '''Calculates GAP forces, energies and virials by config_type'''
        E = [[] for i in configs]
        F = [[] for i in configs]
        V = [[] for i in configs]
        for i, val in enumerate(configs):
            for dba in val:
                a = dba.copy()
                a.set_calculator(self.pot)
                E[i].append(a.get_total_energy()/len(a))
                F[i].extend(a.get_forces().flatten())
                if virials:
                    V[i].append((-a.get_stress(voigt=False)*a.get_volume()))
            print('Config %s done' % self.config_labels[i])
        return E, F, V

    def basic_calc(self, configs, virials=True):
        E, F, V = [], [], []
        for i, val in enumerate(configs):
            val.set_calculator(self.pot)
            E[i].append(val.get_total_energy()/len(val))
            F[i].extend(val.get_forces().flatten())
            if virials:
                V[i].append((-val.get_stress(voigt=False)*val.get_volume()))
        return E, F, V

    def rms_dict(self, x_ref, x_pred):
        """ Takes two datasets of the same shape and returns a dictionary containing RMS error data"""

        x_ref = np.array(x_ref)
        x_pred = np.array(x_pred)

        if np.shape(x_pred) != np.shape(x_ref):
            raise ValueError('WARNING: not matching shapes in rms. Shapes: {0}, {1}'
                             .format(np.shape(x_ref), np.shape(x_pred)))

        error_2 = (x_ref - x_pred) ** 2

        average = np.sqrt(np.average(error_2))
        std_ = np.sqrt(np.var(error_2))

        return {'rmse': average, 'std': std_}

    def flatten(self, o):
        return [item for sublist in o for item in sublist]


    def sort_by_config_type(self, d, sorted_order):
        return [d[i] for i in sorted_order]


    def calc(self, virials=True, train=True, val=True):
        st = time.time()
        if val:
            self.gap_v = list(self.calc_gap_observables(self.V_configs, virials=virials))
        else:
            self.gap_v = None
        if train:
            self.gap_t = list(self.calc_gap_observables(self.T_configs, virials=virials))
        else:
            self.gap_t = None
        print("--- %s seconds ---" % (time.time() - st))

    def analyse(self, sorted_order=None, virials=True, train=True):
        n = len(self.qm_t); m = len(self.config_labels)
        if self.gap_v:
            err_v = [[[(np.array(i) - np.array(j)).tolist() for i, j in zip(self.qm_v[k][l],
                                                                            self.gap_v[k][l])] for l in range(m)] for k in range(n)]
            rms_v = [[self.rms_dict(self.qm_v[k][l], self.gap_v[k][l]) for l in range(m)] for k in range(n)]
        else:
            self.gap_v, err_v, rms_v = [[[] for i in range(n)] for i in range(3)]

        if self.gap_t:
            err_t = [[(np.array(i) - np.array(j)).tolist() for i,j in zip(self.qm_t[k], self.gap_t[k])] for k in range(n)]
            rms_t = [[self.rms_dict(self.qm_t[k][l], self.gap_t[k][l]) for l in range(m)] for k in range(n)]
        else:
            self.gap_t, err_t, rms_t = [[[] for i in range(n)] for i in range(3)]

        self.data_dict.update({
            'GAP_E_t':self.gap_t[0], 'GAP_F_t':self.gap_t[1], 'GAP_V_t':self.gap_t[2],
            'E_err_t':err_t[0], 'F_err_t':err_t[1], 'V_err_t':err_t[2],
            'E_rmse_t':rms_t[0],'F_rmse_t':rms_t[1],'V_rmse_t':rms_t[2],
            'GAP_E_v':self.gap_v[0], 'GAP_F_v':self.gap_v[1], 'GAP_V_v':self.gap_v[2],
            'E_err_v':err_v[0], 'F_err_v':err_v[1], 'V_err_v':err_v[2],
            'E_rmse_v':rms_v[0],'F_rmse_v':rms_v[1],'V_rmse_v':rms_v[2]})

        if sorted_order:
            for i in self.data_dict.keys():
                if self.data_dict[i]:
                    self.data_dict[i] = self.sort_by_config_type(self.data_dict[i], sorted_order)
            self.config_labels = self.sort_by_config_type(self.config_labels, sorted_order)
            self.T_configs = self.sort_by_config_type(self.T_configs, sorted_order)
            self.V_configs = self.sort_by_config_type(self.V_configs, sorted_order)
            for i in range(len(self.qm_t)):
                self.gap_v[i] = self.sort_by_config_type(self.gap_v[i], sorted_order)
                self.gap_t[i] = self.sort_by_config_type(self.gap_t[i], sorted_order)
                self.qm_v[i] = self.sort_by_config_type(self.qm_v[i], sorted_order)
                self.qm_t[i] = self.sort_by_config_type(self.qm_t[i], sorted_order)
            print('New order: ', self.config_labels)


    def calc_similarity(self, descriptor=None, zeta=4):
        '''Could read zeta from descriptor specified and use this
        in kernel construction. If construction of the outer product
        is slow, could speed up by taking advantage of symmetry and
        broadcasting'''
        if not descriptor:
            descriptor = quippy.descriptors.Descriptor(
                'soap average=T l_max=6 n_max=12 atom_sigma=0.5 zeta=4 \
                 cutoff=5.0 cutoff_transition_width=1.0 \
                 central_weight=1.0 n_sparse=5000 delta=0.1 \
                 f0=0.0 covariance_type=dot_product \
                 sparse_method=CUR_POINTS')
        descs = np.array([descriptor.calc_descriptor(i) for i in self.flatten(self.T_configs)])
        k_mat = np.array([[2 - 2*np.dot(i[0]**zeta, j[0]**zeta) for j in descs] for i in descs])
        pca = decomposition.PCA(n_components=2)
        pca.fit(k_mat)
        self.red = pca.fit_transform(k_mat)

