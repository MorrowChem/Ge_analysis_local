'''Script to read DFT from and apply GAP potential to a structural database for purposes
of error analysis. Writes a python list to file containing.

Parallelisation of GAP QUIP calculations would be good (speed up object creation)'''

import numpy as np
import re
import pickle
import quippy
from quippy.potential import Potential
from quippy.descriptors import Descriptor
from ase import Atoms
from ase.io import write, read
import pickle
import matplotlib.pyplot as plt
from sklearn import decomposition


class GAP:


    def __init__(self, train_file, val_file=None, pot=None, parameter_names=('dft_energy', 'dft_forces', 'dft_virial'),
                 sorted_order=None):
        self.dft_energy, self.dft_forces, self.dft_virial = parameter_names
        if pot:
            self.pot = Potential(param_filename=pot)
        self.T_ct = 0
        self.V_ct = 0
        with open(train_file) as f:
            for i in f.readlines():
                if re.match("^[1-9]+$", i):
                    self.T_ct += 1
        print('Training set structure count:', self.T_ct)
        if val_file:
            with open(val_file) as f:
                for i in f.readlines():
                    if re.match("^[1-9]+$", i):
                        self.V_ct += 1
            print('Validation set structure count:', self.V_ct)
            V_set = [read(val_file, format='extxyz', index=i) for i in range(1, self.V_ct)]
        # skip the first entry (reference atomic energy)
        T_set = [read(train_file, format='extxyz', index=i) for i in range(1, self.T_ct)]
        self.zero_e = read(train_file, format='extxyz', index=0).info[self.dft_energy]
        
        self.config_labels = []
        print('Read configs, now fixing virials')
        for i in T_set:
            if not i.info['config_type'] in self.config_labels:
                self.config_labels.append(i.info['config_type'])
            if self.dft_virial not in i.info:
                i.info[self.dft_virial] = None
        print('Config labels:', self.config_labels)
        self.T_configs = [[i for i in T_set if i.info['config_type'] == j] for j in self.config_labels]

        self.qm_t = [[[at.info[self.dft_energy]/len(at) for at in j] for j in self.T_configs],
                [[i for at in j for i in at.get_array(self.dft_forces).flatten()] for j in self.T_configs],
                [[at.info[self.dft_virial] for at in j] for j in self.T_configs]]
        if val_file:
            self.V_configs = [[i for i in V_set if i.info['config_type'] == j] for j in self.config_labels]
            self.qm_v = [[[at.info[self.dft_energy]/len(at) for at in j] for j in self.V_configs],
                         [[i for at in j for i in at.get_array(self.dft_forces).flatten()] for j in self.V_configs],
                         [[at.info[self.dft_virial] for at in j] for j in self.V_configs]]
        else:
            self.qm_v = [[] for i in range(len(self.qm_t))]
            self.V_configs = []
        self.data_dict = {'QM_E_t':self.qm_t[0],  'QM_F_t':self.qm_t[1],  'QM_V_t':self.qm_t[2],
                          'QM_E_v':self.qm_v[0],  'QM_F_v':self.qm_v[1],  'QM_V_v':self.qm_v[2]}
        if sorted_order:
            for i in self.data_dict.keys():
                if self.data_dict[i]:
                    self.data_dict[i] = self.sort_by_config_type(self.data_dict[i], sorted_order)
            self.config_labels = self.sort_by_config_type(self.config_labels, sorted_order)
            self.T_configs = self.sort_by_config_type(self.T_configs, sorted_order)
            self.V_configs = self.sort_by_config_type(self.V_configs, sorted_order)
            print('New order: ', self.config_labels)

        self.cfg_i_T = [0]
        self.cfg_i_V = [0]
        for i in self.T_configs:
            self.cfg_i_T.append(len(i) + self.cfg_i_T[-1])
        for i in self.V_configs:
            self.cfg_i_V.append(len(i) + self.cfg_i_V[-1])
        self.cfg_i_T.append(None)
        self.cfg_i_V.append(None)

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
        if val:
            self.gap_v = list(self.calc_gap_observables(self.V_configs, virials=virials))
        if train:
            self.gap_t = list(self.calc_gap_observables(self.T_configs, virials=virials))
        
    def analyse(self, sorted_order=None, virials=True, train=True):
        n = len(self.qm_t); m = len(self.config_labels)
        print(n)
        err_v = [[[(np.array(i) - np.array(j)).tolist() for i, j in zip(self.qm_v[k][l],
                                                                       self.gap_v[k][l])] for l in range(m)] for k in range(n)]
        rms_v = [[self.rms_dict(self.qm_v[k][l], self.gap_v[k][l]) for l in range(m)] for k in range(n)]

        if self.gap_t:
            err_t = [[(np.array(i) - np.array(j)).tolist() for i,j in zip(self.qm_t[k], self.gap_t[k])] for k in range(n)]
            rms_t = [[self.rms_dict(self.qm_t[k][l], self.gap_t[k][l]) for l in range(m)] for k in range(n)]
        else:
            self.gap_t, err_t, rms_t = [[[] for i in range(n)] for i in range(3)]

        self.data_dict = {'QM_E_t':self.qm_t[0],  'QM_F_t':self.qm_t[1],  'QM_V_t':self.qm_t[2],
                          'GAP_E_t':self.gap_t[0], 'GAP_F_t':self.gap_t[1], 'GAP_V_t':self.gap_t[2],
                          'E_err_t':err_t[0], 'F_err_t':err_t[1], 'V_err_t':err_t[2],
                          'E_rmse_t':rms_t[0],'F_rmse_t':rms_t[1],'V_rmse_t':rms_t[2],
                          'QM_E_v':self.qm_v[0],  'QM_F_v':self.qm_v[1],  'QM_V_v':self.qm_v[2],
                          'GAP_E_v':self.gap_v[0], 'GAP_F_v':self.gap_v[1], 'GAP_V_v':self.gap_v[2],
                          'E_err_v':err_v[0], 'F_err_v':err_v[1], 'V_err_v':err_v[2],
                          'E_rmse_v':rms_v[0],'F_rmse_v':rms_v[1],'V_rmse_v':rms_v[2]}

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

        return self.red
