'''Script to read DFT from and apply GAP potential to a structural database for purposes
of error analysis. Writes a python list to file containing'''

from sys import path
path.append('/Users/Moji/Applications/QUIP/build/darwin_x86_64_gfortran')
print(path)
import numpy as np
import re
import pickle
import quippy
from quippy.potential import Potential
from quippy.descriptors import Descriptor
from ase import Atoms
from ase.io import write, read


'''zero_e = read(train_file, format='extxyz', index=0).info['dft_energy']
T_set = [read(train_file, format='extxyz', index=i) for i in range(1, T_ct)]
V_set = [read(val_file, format='extxyz', index=i) for i in range(1, V_ct)]
config_labels = []
for i in T_set:
    if not i.info['config_type'] in config_labels:
        config_labels.append(i.info['config_type'])
print('Config labels:', config_labels)
T_configs = [[i for i in T_set if i.info['config_type'] == j] for j in config_labels]
V_configs = [[i for i in V_set if i.info['config_type'] == j] for j in config_labels]'''


class GAP:


    def __init__(self, train_file, val_file, pot):
        self.pot = Potential(param_filename=pot)
        self.T_ct = 0
        self.V_ct = 0
        with open(train_file) as f:
            for i in f.readlines():
                if re.match("^[1-9]+$", i):
                    self.T_ct += 1
        print('Training set structure count:', self.T_ct)
        with open(val_file) as f:
            for i in f.readlines():
                if re.match("^[1-9]+$", i):
                    self.V_ct += 1
        print('Validation set structure count:', self.V_ct)
        # skip the first entry (reference atomic energy)
        T_set = [read(train_file, format='extxyz', index=i) for i in range(1, self.T_ct)]
        V_set = [read(val_file, format='extxyz', index=i) for i in range(1, self.V_ct)]
        self.zero_e = read(train_file, format='extxyz', index=0).info['dft_energy']
        
        self.config_labels = []
        for i in T_set:
            if not i.info['config_type'] in self.config_labels:
                self.config_labels.append(i.info['config_type'])
        print('Config labels:', self.config_labels)
        self.T_configs = [[i for i in T_set if i.info['config_type'] == j] for j in self.config_labels]
        self.V_configs = [[i for i in V_set if i.info['config_type'] == j] for j in self.config_labels]
        
        self.qm_t = [[[at.info['dft_energy']/len(at) for at in j] for j in self.T_configs],
                [[i for at in j for i in at.get_array('dft_forces').flatten()] for j in self.T_configs],
                [[at.info['dft_virial'] for at in j] for j in self.T_configs]]
        self.qm_v = [[[at.info['dft_energy']/len(at) for at in j] for j in self.V_configs],
                [[i for at in j for i in at.get_array('dft_forces').flatten()] for j in self.V_configs],
                [[at.info['dft_virial'] for at in j] for j in self.V_configs]]

        self.data_dict = {'QM_E_t':self.qm_t[0],  'QM_F_t':self.qm_t[1],  'QM_V_t':self.qm_t[2],
                          'QM_E_v':self.qm_v[0],  'QM_F_v':self.qm_v[1],  'QM_V_v':self.qm_v[2]}

    def save(self, outfile):
        import pickle
        f = open(outfile, 'wb')
        pickle.dump(self.data_dict, f)
        f.close()
        
        
    def calc_gap_observables(self, configs, V=1):
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
                if V:
                    V[i].append(-a.get_stress(voigt=False)*a.get_volume())
            print('Config %s done' % val)
        return E, F, V
    

    def rms_dict(x_ref, x_pred):
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

    flatten = lambda l: [item for sublist in l for item in sublist]


    def sort_by_config_type(self, d, map):
        return sorted(d, key=lambda x: d[map.index(x)], reverse=True)
        
        
    def calc_all(self, sorted=False, new_order=None):
        gap_t = list(self.calc_gap_observables(self.T_configs, self.pot))
        gap_v = list(self.calc_gap_observables(self.V_configs, self.pot))
        
        n = len(self.qm_t)
        err_t = [[(np.array(i) - np.array(j)).tolist() for i,j in zip(self.qm_t[k], gap_t[k])] for k in range(n)]
        err_v = [[(np.array(i) - np.array(j)).tolist() for i,j in zip(self.qm_v[k], gap_v[k])] for k in range(n)]

        rms_t = [[self.rms_dict(i,j) for i,j in zip(self.qm_t[k], gap_t[k])] for k in range(n)]
        rms_v = [[self.rms_dict(i,j) for i,j in zip(self.qm_v[k], gap_v[k])] for k in range(n)]
        
        self.data_dict = {'QM_E_t':self.qm_t[0],  'QM_F_t':self.qm_t[1],  'QM_V_t':self.qm_t[2],
                          'GAP_E_t':gap_t[0], 'GAP_F_t':gap_t[1], 'GAP_V_t':gap_t[2],
                          'E_err_t':err_t[0], 'F_err_t':err_t[1], 'V_err_t':err_t[2],
                          'E_rmse_t':rms_t[0],'F_rmse_t':rms_t[1],'V_rmse_t':rms_t[2],
                          'QM_E_v':self.qm_v[0],  'QM_F_v':self.qm_v[1],  'QM_V_v':self.qm_v[2],
                          'GAP_E_v':gap_v[0], 'GAP_F_v':gap_v[1], 'GAP_V_v':gap_v[2],
                          'E_err_v':err_v[0], 'F_err_v':err_v[1], 'V_err_v':err_v[2],
                          'E_rmse_v':rms_v[0],'F_rmse_v':rms_v[1],'V_rmse_v':rms_v[2]}

        if sorted and new_order:
            for i in self.data_dict.values():
                i = self.sort_by_config_type(i, new_order)

'''data_dir = '/Users/Moji/Documents/Summer20/Ge/'
train_file = data_dir + 'Structure_databases/train_216_125_64_v.xyz'
val_file = data_dir + 'Structure_databases/validate_216_125_64_v.xyz'
pickle_file = data_dir + 'Pickles/data_125_216_d155'
pot1 = Potential(param_filename= data_dir + 'Potentials/Ge_2bSOAP_5000_125_216_d155/Ge_2bSOAP_5000_125_216_d155.xml')

d155 = GAP(train_file, val_file, pot)
d155.calc_all()'''