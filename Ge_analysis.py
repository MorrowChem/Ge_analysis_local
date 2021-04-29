'''Set of functions for plotting/analysis potentials and their databases
Plan is to migrate these to methods of the GAP class at some point'''
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import os
import pickle
from quippy.potential import Potential
from quippy.descriptors import Descriptor
from ase import Atoms
from ase import build
import warnings
from ase.io import write, read
from sklearn import decomposition
from ase.geometry.analysis import Analysis
from ase.io.cfg import read_cfg
from ase.io.proteindatabank import write_proteindatabank
from ase.build import bulk
from ase.lattice import hexagonal, tetragonal, orthorhombic
from ase.constraints import StrainFilter, UnitCellFilter, ExpCellFilter, FixAtoms
from pandas import DataFrame
from ase.optimize import BFGS
from Ge_analysis import *
from copy import deepcopy
import re
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
import pandas as pd
from ase.lattice import hexagonal, tetragonal, orthorhombic
from ase.constraints import StrainFilter, UnitCellFilter, ExpCellFilter, FixAtoms
from ase.spacegroup.symmetrize import FixSymmetry
from ase.optimize import BFGS
from ase.data import covalent_radii, atomic_numbers
import warnings
from ase.spacegroup.symmetrize import check_symmetry


def read_dat(filey, head=True):
    with open(filey) as f:
        lines = f.readlines()
    if head:
        s = 1
    else:
        s = 0
    dat = [[] for i in range(len(lines[1].split()))]
    for i in lines[s:]:
        d = i.split()
        for j, val in enumerate(d):
            dat[j].append(float(val))
    dat = np.array(dat)
    return dat


def get_castep_E(file):
    '''Gets the basis-set corrected total energy from .castep'''
    with open(file) as f:
        for a in f.readlines():
            if re.search(" Total energy", a):
                return extract_float(a)
        print('Warning: no energy found')
        return 1


def extract_float(a):
    '''Extracts the last float in a string'''
    for t in a.split():
        try:
            E = float(t)
        except ValueError:
            pass
    return E


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

def sort_by_timestep(d, i):
    return sorted(d[0][i], key=lambda x: d[2][1][d[0][i].index(x)], reverse=True), \
           sorted(d[1][i], key=lambda x: d[2][1][d[1][i].index(x)], reverse=True)


def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

def cn_count(at, r=None):
    if r is None:
        r = covalent_radii[at.get_atomic_numbers()[0]]*2.5
    i = neighbor_list('i', at, r)
    coord = np.bincount(i)
    stat = np.bincount(coord, minlength=16)

    return stat, coord
'''desc_SOAP = Descriptor("soap l_max=10 n_max=10 \
delta=0.5 atom_sigma=0.5 zeta=4 central_weight=1.0 \
f0=0.0 cutoff=5.0 cutoff_transition_width=1.0 \
config_type_n_sparse={isol:1:liq:2000:amorph:2000} \
covariance_type=dot_product sparse_method=CUR_POINTS add_species=T")'''

# Have a look at the pairwise potential
def dimer_curve(pot, ax=None, color='r', label='Ge-Ge'):
    '''Plots a dimer curve. If ax is specified, adds the curve to an
    existing plot for side-by-side comparison with other potentials'''


    dimers = [Atoms("2Ge", positions=[[0,0,0], [x, 0,0]]) for x in np.linspace(1.6,6,100)]
    dimer_curve_pot = []
    
    for dim in dimers:
        dim.set_calculator(pot)
        dimer_curve_pot.append(dim.get_total_energy())
    if ax:
        ax.plot([dim.positions[1,0] for dim in dimers], np.array(dimer_curve_pot[0])/2.0 - zero_e,
                color=color, label=label)
    else:
        fig, ax = plt.subplots(1, 1)
        ax.plot([dim.positions[1,0] for dim in dimers], np.array(dimer_curve_pot[0])/2.0 - zero_e,
                      color='b', label='5.0')
        ax.axhline(0, color='k')
        ax.legend()
        ax.set_ylim(-1, 2.5)
    return fig, ax
#######################################

def crystal_EV_curve(potentials, crystal):
    return


# Fit plotting ####################
colormap = plt.get_cmap('plasma')
colors = [colormap(i) for i in np.linspace(0, 0.8, 5)]
labels = ['240 ps', '180 ps', '160 ps', '120 ps', '20 ps']
xs = np.linspace(-6, 6, 100)


def energy_error(GAP, ax=None, title=None, file=None, by_config=True, color='r', rmse=True, label=None):
    mi = np.amin(flatten(GAP.data_dict['QM_E_v'])) - 0.01 - GAP.zero_e
    ma = np.amax(flatten(GAP.data_dict['QM_E_v'])) - 0.01 - GAP.zero_e
    xs = np.linspace(mi, ma, 100)
    if not ax:
        fig, ax = plt.subplots()
    if by_config:
        for i in range(len(GAP.config_labels)):
            ax.scatter(np.array(GAP.data_dict['QM_E_v'][i]) - GAP.zero_e,
                       np.array(GAP.data_dict['GAP_E_v'][i]) - GAP.zero_e,
                       marker='.', color=colors[i], label=tex_escape(GAP.config_labels[i]))
    else:
        ax.scatter(np.array(flatten(GAP.data_dict['QM_E_v'])) - GAP.zero_e,
                   np.array(flatten(GAP.data_dict['GAP_E_v'])) - GAP.zero_e,
                   marker='.', color=color, label=tex_escape(label))
    ax.set(xlabel='DFT energies per atom / eV', ylabel='GAP energies / eV', title=title,
                       xlim=(mi, ma), ylim=(mi, ma))
    ax.legend(loc='upper left')
    ax.plot(xs, xs, color='k')
    if rmse:
        ax.text((xs[-1] - xs[0])*2/3 + xs[0], (xs[-1] - xs[0])/3 + xs[0], 'Energy RMSE: {0:6.3f} meV\nStdev: {1:6.3f} meV'.format(
            np.average([i['rmse']*1000 for i in GAP.data_dict['E_rmse_v']]),
            np.average([i['std']*1000 for i in GAP.data_dict['E_rmse_v']])))
    plt.show()
    if file:
        fig.savefig(file)
    return ax


########################################

# Forces plotting ####################
def forces_error(GAP, ax=None, title=None, by_config=True, color='r', file=None, rmse=True):
    mi = np.amin(flatten(GAP.data_dict['QM_F_t'])) - 0.01
    ma = np.amax(flatten(GAP.data_dict['QM_F_t'])) - 0.01
    xs = np.linspace(mi, ma, 100)
    if not ax:
        fig, ax = plt.subplots()
    if by_config:
        for i in range(len(GAP.config_labels)):
            ax.scatter(np.array(GAP.data_dict['QM_F_v'][i]),
                       np.array(GAP.data_dict['GAP_F_v'][i]),
                       marker='.', color=colors[i], label=GAP.config_labels[i])
    else:
        ax.scatter(np.array(flatten(GAP.data_dict['QM_F_t'])),
                   np.array(flatten(GAP.data_dict['GAP_F_t'])),
                   color=color)
    ax.set(xlabel='DFT forces per atom / $\mathrm{eV\;Å^{-1}}$', ylabel='GAP forces / $\mathrm{eV\;Å^{-1}}$', title=title,
           xlim=(mi, ma), ylim=(mi, ma))
    ax.legend(loc='upper left')
    ax.plot(xs, xs, color='k')
    if rmse:
        ax.text((xs[-1] - xs[0])*2/3 + xs[0], (xs[-1] - xs[0])/3 + xs[0],
                'Forces RMSE: {0:6.3f} $\mathrm{{eV\;Å^{{-1}}}}$\nStdev: {1:6.3f} $\mathrm{{eV\;Å^{{-1}}}}$'.format(
                    np.average([i['rmse'] for i in GAP.data_dict['F_rmse_v']]),
                    np.average([i['std'] for i in GAP.data_dict['F_rmse_v']])))
    plt.show()
    return ax

def virials_error(GAP, ax=None, title=None, file=None):
    mi = np.amin(flatten(GAP.data_dict['QM_V_t'])) - 0.01
    ma = np.amax(flatten(GAP.data_dict['QM_V_t'])) - 0.01
    xs = np.linspace(mi, ma, 100)
    if not ax:
        fig, ax = plt.subplots()
    for i in range(len(GAP.config_labels)):
        ax.scatter(np.array(GAP.data_dict['QM_V_v'][i]+GAP.data_dict['QM_V_t'][i]),
                   np.array(GAP.data_dict['GAP_V_v'][i]+GAP.data_dict['GAP_V_t'][i]),
                   marker='.', color=colors[i], label=GAP.config_labels[i])
    ax.set(xlabel='DFT virials / $\mathrm{eV}$', ylabel='GAP virials / $\mathrm{eV}$', title=title,
           xlim=(mi, ma), ylim=(mi, ma))
    ax.legend(loc='upper left')
    ax.plot(xs, xs, color='k')
    ax.text((xs[-1] - xs[0])*2/3 + xs[0], (xs[-1] - xs[0])/3 + xs[0],
            'Virials RMSE: {0:6.3f} $\mathrm{{eV}}$\nStdev: {1:6.3f} $\mathrm{{eV}}$'.format(
                np.average([i['rmse'] for i in GAP.data_dict['V_rmse_v']]),
                np.average([i['std'] for i in GAP.data_dict['V_rmse_v']])))
    plt.show()
    if file:
        fig.savefig(file)
    return ax

########################################

def abs_energy_error(GAP, ax=None, title=None):
    mi = np.amin(flatten(GAP.data_dict['QM_E_t'])) - 0.01 - GAP.zero_e
    ma = np.amax(flatten(GAP.data_dict['QM_E_t'])) - 0.01 - GAP.zero_e
    if not ax:
        leg = True
        fig, ax = plt.subplots()
    
    for i in range(len(GAP.config_labels)):
        ax.scatter(np.array(GAP.data_dict['QM_E_v'][i]) - GAP.zero_e, abs(np.array(GAP.data_dict['E_err_v'][i]))*1000,
                        marker='.', s=12, color=colors[i], label=GAP.config_labels[i])
    ax.set(xlabel='DFT Energies / eV', ylabel='Abs GAP error / meV', title=title)
    ax.text(mi, np.amax(abs(np.array(flatten(GAP.data_dict['E_err_v'])))*1000)-2,
            'Energy RMSE: {0:6.3f} meV\nStdev: {1:6.3f} meV'.format(
            np.average([i['rmse']*1000 for i in GAP.data_dict['E_rmse_v']]),
            np.average([i['std']*1000 for i in GAP.data_dict['E_rmse_v']])))
    plt.show()
    if leg:
        ax.legend()
        return ax

def abs_force_error(GAP, ax=None, title=None):
    ma = np.amax(abs(np.array(flatten(GAP.data_dict['F_err_v'])))) - 0.2
    if not ax:
        leg = True
        fig, ax = plt.subplots()

    for i in range(len(GAP.config_labels)):
        ax.scatter(np.array(GAP.data_dict['QM_F_v'][i]), abs(np.array(GAP.data_dict['F_err_v'][i])),
                   marker='.', s=12, color=colors[i], label=GAP.config_labels[i])
    ax.set(xlabel='DFT Forces / $\mathrm{eV\;Å^{-1}}$',
           ylabel='Abs GAP error / $\mathrm{eV\;Å^{-1}}$',
           title=title)
    ax.text(0, ma,
            'Forces RMSE: {0:6.3f} $\mathrm{{eV\;Å^{{-1}}}}$\nStdev: {1:6.3f} $\mathrm{{eV\;Å^{{-1}}}}$'.format(
            np.average([i['rmse'] for i in GAP.data_dict['F_rmse_v']]),
            np.average([i['std'] for i in GAP.data_dict['F_rmse_v']])))
    plt.show()
    if leg:
        ax.legend()
        return ax

def abs_virial_error(GAP, ax=None, title=None):
    ma = np.amax(abs(np.array(flatten(GAP.data_dict['V_err_v'])))) - 0.2
    if not ax:
        leg = True
        fig, ax = plt.subplots()

    ax.scatter(np.array(GAP.data_dict['QM_V_v'][i]), abs(np.array(GAP.data_dict['V_err_v'][i])),
               marker='.', s=12, color=colors[i], label=GAP.config_labels[i])
    ax.set(xlabel='DFT Forces / $\mathrm{eV\;Å^{-1}}$',
           ylabel='Abs GAP error / $\mathrm{eV\;Å^{-1}}$',
           title=title)
    ax.text(0, ma,
            'Forces RMSE: {0:6.3f} $\mathrm{{eV\;Å^{{-1}}}}$\nStdev: {1:6.3f} $\mathrm{{eV\;Å^{{-1}}}}$'.format(
                np.average([i['rmse'] for i in GAP.data_dict['V_rmse_v']]),
                np.average([i['std'] for i in GAP.data_dict['V_rmse_v']])))
    plt.show()
    if leg:
        ax.legend()
        return ax
########## up to here #########
def dens_error_plot(GAP, title=None):
    fig, ax = plt.subplots(1,2)

    x = [np.array(flatten(GAP.data_dict['QM_E_v'])) - GAP.zero_e, np.array(GAP.data_dict['QM_F_v'])]
    y = [abs(np.array(GAP.data_dict['E_err_v'])*1000), abs(np.array(GAP.data_dict['F_err_v']))]
    xy = [np.vstack([x[0], y[0]]),
          np.vstack([x[1], y[1]])]
    z = [gaussian_kde(xy[0])(xy[0]), gaussian_kde(xy[1])(xy[1])]
    idx = [z[0].argsort(), z[1].argsort()]
    x[0], y[0], z[0] = x[0][idx[0]], y[0][idx[0]], z[0][idx[0]]
    x[1], y[1], z[1] = x[1][idx[1]], y[1][idx[1]], z[1][idx[1]]
    emin = np.amin(z[0]); emax = np.amax(z[0])
    fmin = np.amin(z[1]); fmax = np.amax(z[1])

    escat = ax[0].scatter(x[0], y[0], c=z[0], cmap=plt.get_cmap('plasma'),
                           vmin=emin, vmax=emax, s=10, edgecolor='')
    ax[0].text(-0.2, 19, 'RMSE: {0:6.3f} $\mathrm{{eV}}$\nStdev: {1:6.3f} $\mathrm{{eV}}$'.format(
        np.average([i['rmse'] for i in GAP.data_dict['F_rmse_v']]),
        np.average([i['std'] for i in GAP.data_dict['F_rmse_v']])))
    ax[0].set(xlabel='DFT Energies / $\mathrm{eV}$', title='Energies')
    fig.colorbar(escat, ax=ax[0])
    ax[0].set(xlabel='DFT Energies / $\mathrm{eV}$',
               ylabel='Abs GAP error / $\mathrm{eV}$')

    fscat = ax[1].scatter(x[1], y[1], c=z[1], cmap=plt.get_cmap('plasma'),
                           vmin=fmin, vmax=fmax, s=10, edgecolor='')
    ax[1].text(-5, 0.90, 'RMSE: {0:6.3f} $\mathrm{{eV\;Å^{{-1}}}}$\nStdev: {1:6.3f} $\mathrm{{eV\;Å^{{-1}}}}$'
                .format(
        np.average([i['rmse'] for i in GAP.data_dict['F_rmse_v']]),
        np.average([i['std'] for i in GAP.data_dict['F_rmse_v']])))
    ax[1].set(xlabel='DFT Forces / $\mathrm{eV\;Å^{-1}}$', title='Forces')

    fig.colorbar(fscat, ax=ax[1], label='Density of points')
    ax[1].set(xlabel='DFT Forces / $\mathrm{eV\;Å^{-1}}$',
               ylabel='Abs GAP error / $\mathrm{eV\;Å^{-1}}$',
               title='Forces')
    plt.subplots_adjust(hspace=1.0)
    plt.show()
########################################
# extra plot types to incorporate (based on GAP object):
# rdfs (by config-type and generally)
# structure maps (using the TiO2 paper recommended by Volker to make sure it's calculated right
# Density of points plot for the forces

def similarity_map(GAP):
    '''Plots a similarity map based on the
    kernel values calculated by the GAP.calc_similarity method'''
    symbols = ['x' for i in GAP.T_configs]
    colormap = plt.get_cmap('plasma')
    colors = [colormap(i) for i in np.linspace(0, 0.8, len(GAP.T_configs))]

    fig, ax = plt.subplots()
    for i in range(len(GAP.T_configs)):
        ax.scatter(GAP.red.T[0][GAP.cfi_i_T[i]:GAP.cfi_i_T[i+1]],
                   GAP.red.T[1][GAP.cfi_i_T[i]:GAP.cfi_i_T[i+1]],
                   color=colors[i], label=GAP.config_labels[i],
                   marker=symbols[i])
    ax.legend()
    fig.show()

def plot_rdf(GAP, ax=None):
    if not ax:
        leg = True
        fig, ax = plt.subplots()
    ana = [Analysis(i) for i in GAP.T_configs]
    rdf = [np.average(i.get_rdf(5.0, 400), axis=0) for i in ana]
    dists = np.linspace(0, 5.0, 400)

    for i in range(len(rdf)):
        ax.scatter(dists, rdf[i],
                    color=colors[i], label=GAP.config_labels[i])
    ax.legend()
    fig.show()

'''
e_dens_fig, e_dens_ax = plt.subplots(1, len(cutoff_data), figsize=(4*len(cutoff_data), 6))
e_dens_fig.suptitle('Error vs. cutoff n_sparse=2000')
x, y, z, xy, idx, cax = [], [], [], [], [], []
print(len(cutoff_data))
for j, val in enumerate(cutoff_data):
    x.append(np.array(flatten(val[1][2])))
    y.append(abs(np.array(flatten(val[1][6]))))
    xy.append(np.vstack([np.array(flatten(val[1][2])), abs(np.array(flatten(val[1][6])))]))
    z.append(gaussian_kde(xy[j])(xy[j]))
    idx.append(z[j].argsort())
    x[j], y[j], z[j] = x[j][idx[j]], y[j][idx[j]], z[j][idx[j]]
    vmin = np.amin(z); vmax = np.amax(z)
for j in range(len(cutoff_data)):
    cax.append(e_dens_ax[j].scatter(x[j], y[j], c=z[j], cmap=plt.get_cmap('plasma'),
                                    vmin=vmin, vmax=vmax, s=10, edgecolor=''))
    e_dens_ax[j].text(-5, 0.90, 'RMSE: {0:6.3f} $\mathrm{{eV\;Å^{{-1}}}}$\nStdev: {1:6.3f} $\mathrm{{eV\;Å^{{-1}}}}$'.format(
        TFRMSE[j]['rmse'], TFRMSE[j]['std']))
    e_dens_ax[j].set(ylim=(0,1.0), xlabel='DFT Forces / $\mathrm{eV\;Å^{-1}}$', title=cutoffs[j])

e_dens_fig.colorbar(cax[2], ax=e_dens_ax, label='Density of points')
e_dens_ax[0].set(xlabel='DFT Forces / $\mathrm{eV\;Å^{-1}}$',
                 ylabel='Abs GAP error / $\mathrm{eV\;Å^{-1}}$',
                 title='5.0')
e_dens_fig.savefig('GAPPY/density_of_error.png')

# Cumulative error plotting ####################
n_bins = 500
colorcycle = ['b', 'm', 'r', 'cyan', 'Grey']
cu_fig, cu_ax = plt.subplots(1,2,figsize=(12, 4))
for j,val in enumerate(cutoff_data):
    cu_ax[0].hist(abs(np.array(flatten(val[0][6]))), n_bins, density=True, histtype='step',
                  cumulative=True, label='{} training'.format(cutoffs[j]),
                  color=colorcycle[j])
    cu_ax[0].hist(abs(np.array(flatten(val[1][6]))), n_bins, density=True, histtype='step',
                  cumulative=True, label='{} validation'.format(cutoffs[j]),
                  color=colorcycle[j])
    plt.xscale('log')
    cu_ax[1].hist(abs(np.array(flatten(val[0][5])))*1000, n_bins, density=True, histtype='step',
                  cumulative=True, color=colorcycle[j])
    cu_ax[1].hist(abs(np.array(flatten(val[1][5])))*1000, n_bins, density=True, histtype='step',
                  cumulative=True, label='{} validation energies'.format(cutoffs[j]),
                  color=colorcycle[j])
    plt.xscale('log')

# Add a line showing the expected distribution.
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
y = y.cumsum()
y /= y[-1]

cu_ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')

# tidy up the figure
def fix_hist_step_vertical_line_at_end(ax):
    import matplotlib as mpl
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])
fix_hist_step_vertical_line_at_end(cu_ax[0])
fix_hist_step_vertical_line_at_end(cu_ax[1])
cu_ax[0].grid(True)
cu_ax[1].grid(True)
cu_fig.legend(bbox_to_anchor=[1.0, 0.9], loc='upper left')
#cu_ax.set_title('')
cu_ax[0].set_xlabel('Abs force error / $\mathrm{eV\;Å^{-1}}$')
cu_ax[1].set_xlabel('Abs energy error / meV')
cu_ax[0].set_ylabel('Cumulative / %')
cu_ax[0].set(xlim=(0.001, 0.6), ylim=(0, 1), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
             yticklabels=[0, 20, 40, 60, 80, 100], xscale='log')
cu_ax[1].set(xlim=(0.001, 2), ylim=(0, 1), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
             yticklabels=[0, 20, 40, 60, 80, 100], xscale='log')
cu_fig.savefig('GAPPY/cumulative_cutoffcomp.png', bbox_inches='tight')
########################################
abs_ax[0,0].set(xlabel='dft energies / ev', ylabel='abs gap error / mev', title='5.0')
abs_ax[0,0].legend()
abs_ax[0,1].set(xlabel='dft energies / ev', title='5.5')
abs_ax[0,2].set(xlabel='dft energies / ev', title='6.0')
abs_ax[1,0].set(xlabel='dft forces / $\mathrm{ev\;å^{-1}}$',
                ylabel='abs gap error / $\mathrm{ev\;å^{-1}}$')
abs_ax[1,1].set(xlabel='dft forces / $\mathrm{ev\;å^{-1}}$')
abs_ax[1,2].set(xlabel='dft forces / $\mathrm{ev\;å^{-1}}$')
plt.subplots_adjust(wspace=0.5)'''

# learning curve plotting
'''ermse_2000 = rms_dict(flatten(V_qm_energies), flatten(nsparse_list[3][1]))
ERMSE_3000 = rms_dict(flatten(V_qm_energies), flatten(nsparse_list[3][2]))
ERMSE_5000 = rms_dict(flatten(V_qm_energies), flatten(nsparse_list[3][3]))
FRMSE_2000 = rms_dict(flatten(V_qm_forces), flatten(nsparse_list[5][1]))
FRMSE_3000 = rms_dict(flatten(V_qm_forces), flatten(nsparse_list[5][2]))
FRMSE_5000 = rms_dict(flatten(V_qm_forces), flatten(nsparse_list[5][3]))
print(ERMSE_2000,
      ERMSE_3000,
      ERMSE_5000,
      FRMSE_2000,
      FRMSE_3000,
      FRMSE_5000)

lcurve_fig, lcurve_ax = plt.subplots(1,1)
lcurve_dax = lcurve_ax.twinx()
lcurve_ax.errorbar([950, 1950, 2950, 4950], np.array([VERMSE['rmse'], ERMSE_2000['rmse'], ERMSE_3000['rmse'],
                                                  ERMSE_5000['rmse']])*1000, color='b', fmt='^',
                    yerr=np.array([VERMSE['std'], ERMSE_2000['std'], ERMSE_3000['std'], ERMSE_5000['std']])*1000,
                    label='RMSE Energies')
lcurve_ax.set(xlabel='SOAP n_sparse', ylabel='Energy error / meV')
lcurve_dax.errorbar([1050, 2050, 3050, 5050], np.array([VFRMSE['rmse'], FRMSE_2000['rmse'], FRMSE_3000['rmse'],
                                                   FRMSE_5000['rmse']]), color='g', fmt='s',
                    yerr=np.array([VFRMSE['std'], FRMSE_2000['std'], FRMSE_3000['std'], FRMSE_5000['std']]),
                    label='RMSE forces')
lcurve_dax.set(ylabel='Force error / $\mathrm{eV\;Å^{-1}}$')
lcurve_fig.legend()
lcurve_fig.savefig('GAPPY/Ge_RMSE_nsparse_covergence', bbox_inches='tight')'''
########################################

def rings(rdir, cfg_file=None, atoms=None, opts={}, rings_in={}, rings_command='rings'):
    '''Runs rings with inputs set by dictionary. So as not to lose data,
    the output is written to rdir. Provide either a cfg file (lammps dump)
    or an atoms object for analysis

    Needs implementation: rings seems to operate on full MD runs, this may
    be more efficient than running multiple instances on different structures.'''
    os.mkdir(rdir)
    if 'data' not in os.listdir(rdir):
        os.mkdir(rdir + '/data')
    if cfg_file:
        f = read_cfg(cfg_file)
        print(f.get_atomic_numbers()[0])
        #f.set_atomic_numbers([14 for i in range(len(f))])
        nf = cfg_file.split('/')[-1][:-4] + '.pdb'
        write_proteindatabank(rdir + '/data/' + nf, f)
    if atoms:
        f = atoms
        #f.set_atomic_numbers([3 for i in range(len(f))]) # need to make automatic
        nf = f.info['file'].split('/')[-1][:-4] + '.pdb'
        write_proteindatabank(rdir + '/data/' + nf, f)
    N = len(f)
    #Nchem = f.get_chemical_symbols()
    cell = f.get_cell()
    if len(cell) == 3:
        cell = cell.tolist()
    cfg_N = 1

    options = {
       'PBC':True,
       'Frac':False,
       'g(r)':False,
       'S(q)':False,
       'S(k)':False,
       'gfft(r)':False,
       'MSD':False,
       'atMSD':False,
       'Bonds':False,
       'Angles':False,
       'Chains':False,
       'chain_options':{
       'Species':0,
       'AAAA':False,
       'ABAB':False,
       '1221':False},
       'Rings':False,
       'ring_options':{
       'Species':0,
       'ABAB':False,
       'Rings0':False,
       'Rings1':False,
       'Rings2':False,
       'Rings3':False,
       'Rings4':False,
       'Prim_Rings':False,
       'Str_Rings':False,
       'BarycRings':False,
       'Prop-1':False,
       'Prop-2':False,
       'Prop-3':False,
       'Prop-4':False,
       'Prop-5':False},
       'Vacuum':False,
       'Evol':False,
       'Dxout':False,
       'RadOut':False,
       'RingsOut':False,
       'DRngOut':False,
       'VoidsOut':False,
       'TetraOut':False,
       'TrajOut':False,
       'Output':'my-output.out'
    }

    for i in opts:
        if i=='chain_options' or i=='ring_options':
            for j in opts[i]:
                options[i][j] = opts[i][j]
        else:
            options[i] = opts[i]


    with open(rdir + '/options', 'w') as file:
        file.write('#######################################\n' +
                   'R.I.N.G.S. options file       #\n' +
                   '#######################################\n')
        for i in options:
            if i=='chain_options' or i=='ring_options':
                file.write('----- ! {} statistics options ! -----\n'.format(i.split('_')[0]))
                for j in options[i]:
                    if type(options[i][j]) is bool:
                        file.write('{} .{}.\n'.format(j, str(options[i][j])))
                    else:
                        file.write('{} {}\n'.format(j, str(options[i][j])))
                file.write('---------------------------------------\n')
            else:
                if i=='Evol':
                    file.write('#######################################\n' +
                               '  Outputting options           #\n' +
                               '#######################################\n')
                if type(options[i]) is bool:
                    file.write('{} .{}.\n'.format(i, str(options[i])))
                else:
                    file.write('{} {}\n'.format(i, str(options[i])))
                if i in ['Dxout', 'TrajOut']:
                    file.write('-----------------------\n')
        file.write('######################################\n')

    rings_in_def = {
        'label' : 'amorphous Ge',
        'species' : ['Ge'],
        'MD_timestep' : 1,
        'cutoffs' : [3.2], # in format 1:1, 1:2, ... 1:n, 2:2, 2:3.. n:3
        'Grtot' : 3.2,
        'real_disc' : 200,
        'recip_disc' : 500,
        'max_mod_recip' : 25
    }
    for i in rings_in:
        rings_in_def[i] = rings_in[i]

    with open(rdir + '/rings.in', 'w') as file:
        '''Some parts of this could do with automation/user access via
        a dictionary. Left for now'''
        file.write('# R.I.N.G.S input file ######\n#\n#\n' +
        '{}\n'.format(rings_in_def['label']) +  # automate
        '{} # natom\n'.format(N) +
        '1  # number of chemical species\n' +
        str('{} '*len(rings_in_def['species'])).format(*rings_in_def['species']) + '# chemical species\n' +
        '{}  # number of M.D. steps\n'.format(cfg_N) +
        '1 \n' +
        '{}    {}    {}\n'.format(*cell[0]) +
        '{}    {}    {}\n'.format(*cell[1]) +
        '{}    {}    {}\n'.format(*cell[2]) +
        '{} # integration time step for M.D\n'.format(rings_in_def['MD_timestep']) +
        'PDB # file format\n' +
        '{} # name of file\n'.format(nf) +
        '200     # real space discretization g(r)\n' +
        '500     # reciprocal space disc. S(q)\n' +
        '25      # max modulus of recip vectors\n' +
        '0.125   # Smoothing factor for S(q)\n' +
        '90      # angular disc.\n' +
        '20      # real space disc. for voids and ring stats\n' +
        '10      # max search depth/2 for ring stats\n' +
        '15      # max search depth for chain stats\n' +
        '#######################################\n' +
        '{} {}    {}  # cutoff radius for g(r) partials\n'.format(rings_in_def['species'][0],
                                                                   rings_in_def['species'][0],
                                                                  rings_in_def['cutoffs'][0]) +
        'Grtot   {}   # cutoff for total g(r)\n'.format(rings_in_def['Grtot']) +
        '#######################################\n'
        )
    os.chdir(rdir)
    exit = os.system('{} rings.in >rings.log 2>&1'.format(rings_command))
    if exit \
            != 0:
        with open('ring.log', 'r') as f:
            for i in f.readlines():
                print(f)
    os.chdir('../')
    return exit


class bSn_factory(tetragonal.CenteredTetragonalFactory):
    xtal_name='bSn'
    bravais_basis=[[0, 0.5, 0.25], [0.0, 0.0, 0.5]]

class Imma_factory(orthorhombic.BodyCenteredOrthorhombicFactory):
    xtal_name='Imma'
    bravais_basis=[[0.0641, 0.0000, 0.7500], [0.4359, 0.5000, 0.7500]]

# class CrystTest:
#
#     def __init__(self, pots, element,
#                  extra_configs=[], extra_labels=[], n=10,
#                  opt=True, silent=False, load=False,
#                  traj=False):
#         '''pots: list of potentials
#         element: element used for the pots (needs updating for multicomponent systems
#         extra_configs/labels for the non-supported crystal structures (see add_config method)
#         n: number of V points calculated per crystal structure
#         opt: bool, optimize the crystal structures? Need to implement separate opt for
#         each pot if desired
#         silent: bool, silence output'''
#         if not isinstance(pots, list):
#             self.pots = [pots]
#         else:
#             self.pots = pots
#         if load:
#             return
#
#         bSn_fact = bSn_factory()
#         bSn = Atoms(bSn_fact(symbol=element, latticeconstant={'a':5.2, 'c':2.87}))
#
#         Imma_fact = Imma_factory()
#         Imma = Imma_fact(symbol=element, latticeconstant={'a':2.87, 'b':4.99, 'c':5.30})
#
#         structs = [bulk(element, crystalstructure='fcc', a=3.8, cubic=True),
#                    bulk(element, crystalstructure='diamond', a=5.43,  cubic=True),
#                    bulk(element, crystalstructure='hcp', a=3, c=5),
#                    bulk(element, crystalstructure='bcc', a=3.0, cubic=True),
#                    bulk(element, crystalstructure='sc', a=3.0),
#                    Atoms(hexagonal.Hexagonal(symbol=element, latticeconstant={'a':3.0, 'c':3.0})),
#                    bSn,
#                    Imma] + \
#                   [deepcopy(i) for i in extra_configs]
#
#         labels = ['fcc', 'dia', 'hcp', 'bcc', 'sc', 'sh', 'bSn', 'Imma'] + extra_labels
#
#         cells = [[] for i in range(len(pots))]
#         E = [[[] for j in range(len(structs))] for i in range(len(pots))]
#         V = [[] for j in range(len(structs))]
#         for j, p in enumerate(self.pots):
#             copy_structs = deepcopy(structs)
#             for ct, val in enumerate(copy_structs):
#                 if opt and ct < 8:
#                     val.set_calculator(p)
#                     uf = StrainFilter(val)
#                     opt = BFGS(uf, logfile='/dev/null')
#                     opt.run(0.05)
#                     val.set_calculator(None)
#                 cells[j].append(val.get_cell())
#         if not silent:
#             print('opts done')
#
#         for j, p in enumerate(self.pots):
#             copy_structs = deepcopy(structs)
#             for ct, val in enumerate(copy_structs):
#                 for i in np.linspace(0.95, 1.05, n):
#                     val.set_cell(i*cells[j][ct])
#                     val.set_calculator(p)
#                     E[j][ct].append(val.get_potential_energy()/len(val))
#                     if j == 0:
#                         V[ct].append(val.get_volume()/len(val))
#             if not silent:
#                 print('pot {0} done'.format(p.name))
#
#         dat = {'Structure':structs, 'Volumes':V}
#         for i, val in enumerate(self.pots):
#             dat.update({val.name:E[i]})
#         self.df = DataFrame(dat, index=labels)
#
#     def add_config(self, config, label, opt=False, n=10):
#
#         config = deepcopy(config)
#         if opt:
#             config.set_calculator(self.pots[0])
#             uf = UnitCellFilter(config)
#             opt = BFGS(uf, logfile='dump')
#             opt.run(0.05)
#             config.set_calculator(None)
#         cell = config.get_cell()
#
#         E = [[] for i in self.pots]; V = []
#         for j, p in enumerate(self.pots):
#             for i in np.linspace(0.95, 1.05, n):
#                 config.set_cell(i*cell)
#                 config.set_calculator(p)
#                 E[j].append(config.get_potential_energy()/len(config))
#                 if j == 0:
#                     V.append(config.get_volume()/len(config))
#             print('pot {0} done'.format(p.name))
#
#         dat = {'Structure':[config], 'Volumes':[V]}
#         for i, val in enumerate(self.pots):
#             dat.update({val.name:[E[i]]})
#         tdf = DataFrame(dat, index=[label])
#         self.df = self.df.append(tdf)
#
#     def add_config_by_mpid(self):
#         return
#
#     def calc_HP(self, ps, pf, pn, opt_freq=4, subset=slice(0, 8, None)):
#         p = np.linspace(ps, pf, pn)
#         pea = p*0.05/8
#         copy_structs = deepcopy(self.df['Structure'][subset])
#         # consider initializing below as NaNs for pandas
#         v = [[np.zeros(pn) for i in range(len(self.df))] for j in range(len(self.pots))]
#         H = [[np.zeros(pn) for i in range(len(self.df))] for j in range(len(self.pots))]
#         for j, pot in enumerate(self.pots):
#             for ct, val in enumerate(copy_structs):
#                 val.set_calculator(pot)
#                 for i in range(len(p)):
#                     if i % opt_freq == 0:
#                         #val.set_constraint(FixAtoms(mask=[True for atom in val]))
#                         #uf = ExpCellFilter(val, scalar_pressure=pea[i], hydrostatic_strain=True)
#                         uf = ExpCellFilter(val, scalar_pressure=pea[i], hydrostatic_strain=True)
#                         opt = BFGS(uf, logfile='/dev/null')
#                         opt.run(0.05, steps=20)
#                     v[j][ct][i] = (vt := val.get_volume()/len(val))
#                     H[j][ct][i] = val.get_potential_energy()/len(val) + pea[i]*vt
#                 print('struct {} done'.format(self.df.index[ct]))
#
#             print('\npot {0} done\n'.format(pot.name))
#             self.df['H(P) {}'.format(pot.name)] = H[j]
#         self.df['P'] = [p for i in range(len(self.df))]
#
#         return
#
#     def save(self, outfile):
#         for i in flatten(self.df['Structure']):
#             i.calc = None
#         f = open(outfile, 'wb')
#         pickle.dump(self.df, f)
#         f.close()
#
#     def load(self, infile):
#         f = open(infile, 'rb')
#         self.df = pickle.load(f)
#
#
#     def add_pot(self):
#         return
#
#     def plot_EV(self):
#         return



class Cmca_factory(orthorhombic.BaseCenteredOrthorhombicFactory):
    xtal_name='Cmca'
    bravais_basis=[[0.717993, 0.282007, 0.5     ],
                   [0.217993, 0.782007, 0.      ],
                   [0.282007, 0.717993, 0.5     ],
                   [0.782007, 0.217993, 0.      ],
                   [0.331168, 0.331168, 0.829096],
                   [0.668832, 0.668832, 0.170904],
                   [0.168832, 0.168832, 0.329096],
                   [0.831168, 0.831168, 0.670904]]

class bSn_factory(tetragonal.CenteredTetragonalFactory):
    xtal_name='bSn'
    bravais_basis=[[0, 0.5, 0.25], [0.0, 0.0, 0.5]]

class Imma_factory(orthorhombic.BodyCenteredOrthorhombicFactory):
    xtal_name='Imma'
    bravais_basis=[[0.0641, 0.0000, 0.7500], [0.4359, 0.5000, 0.7500]]

class CrystTest:
    '''Centre for the analysis of crystal structure minima using ASE calculators

    Data is returned in a pandas DataFrame, indexed by the crystal structures
                        Default V   V           E           V           E
                Atoms   1st pot     1st pot     1st pot     2nd pot     2nd pot
    --------------------------------------------------------------------------
    fcc     .
    bcc     .
    diamond .
    .       .
    .       .
    .       .
    TODO:
        Automatic scaling of lattice parameter initial guesses using covalent radii
        Ability to add custom crystal structures
        Better error handling for unconverged geometries

        '''

    def __init__(self, pots, element,
                 extra_configs=[], extra_labels=[], n=10, opt=True, silent=False,
                 load=False, start_cells = {}, steps=100):
        '''pots: list of ase calculator objects (currently GAPs and MTPs). Strings will be
                    interpreted as GAP param_filenames
        element: element used for the pots (multicomponent systems not yet supported)
        extra_configs/labels: custom crystal structures for evaluation (see add_config method)
        n: number of V points calculated per crystal structure
        opt: bool, optimize the crystal structures? If a particular calculator is unstable, add this using
            add_pot()
        silent: bool, silence output

        '''
        if not isinstance(pots, list):
            self.pots = [pots]
        else:
            self.pots = pots
        if load: # awaiting implementation
            raise NotImplementedError('Hang tight, coming soon')

        for i in range(len(pots)):
            if not hasattr(pots[i], 'calculate'):
                # assume is the filename (str) of a GAP from typical analysis
                pots[i] = Potential(param_filename=pots[i])

        # Need a better way of guessing the initial latice parameters - e.g. scaling by atomic radius
        # based on True Si data
        e_r = covalent_radii[atomic_numbers[element]]
        Si_r = covalent_radii[14]

        bSn_fact = bSn_factory()
        bSn = Atoms(bSn_fact(symbol=element, latticeconstant={'a':5.2, 'c':2.87}))

        Imma_fact = Imma_factory()
        Imma = Imma_fact(symbol=element, latticeconstant={'a':2.87, 'b':4.99, 'c':5.30})

        structs = [build.bulk(element, crystalstructure='fcc', a=3.0, cubic=True),
                   build.bulk(element, crystalstructure='diamond', a=5.0,  cubic=True),
                   build.bulk(element, crystalstructure='hcp', a=3, c=5),
                   build.bulk(element, crystalstructure='bcc', a=3.0, cubic=True),
                   build.bulk(element, crystalstructure='sc', a=3.0),
                   Atoms(hexagonal.Hexagonal(symbol=element, latticeconstant={'a':3.0, 'c':3.0})),
                   bSn,
                   Imma] + \
                  [deepcopy(i) for i in extra_configs]

        self.labels = ['fcc', 'dia', 'hcp', 'bcc', 'sc', 'sh', 'bSn', 'Imma'] + extra_labels
        self.opt_structs = []

        # Optimise the unit cell vectors if required
        cells = [[] for i in range(len(pots))]
        for j, pot in enumerate(self.pots):
            copy_structs = deepcopy(structs)
            for ct, val in enumerate(copy_structs):
                if opt:
                    val.set_calculator(pot)
                    uf = StrainFilter(val) # should ensure only the lattice can move, not atomic positions
                    opt = BFGS(uf, logfile='/dev/null')
                    opt.run(0.05, steps=steps)
                    if not opt.converged():
                        warnings.warn(('Warning: pot \'{}\' failed to converge on structure {}\n' + \
                                       'in {} steps').format(pot.name, val, steps))
                        val.set_cell(np.NaN*np.ones(3)) # flag unconverged geometry
                    val.set_calculator(None)
                cells[j].append(val.get_cell())
                if j == 0:
                    self.opt_structs.append(val)
        if not silent:
            print('opts done')

        # Evaluate the E/V curves on the optimised cells
        E = [[[] for j in range(len(structs))] for i in range(len(pots))]
        V = [[[] for j in range(len(structs))] for i in range(len(pots))]
        for j, pot in enumerate(self.pots):
            copy_structs = deepcopy(structs)
            for ct, val in enumerate(copy_structs):
                for i in np.linspace(0.95, 1.05, n):
                    val.set_cell(i*cells[j][ct], scale_atoms=True)
                    val.set_calculator(pot)
                    E[j][ct].append(val.get_potential_energy()/len(val))
                    V[j][ct].append(val.get_volume()/len(val))

            if not silent:
                print('pot {0} done'.format(pot.name))

        # Construct a DataFrame with the data
        dat = {'Structure':structs, 'Volumes':V[0]}
        for i, val in enumerate(self.pots):
            dat.update({val.name +  '_V' : V[i], val.name + '_E' : E[i]})
        print(dat)
        self.df = pd.DataFrame(dat, index=self.labels)

    def add_config(self, config, label, opt=False, n=10):

        config = deepcopy(config)
        if opt:
            config.set_calculator(self.pots[0])
            uf = UnitCellFilter(config)
            opt = BFGS(uf, logfile='dump')
            opt.run(0.05)
            config.set_calculator(None)
        cell = config.get_cell()

        E = [[] for i in self.pots]; V = []
        for j, p in enumerate(self.pots):
            for i in np.linspace(0.95, 1.05, n):
                config.set_cell(i*cell)
                config.set_calculator(p)
                E[j].append(config.get_potential_energy()/len(config))
                if j == 0:
                    V.append(config.get_volume()/len(config))
            print('pot {0} done'.format(p.name))

        dat = {'Structure':[config], 'Volumes':[V]}
        for i, val in enumerate(self.pots):
            dat.update({val.name:[E[i]]})
        tdf = pd.DataFrame(dat, index=[label])
        self.df = self.df.append(tdf)

    def add_config_by_mpid(self):
        return

    def calc_HP(self, ps, pf, pn, opt_freq=4, subset=slice(0, 8, None)):
        '''Calculates the Enthalpy/Pressure curves for the structures.
        ps: float Starting pressure / GPa
        pf: float Final pressure
        pn: int number of pressure points
        opt_freq: how often is the geom reoptimised w.r.t. pressure points
        subset: slice which crystal structures to include'''


        p = np.linspace(ps, pf, pn)
        pea = p*0.05/8
        copy_structs = deepcopy(self.df['Structure'][subset])
        # consider initializing below as NaNs for pandas
        v = [[np.zeros(pn) for i in range(len(self.df))] for j in range(len(self.pots))]
        H = [[np.zeros(pn) for i in range(len(self.df))] for j in range(len(self.pots))]
        for j, pot in enumerate(self.pots):
            for ct, val in enumerate(copy_structs):
                val.set_calculator(pot)
                for i in range(len(p)):
                    if i % opt_freq == 0:
                        #val.set_constraint(FixAtoms(mask=[True for atom in val]))
                        #uf = ExpCellFilter(val, scalar_pressure=pea[i], hydrostatic_strain=True)
                        uf = ExpCellFilter(val, scalar_pressure=pea[i], hydrostatic_strain=True)
                        opt = BFGS(uf, logfile='/dev/null')
                        opt.run(0.05, steps=20)
                        if not opt.converged():
                            warnings.warn('Warning: {} did not converge at pressure {}'.format(
                                self.labels[ct],p[i]))
                    v[j][ct][i] = (vt := val.get_volume()/len(val))
                    H[j][ct][i] = val.get_potential_energy()/len(val) + pea[i]*vt
                print('struct {} done'.format(self.df.index[ct]))

            print('\npot {0} done\n'.format(pot.name))
            self.df['H(P) {}'.format(pot.name)] = H[j]
        self.df['P'] = [p for i in range(len(self.df))]

        return

    def save(self, outfile):
        for i in flatten(self.df['Structure']):
            i.calc = None
        f = open(outfile, 'wb')
        pickle.dump(self.df, f)
        f.close()

    def load(self, infile):
        f = open(infile, 'rb')
        self.df = pickle.load(f)


    def add_pot(self, pot, opt=True, steps=100, n=10):

        if not hasattr(pot, 'calculate'):
            # assume is the filename (str) of a GAP from typical analysis
            print('Interpreting {} as a GAP'.format(pot))
            pot = Potential(param_filename=pot)
            pot.name = os.path.splitext(os.path.basename(pot))[0]

        self.pots.append(pot)
        cells = []

        if opt:
            copy_structs = deepcopy(self.df['Structure'])
            for ct, val in enumerate(copy_structs):
                if opt and ct < 8:
                    val.set_calculator(pot)
                    uf = StrainFilter(val)
                    opt = BFGS(uf, logfile='/dev/null')
                    opt.run(0.05, steps=steps)
                    if not opt.converged():
                        warnings.warn(('Warning: pot \'{}\' failed to converge on structure {}\n' + \
                                       'in {} steps').format(pot.name, val, steps))
                    val.set_calculator(None)
                cells.append(val.get_cell())
        else:
            cells = [i.get_cell() for i in self.opt_structs]

        E = [[] for j in range(len(self.df['Structure']))]
        V = [[] for j in range(len(self.df['Structure']))]
        copy_structs = deepcopy(self.df['Structure'])
        for ct, val in enumerate(copy_structs):
            val.set_calculator(pot)
            for i in np.linspace(0.95, 1.05, n):
                val.set_cell(i*cells[ct], scale_atoms=True)
                E[ct].append(val.get_potential_energy()/len(val))
                V[ct].append(val.get_volume()/len(val))

        self.df = self.df.join(pd.DataFrame({pot.name+'_V':V, pot.name+'_E':E}, index=self.labels))



    def plot_EV(self):
        return


def kernel_compare(cfgs, comp,
                         desc=None,
                         zeta=4, similarity=False, average=True):
    '''calculates the average/std dev similarity kernel between a set of
    configs and a reference (or set of references).
    '''
    if desc is None:
        desc_str = 'soap l_max=6 n_max=12 \
                   atom_sigma=0.5 cutoff=5.0 \
                   cutoff_transition_width=1.0 central_weight=1.0'
    else:
        desc_str = desc

    if ' average=' not in desc_str:
        if average:
            desc_str += ' average=T'
        else:
            desc_str += ' average=F'

    if not isinstance(comp, list):
        comp = [comp]
    
    desc = Descriptor(desc_str)
    descs = np.array(desc.calc_descriptor(cfgs))

    comp_desc = Descriptor(desc_str.replace(' average=F', ' average=T'))
    comp_descs = np.array(comp_desc.calc_descriptor(comp)) # currently averages over comparison atoms
    print(descs.shape, comp_descs.shape)
    if similarity:
        k = np.einsum('i...j,k...j', descs, comp_descs)**zeta
    else:
        k = np.array(2 - 2*np.einsum('i...j,k...j', descs, comp_descs)**zeta)


    return k.T.squeeze()

def print_DB_stats(atoms, by_config_type=True):
    print('Size statistics:\n'+'-'*36)
    if isinstance(atoms[0], list):
        atoms = flatten(atoms)
    hist = [len(i) for i in atoms]
    tot = sum(hist)
    sizes, freq = np.unique(hist, return_counts=True)
    print(('{:<12s}'*3).format('size', 'freq', 'percentage'))
    for j in range(len(sizes)):
        print('{:<12d}{:<12d}{:<11.1f}'.format(sizes[j], freq[j], 100*sizes[j]*freq[j]/tot))

    if by_config_type:
        labels = []
        catoms = []
        for i in atoms:
            if 'config_type' in i.info.keys():
                if (l := i.info['config_type']) not in labels:
                    labels.append(l)
                    catoms.append([])
                catoms[labels.index(l)].append(i)
        print('\nBy config types:\n'+'-'*36)
        for i, val in enumerate(catoms):
            hist = [len(j) for j in val]
            tot = sum(hist)
            sizes, freq = np.unique(hist, return_counts=True)
            print('{:<16s} {} atoms'.format(val[0].info['config_type'], tot))
            for j in range(len(sizes)):
                print('{:<12d}{:<12d}{:<11.1f}'.format(sizes[j], freq[j], 100*sizes[j]*freq[j]/tot))
            print('-'*36+'\n')
    return


def pad_rstats(rs):
    max = 0
    for i in rs:
        if max < len(i):
            max = len(i)
    max += 2

    for ct, i in enumerate(rs):
        if len(i) < max:
            rs[ct] = np.pad(i, (0, max - len(i)))
    return np.array(rs)
