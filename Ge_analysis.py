'''Set of functions for plotting/analysis potentials and their databases
Plan is to migrate these to methods of the GAP class at some point'''
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import gaussian_kde
import os
import pickle
from quippy.potential import Potential
from quippy.descriptors import Descriptor
from ase import Atoms
from ase.io import write, read
from sklearn import decomposition
from ase.geometry.analysis import Analysis


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


data_dir = '/Users/Moji/Documents/Summer20/Ge/'
'''data_216_125_2000 = data_dir + 'Pickles/data_3b2000_5c_216_125'
data_216_125_4000 = data_dir + 'Pickles/data_3b4000_5c_216_125'
data_64_5000 = data_dir + 'Pickles/data_2bSOAP5000_5c_64'
data_d1 = data_dir + 'Pickles/data_125_216_d1'
data_d2 = data_dir + 'Pickles/data_125_216_d2'
data_d155_f = data_dir + 'Pickles/data_125_216_d155'
train_file = data_dir + 'Structure_databases/train_216_125_64.xyz'
val_file = data_dir + 'Structure_databases/validate_216_125_64.xyz'

gaps = [Potential(param_filename= data_dir + 'Potentials/Ge_3bSOAP_2000/Ge_3bSOAP_2000_5cut.xml'),
        Potential(param_filename= data_dir + 'Potentials/Ge_3bSOAP_4000/Ge_3bSOAP_4000_5cut.xml')]

data_216_125_2000 = read_database(data_216_125_2000)[0]
data_216_125_4000 = read_database(data_216_125_4000)[0]
data_64_5000 = read_database(data_64_5000)[0]
data_d1 = read_database(data_d1)[0]
data_d2 = read_database(data_d2)[0]
data_d155, d155_dict = read_database(data_d155_f)
#datasets = [data_216_125_2000, data_216_125_4000, data_64_5000, data_d155]
datasets = [data_d155]

for j in datasets:
    j[2][1] = [240, 120, 20, 180, 160]
    for i in range(0, len(datasets[0][0])):
        if i > 9:
            pass
        try:
            j[0][i], j[1][i] = sort_by_timestep(j, i)
        except:
            print('data not regular in ', i)

    j[2][1].sort(reverse=True)'''
    

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

# Fit plotting ####################
colormap = plt.get_cmap('plasma')
colors = [colormap(i) for i in np.linspace(0, 0.8, 5)]
labels = ['240 ps', '180 ps', '160 ps', '120 ps', '20 ps']
xs = np.linspace(-6, 6, 100)


def energy_error(GAP, ax=None, title=None, file=None, by_config=True, color='r', rmse=True, label=None):
    mi = np.amin(flatten(GAP.data_dict['QM_E_t'])) - 0.01 - GAP.zero_e
    ma = np.amax(flatten(GAP.data_dict['QM_E_t'])) - 0.01 - GAP.zero_e
    xs = np.linspace(mi, ma, 100)
    if not ax:
        fig, ax = plt.subplots()
    if by_config:
        for i in range(len(GAP.config_labels)):
            ax.scatter(np.array(GAP.data_dict['QM_E_v'][i]) - GAP.zero_e,
                       np.array(GAP.data_dict['GAP_E_v'][i]) - GAP.zero_e,
                       marker='.', color=colors[i], label=GAP.config_labels[i])
    else:
        ax.scatter(np.array(flatten(GAP.data_dict['QM_E_v'])) - GAP.zero_e,
                   np.array(flatten(GAP.data_dict['GAP_E_v'])) - GAP.zero_e,
                   marker='.', color=color, label=label)
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
        ax.scatter(np.array(flatten(GAP.data_dict['QM_F_v'])),
                   np.array(flatten(GAP.data_dict['GAP_F_v'])),
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

    for i in range(len(GAP.config_labels)):
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

    x = [np.array(flatten(GAP.data_dict['QM_E_v'])) - GAP.zero_e, np.array(flatten(GAP.data_dict['QM_F_v']))]
    y = [abs(np.array(flatten(GAP.data_dict['E_err_v']))*1000), abs(np.array(flatten(GAP.data_dict['F_err_v'])))]
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

