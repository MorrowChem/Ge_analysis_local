from ase.io.extxyz import read_xyz, write_xyz

from ase.io.extxyz import read_xyz, write_xyz

def red_DB_cfg(xyz, ctypes = [], outfile=None):
    ats = read_xyz(xyz, index=slice(0,None))
    keeps = []
    ct = 0; ct_kept = 0; popped = []
    for i, val in enumerate(ats):
        if val.info['config_type'] in ctypes or 'isol' in val.info['config_type']:
            keeps.append(val)
            ct_kept += 1
        else:
            ct += 1
            if val.info['config_type'] not in popped:
                popped.append(val.info['config_type'])

    if outfile is None:
        outfile = xyz.split('.')[:-1] + '_red.xyz'

    write_xyz(outfile, keeps)

    print('total {0} popped\n      {1} kept\nconfig_types removed: {2}'.format(ct, ct_kept, popped))

# red_DB_cfg('/data/chem-amais/hert5155/Ge/Potentials/Si_lit_GAP/GAP/gp_iter6_sparse9k.xml.xyz',
#            ctypes=['liq', 'amorph'],
#            outfile='gp_iter6_liqamo.xyz')
#
# red_DB_cfg('/data/chem-amais/hert5155/Ge/Potentials/Si_lit_GAP/GAP/gp_iter6_sparse9k.xml.xyz',
#            ctypes=['liq', 'amorph', 'dia'],
#            outfile='gp_iter6_liq_amo_dia.xyz')
