import numpy as np

from rpfm.utils.pickle_utils import load, save
from rpfm.data.continuation.data_continuation import dirname_continuation
from rpfm.data.metamodels.data_mm import dirname_metamodels
from rpfm.plots.response_surfaces import RespSurf

fid = '/'.join([dirname_metamodels, 'tests/llo2apo_mm_lin80.pkl'])
d = load(fid)

t = load('/'.join([dirname_continuation, 'tests/test_lin260.pkl']))

isp = np.reshape(d['Isp'], (1, np.size(d['Isp'])))
twr = np.reshape(d['twr'], (np.size(d['twr']), 1))
en = d['energy']

isp_mat = np.ones(np.shape(en))*isp
twr_mat = np.ones(np.shape(en))*twr

twr_fail = twr_mat[en > -7e4]
isp_fail = isp_mat[en > -7e4]

# new
m_prop = d['m_prop'][1:, :]
energy = d['energy'][1:, :]
failures = d['failures'][1:, :]
isp = d['Isp']
twr = d['twr'][1:]

m_new = np.flip(t.m_prop_list)[4]
twr_new = np.flip(t.twr_list)[4]
en_new = np.flip(t.energy_list)[4]

m_prop[1, 1] = m_new
energy[1, 1] = en_new

rs = RespSurf(isp, twr, energy.T, 'propellant fraction', nb_lines=50)
rs.plot()

d = {'Isp': isp, 'twr': twr, 'm_prop': m_prop, 'energy': energy, 'failures': failures}
# fid = '/'.join([dirname, 'metamodels', 'llo2apo_lin_79_26.pkl'])
# save(d, fid)
