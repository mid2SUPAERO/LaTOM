import numpy as np

from latom.utils.pickle_utils import load, save
from latom.data.metamodels.data_mm import dirname_metamodels
from latom.plots.response_surfaces import RespSurf

plot = 'mp'  # 'mp' to plot the propellant fraction, 'en' to plot the spacecraft energy
fid = None  # 'llo2apo_lin_19_12_390-500.pkl'  # filename for adjusted solution
direction = 1  # stack Isp (columns, 1) or thrust/weight ratios (rows, 0)

# load raw solutions
fid1 = '/'.join([dirname_metamodels, 'tests/llo2apo_mm_lin159.pkl'])
fid2 = '/'.join([dirname_metamodels, 'llo2apo_lin_19_12_390-500.pkl'])
d1 = load(fid1)
d2 = load(fid2)

# check superposition
isp1 = d1['Isp'][-12:]
twr1 = d1['twr'][:19]
m1 = d1['m_prop'][:19, -12:]
e1 = d1['energy'][:19, -12:]
print(np.isclose(m1, d2['m_prop'], rtol=1e-2, atol=1e-2))
print(np.isclose(e1, d2['energy'], rtol=1e-2, atol=1e-2))

m_prop = d1['m_prop']
energy = d1['energy']
m_prop[:19, -12:] = d2['m_prop']
energy[:19, -12:] = d2['energy']

isp = d1['Isp']
twr = d1['twr'][2:]
m_prop = m_prop[2:, :]
energy = energy[2:, :]
failures = np.zeros(np.shape(energy))

rs_m = RespSurf(isp, twr, m_prop.T, 'propellant fraction', nb_lines=50)
rs_m.plot()
rs_en = RespSurf(isp, twr, energy.T, 'spacecraft energy', nb_lines=50)
rs_en.plot()

d = {'Isp': isp, 'twr': twr, 'm_prop': m_prop, 'energy': energy, 'failures': failures}
# save(d, '/'.join([dirname_metamodels, 'test.pkl']))



