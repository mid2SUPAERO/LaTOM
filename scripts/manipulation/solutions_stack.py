import numpy as np

from rpfm.utils.pickle_utils import load, save
from rpfm.data.metamodels.data_mm import dirname_metamodels
from rpfm.plots.response_surfaces import RespSurf

plot = 'mp'  # 'mp' to plot the propellant fraction, 'en' to plot the spacecraft energy
fid = None  # 'llo2apo_lin_19_12_390-500.pkl'  # filename for adjusted solution
direction = 1  # stack Isp (columns, 1) or thrust/weight ratios (rows, 0)

# load raw solutions
fid1 = '/'.join([dirname_metamodels, 'llo2apo_lin_19_6_390-440.pkl'])
fid2 = '/'.join([dirname_metamodels, 'tests/llo2apo_mm_lin450-500.pkl'])
d1 = load(fid1)
d2 = load(fid2)

# stack solutions
if np.isclose(direction, 0):  # same Isp bounds and step

    m_prop = np.vstack((d1['m_prop'], d2['m_prop']))
    energy = np.vstack((d1['energy'], d2['energy']))
    failures = np.vstack((d1['failures'], d2['failures']))

    twr = np.hstack((d1['twr'], d2['twr']))
    isp = d1['Isp']

elif np.isclose(direction, 1):  # same thrust/weight ratio bounds and step

    m_prop = np.hstack((d1['m_prop'], d2['m_prop']))
    energy = np.hstack((d1['energy'], d2['energy']))
    failures = np.hstack((d1['failures'], d2['failures']))

    twr = d1['twr']
    isp = np.hstack((d1['Isp'], d2['Isp']))

else:
    raise ValueError('matrices not compatible')

print(f"\nIsp: {isp[0]:.4f}s - {isp[-1]:.4f}s at {(isp[1] - isp[0]):.4f}s step")
print(f"twr: {twr[0]:.4f} - {twr[-1]:.4f} at {(twr[1] - twr[0]):.4f} step")
print(f"Max spacecraft energy: {np.max(energy):.4f} m^2/s^2\n")

if plot == 'mp':
    rs = RespSurf(isp, twr, m_prop.T, 'propellant fraction', nb_lines=50)
    rs.plot()
elif plot == 'en':
    rs = RespSurf(isp, twr, energy.T, 'spacecraft energy', nb_lines=50)
    rs.plot()

if fid is not None:
    d = {'Isp': isp, 'twr': twr, 'm_prop': m_prop, 'energy': energy, 'failures': failures}
    fid = '/'.join([dirname_metamodels, fid])
    save(d, fid)
