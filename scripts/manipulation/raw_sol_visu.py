import numpy as np

from latom.utils.pickle_utils import load, save
from latom.data.continuation.data_continuation import dirname_continuation
from latom.data.metamodels.data_mm import dirname_metamodels
from latom.plots.response_surfaces import RespSurf

plot = 'mp'  # 'mp' to plot the propellant fraction, 'en' to plot the spacecraft energy
fid = None  # 'llo2apo_lin_005-05_250-330.pkl'  # filename for adjusted solution

en_threshold = -6e4  # spacecraft energy threshold
isp_lim = (0, None)  # new specific impulse limits
twr_lim = (0, None)  # new thrust/weight ratio limits

# load raw solution
fid_raw = '/'.join([dirname_metamodels, 'tests/llo2apo_mm_lin360-370.pkl'])
d = load(fid_raw)

# extract data from dictionary
isp_raw = np.reshape(d['Isp'], (1, np.size(d['Isp'])))
twr_raw = np.reshape(d['twr'], (np.size(d['twr']), 1))
m_prop_raw = d['m_prop']
energy_raw = d['energy']

# Isp and thrust/weight ratio matrices
isp_mat = np.ones(np.shape(energy_raw))*isp_raw
twr_mat = np.ones(np.shape(energy_raw))*twr_raw

# Isp and thrust/weight ratios for which the solution degenerated into a parabola
twr_fail = twr_mat[energy_raw > en_threshold]
isp_fail = isp_mat[energy_raw > -en_threshold]

# extract adjusted solution
m_prop = m_prop_raw[twr_lim[0]:twr_lim[1], isp_lim[0]:isp_lim[1]]
energy = energy_raw[twr_lim[0]:twr_lim[1], isp_lim[0]:isp_lim[1]]
failures = d['failures'][twr_lim[0]:twr_lim[1], isp_lim[0]:isp_lim[1]]
isp = isp_raw[:, isp_lim[0]:isp_lim[1]].flatten()
twr = twr_raw[twr_lim[0]:twr_lim[1], :].flatten()
delta_isp = np.max(isp[1:] - isp[:-1])
delta_twr = np.max(twr[1:] - twr[:-1])

print(f"\nIsp: {isp[0]:.4f}s - {isp[-1]:.4f}s at {delta_isp:.4f}s step")
print(f"twr: {twr[0]:.4f} - {twr[-1]:.4f} at {delta_twr:.4f} step")
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
