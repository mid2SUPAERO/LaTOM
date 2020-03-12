import numpy as np

from rpfm.utils.pickle_utils import load
from rpfm.data.data import dirname
from rpfm.surrogate.meta_models_llo2heo import TwoDimLLO2ApoContinuationMetaModel
from rpfm.plots.continuation import MassEnergyContinuation

isp = np.linspace(250., 500., 51)
isp2 = np.linspace(250., 500., 26)
twr_log = np.linspace(-3, 1.6, 93)
twr = np.exp(twr_log)
delta_log = np.hstack(([0.], (twr_log[1:] - twr_log[:-1])))
delta = np.hstack(([0.], (twr[1:] - twr[:-1])))

print(np.size(twr), '\n')

for i in range(np.size(twr)):
    print(f"{twr_log[i]:6.4f}\t\t{delta_log[i]:6.4f}\t\t{twr[i]:6.4f}\t\t{delta[i]:6.4f}")

"""
d = load(dirname + '/metamodels/llo2apo_mm_log10.pkl')
rows, cols = np.shape(d['m_prop'])
twr = d['twr']
isp = d['Isp']
energy = d['energy']
m_prop = d['m_prop']

for i in range(cols):
    # p = MassEnergyContinuation(twr, m_prop[:, i], energy[:, i])
    # p.plot()
    max_en = np.max(energy[:, i])
    print(f"{isp[i]:f}\t\t{max_en:f}")
"""