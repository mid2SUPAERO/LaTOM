import numpy as np

from rpfm.utils.pickle_utils import load
from rpfm.data.data import dirname
from rpfm.surrogate.meta_models_llo2heo import TwoDimLLO2ApoContinuationMetaModel

isp = np.linspace(250., 495., 50)
twr_log = np.linspace(-3, 1.9, 50)
twr = np.exp(twr_log)
delta = np.hstack(([0.], (twr[1:] - twr[:-1])))

print(np.size(twr), '\n')
print(np.around(twr, 4), '\n')
print(np.around(delta, 4), '\n')
print(np.around(twr_log, 4), '\n')

"""
for i in range(np.size(twr)):
    print(f"\t{twr[i]:.4f}\t{delta[i]:.4f}\t{twr_log[i]:.4f}")
"""

"""
mm = TwoDimLLO2ApoContinuationMetaModel(rec_file='test.pkl')

r1 = load('/'.join([dirname, 'continuation', 'test_log400.pkl']))
r2 = load('/'.join([dirname, 'continuation', 'test_log450.pkl']))

dm1 = mm.m_prop[:, 0] - np.flip(r1.m_prop_list)
dm2 = mm.m_prop[:, -1] - np.flip(r2.m_prop_list)

den1 = mm.energy[:, 0] - np.flip(r1.energy_list)
den2 = mm.energy[:, -1] - np.flip(r2.energy_list)

for d in [dm1, dm2, den1, den2]:
    print(np.max(np.fabs(d)))
"""