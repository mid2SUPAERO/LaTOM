import numpy as np

from rpfm.utils.pickle_utils import load
from rpfm.data.data import dirname
from rpfm.plots.response_surfaces import RespSurf

fid = '/'.join([dirname, 'metamodels', 'llo2apo_mm_lin80.pkl'])
d = load(fid)

isp = np.reshape(d['Isp'], (1, np.size(d['Isp'])))
twr = np.reshape(d['twr'], (np.size(d['twr']), 1))
en = d['energy']

isp2 = np.ones(np.shape(en))*isp
twr2 = np.ones(np.shape(en))*twr

twr_fail = twr2[en > -7e4]
isp_fail = isp2[en > -7e4]
