"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.pickle_utils import load
from scipy.io import savemat

path = '/home/alberto/Nextcloud/Documents/ResearchProject/surrogate/'
kind = 'desc_const'

fid_lhs = path + kind + '_lhs.pkl'
fid_full = path + kind + '_full.pkl'

train = True
nb_eval = 2500

sm_lhs = load(fid_lhs)
sm_full = load(fid_full)

if train:
    for s in [sm_lhs]:
        s.train('KRG')  # train_method must be one between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC
        s.evaluate(nb_eval=nb_eval)

for s in [sm_lhs, sm_full]:
    s.plot()

d = {'isp': sm_full.isp, 'twr': sm_full.twr, 'm': sm_full.m_mat, 'tof': sm_full.tof_mat}
savemat(kind + '.mat', d)
