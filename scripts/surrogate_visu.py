"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.pickle_utils import load
from rpfm.plots.response_surfaces import RespSurf

path = '/home/alberto/Nextcloud/HOmeBOX/Documents/surrogate/'

fid = path + 'sm_var1.pkl'
sm = load(fid)
print(len(sm.isp))

train = True
nb_eval = 2500

if train:
    sm.train('QP')  # train_method must be one between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC
    sm.evaluate(nb_eval=nb_eval)

sm.plot()

"""
fid_m = path + 'sm_maximin.pkl'  # maximin
fid_ese = path + 'sm_ese.pkl'  # ESE
fid_full = path + 'sm_full.pkl'  # Full Factorial

train = False
nb_eval = 900

sm_m = load(fid_m)
sm_ese = load(fid_ese)
sm_full = load(fid_full)

if train:
    for s in [sm_m, sm_ese]:
        s.train('QP')  # train_method must be one between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC
        s.evaluate(nb_eval=nb_eval)

for s in [sm_m, sm_ese, sm_full]:
    s.plot()

# error
err_m_m = np.fabs(sm_m.m_mat - sm_full.m_mat)/sm_full.m_mat*100
err_ese_m = np.fabs(sm_ese.m_mat - sm_full.m_mat)/sm_full.m_mat*100
err_m_tof = np.fabs(sm_m.tof_mat - sm_full.tof_mat)/sm_full.tof_mat*100
err_ese_tof = np.fabs(sm_ese.tof_mat - sm_full.tof_mat)/sm_full.tof_mat*100

rs_m = RespSurf(sm_m.isp, sm_m.twr, err_m_m, err_m_tof)
rs_ese = RespSurf(sm_ese.isp, sm_ese.twr, err_ese_m, err_ese_tof)

rs_m.plot()
rs_ese.plot()
"""
