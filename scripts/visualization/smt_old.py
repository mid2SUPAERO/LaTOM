"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.utils.pickle_utils import load, save
from latom.data.smt.data_smt import dirname_smt

# choose the kind of simulation between the followings:
# asc_const: ascent trajectory with constant thrust
# asc_var: ascent trajectory with variable thrust
# asc_vtoff: ascent trajectory with variable thrust and constraint on minimum safe altitude (not available for full)
# desc_const: descent trajectory with constant thrust
# desc_var: descent trajectory with variable thrust
# desc_vland: descent trajectory with variable thrust and constraint on minimum safe altitude (not available for full)

kind_lhs = 'asc_const'  # kind of transfer with latin hypercube sampling
kind_full = 'asc_const'  # kind of transfer with full grid sampling

fid_lhs = ''.join([dirname_smt, '/smt_old/', kind_lhs, '_lhs.pkl'])
fid_full = ''.join([dirname_smt, '/smt_old/', kind_full, '_full.pkl'])

train = True  # train the surrogate model (to be done before the first iteration)
plot = True
store = False
nb_eval = 2500  # number of evaluation point (must be a perfect square)

sm_lhs = load(fid_lhs)  # object for full grid sampling
sm_full = load(fid_full)  # object for latin hypercube sampling

if train:
    for s in [sm_lhs]:
        s.train('KRG')  # train_method must be one between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC
        s.evaluate(nb_eval=nb_eval)  # evaluate the model on nb_eval points

if plot:
    for s in [sm_lhs, sm_full]:
        s.plot()  # plot the contour plots

# retrieve the final mass and time of flight for a specific (Isp, twr)
isp = 450.  # Isp [s]
twr = 2.  # thrust/weight ratio [-]

m_final, tof = sm_lhs.evaluate(isp=isp, twr=twr)

print('final mass:', m_final[0, 0], 'kg')
print('time of flight:', tof[0, 0], 's')

d_lhs = {'limits': sm_lhs.limits, 'x_samp': sm_lhs.x_samp, 'm_prop': (1.0 - sm_lhs.m_samp),
         'failures': np.zeros(np.shape(sm_lhs.m_samp))}
d_full = {'limits': sm_full.limits, 'x_samp': sm_full.x_samp, 'm_prop': (1.0 - sm_full.m_samp),
          'failures': np.zeros(np.shape(sm_full.m_samp))}

if store:
    fid_lhs_new = ''.join([dirname_smt, '/', kind_lhs, '_lhs.pkl'])
    fid_full_new = ''.join([dirname_smt, '/', kind_full, '_full.pkl'])
    save(d_lhs, fid_lhs_new)
    save(d_full, fid_full_new)
