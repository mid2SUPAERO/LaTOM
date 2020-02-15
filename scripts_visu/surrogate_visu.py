"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.utils.pickle_utils import load
from rpfm.data.data import dirname

path = dirname  # directory where the data are stored

# choose the kind of simulation between the followings:
# asc_const: ascent trajectory with constant thrust
# asc_var: ascent trajectory with variable thrust
# asc_vtoff: ascent trajectory with variable thrust and constraint on minimum safe altitude (not available for full)
# desc_const: descent trajectory with constant thrust
# desc_var: descent trajectory with variable thrust
# desc_vland: descent trajectory with variable thrust and constraint on minimum safe altitude (not available for full)

kind_lhs = 'desc_var'  # kind of transfer with full grid sampling
kind_full = 'desc_var'  # kind of transfer with latin hypercube sampling

fid_lhs = ''.join([path, '/smt/', kind_lhs, '_lhs.pkl'])
fid_full = ''.join([path, '/smt/', kind_full, '_full.pkl'])

train = True  # train the surrogate model (to be done before the first iteration)
plot = True
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
