"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.surrogate.surrogate_2d import TwoDimAscConstSurrogate, TwoDimAscVarSurrogate, TwoDimAscVToffSurrogate,\
    TwoDimDescVertSurrogate
from rpfm.utils.primary import Moon

# trajectory
kind = 'c'
moon = Moon()
alt = 86.87e3  # final orbit altitude [m]
theta = np.pi/2  # guessed spawn angle [rad]
tof_asc = 500  # guessed time of flight (ascent) [s]
tof_desc = (1000, 100)  # guessed time of flight (descent) [s]
t_bounds = None  # time of flight bounds [-]
alt_p = 15e3  # perigee altitude (descent only) [m]
alt_switch = 3e3  # switch altitude (descent only) [m]
alt_safe = 5e3  # minimum safe altitude (ascent safe only) [m]
slope = 10.  # slope of the constraint on minimum safe altitude (ascent safe only) [-]
isp_lim = (300., 450.)
twr_lim = (1.1, 3.)

# NLP
method = 'gauss-lobatto'
segments_asc = 10
segments_desc = (10, 10)
order = 3
solver = 'SNOPT'

# surrogate model
samp_method = 'full'
train_method = 'RMTB'
nb_samp = 100
nb_eval = 100
train = True

if kind == 'c':
    sm = TwoDimAscConstSurrogate(moon, isp_lim, twr_lim, alt, theta, tof_asc, t_bounds, method, segments_asc, order,
                                 solver, nb_samp, samp_method=samp_method, nb_eval=nb_eval)
elif kind == 'v':
    sm = TwoDimAscVarSurrogate(moon, isp_lim, twr_lim, alt, t_bounds, method, segments_asc, order, solver, nb_samp,
                               samp_method=samp_method, nb_eval=nb_eval)
elif kind == 's':
    sm = TwoDimAscVToffSurrogate(moon, isp_lim, twr_lim, alt, alt_safe, slope, t_bounds, method, segments_asc, order,
                                 solver, nb_samp, samp_method=samp_method, nb_eval=nb_eval)
elif kind == 'd':
    sm = TwoDimDescVertSurrogate(moon, isp_lim, twr_lim, alt, alt_p, alt_switch, theta, tof_desc, t_bounds, method,
                                 segments_desc, order, solver, nb_samp, samp_method=samp_method, nb_eval=nb_eval)
else:
    raise ValueError('kind must be c, v, s or d')

sm.sampling()

if train:
    sm.train(train_method)
    sm.evaluate()
    sm.plot()
