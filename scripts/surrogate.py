"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.surrogate.surrogate_2d import TwoDimAscConstSurrogate, TwoDimAscVarSurrogate, TwoDimAscVToffSurrogate,\
    TwoDimDescVertSurrogate, TwoDimDescConstSurrogate, TwoDimDescVarSurrogate, TwoDimDescVLandSurrogate

from rpfm.utils.primary import Moon

# trajectory
kind = 'dv'
moon = Moon()
alt = 100e3  # final orbit altitude [m]
theta = np.pi/2  # guessed spawn angle [rad]
tof = 1000  # guessed time of flight [s]
tof_desc_2p = (1000, 100)  # guessed time of flight (descent with 2 phases) [s]
t_bounds = None  # time of flight bounds [-]
alt_p = 15e3  # perigee altitude (descent 2 phases only) [m]
alt_switch = 3e3  # switch altitude (descent 2 phases only) [m]
alt_safe = 5e3  # minimum safe altitude (ascent and descent safe only) [m]
slope = 10.  # slope of the constraint on minimum safe altitude (ascent and descent safe only) [-]
isp_lim = (250., 500.)  # specific impulse lower and upper limits [s]
twr_lim = (1.1, 4.)  # initial thrust/weight ratio lower and upper limits [-]

# NLP
method = 'gauss-lobatto'
segments_asc = 20
segments_desc_2p = (10, 10)
order = 3
solver = 'SNOPT'

# sampling scheme
samp_method = 'lhs'
nb_samp = 10

# surrogate model (accepted methods are IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC)
criterion = 'm'
train_method = 'QP'
nb_eval = 400

if kind == 'ac':
    sm = TwoDimAscConstSurrogate(moon, isp_lim, twr_lim, alt, theta, tof, t_bounds, method, segments_asc, order,
                                 solver, nb_samp, samp_method=samp_method, criterion=criterion)
elif kind == 'av':
    sm = TwoDimAscVarSurrogate(moon, isp_lim, twr_lim, alt, t_bounds, method, segments_asc, order, solver, nb_samp,
                               samp_method=samp_method, criterion=criterion)
elif kind == 'as':
    sm = TwoDimAscVToffSurrogate(moon, isp_lim, twr_lim, alt, alt_safe, slope, t_bounds, method, segments_asc, order,
                                 solver, nb_samp, samp_method=samp_method, criterion=criterion)
elif kind == 'd2p':
    sm = TwoDimDescVertSurrogate(moon, isp_lim, twr_lim, alt, alt_p, alt_switch, theta, tof_desc_2p, t_bounds, method,
                                 segments_desc_2p, order, solver, nb_samp, samp_method=samp_method, criterion=criterion)
elif kind == 'dc':
    sm = TwoDimDescConstSurrogate(moon, isp_lim, twr_lim, alt, alt_p, theta, tof, t_bounds, method, segments_asc, order,
                                  solver, nb_samp, samp_method=samp_method, criterion=criterion)
elif kind == 'dv':
    sm = TwoDimDescVarSurrogate(moon, isp_lim, twr_lim, alt, t_bounds, method, segments_asc, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion)
elif kind == 'ds':
    sm = TwoDimDescVLandSurrogate(moon, isp_lim, twr_lim, alt, alt_safe, -slope, t_bounds, method, segments_asc,
                                  order, solver, nb_samp, samp_method=samp_method, criterion=criterion)
else:
    raise ValueError('kind must be ac, av, as or d2p, dc, dv, ds')

sm.sampling()

if samp_method != 'full':
    sm.train(train_method)
    sm.evaluate(nb_eval=nb_eval)
else:
    sm.evaluate()

sm.plot()
