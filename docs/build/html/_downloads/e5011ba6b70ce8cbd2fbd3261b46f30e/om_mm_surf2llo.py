"""
OpenMDAO MetaModel for Moon to LLO and LLO to Moon transfers
============================================================

This example computes the sampling grid and training points to assemble an OpenMDAO MetaModel for a Moon to LLO or LLO
to Moon transfer with constant or variable thrust and optional minimum safe altitude.

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from latom.surrogate.om_metamodels import *
from latom.utils.primary import Moon


# MetaModel settings
distributed = False  # variables distributed across multiple processes
extrapolate = False  # extrapolation for out-of-bounds inputs
interp_method = 'slinear'  # interpolation method
training_data_gradients = True  # compute gradients wrt output training data
vec_size = 1  # number of points to evaluate at once
nb_samp = (2, 2)  # number of samples on which the actual solution is computed
rec_file = 'example.pkl'  # name of the file in latom.data.metamodels in which the solution is serialized

# transfer type among the followings:
# ac: ascent constant, av: ascent variable, as: ascent vertical takeoff
# dc: descent constant, dv: descent variable, ds: descent vertical landing, d2p: two-phases descent vertical landing
kind = 'ac'

moon = Moon()  # central attracting body

# trajectory
alt = 100e3  # final orbit altitude [m]
alt_p = 15e3  # periselene altitude [m]
alt_safe = 4e3  # minimum safe altitude or switch altitude [m]
slope = 10.  # slope of the constraint on minimum safe altitude [-]
theta = np.pi  # guessed spawn angle [rad]
tof = (1000., 100.)  # guessed time of flight [s]
t_bounds = (0., 2.)  # time of flight bounds [-]
fix = 'alt'  # fixed parameter at phase switch between alt or time

# grid limits
isp = [250., 350.]  # specific impulse [s]
twr = [0.5, 2.0]  # initial thrust/weight ratio [-]

# NLP
method = 'gauss-lobatto'
segments = (100, 20)
order = 3
solver = 'IPOPT'
snopt_opts = {'Major feasibility tolerance': 1e-12, 'Major optimality tolerance': 1e-12,
              'Minor feasibility tolerance': 1e-12}

if kind == 'ac':
    a = TwoDimAscConstMetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
                                training_data_gradients=training_data_gradients, vec_size=vec_size)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file, theta=theta, tof=tof)
elif kind == 'av':
    a = TwoDimAscVarMetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
                              training_data_gradients=training_data_gradients, vec_size=vec_size)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file)
elif kind == 'as':
    a = TwoDimAscVToffMetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
                                training_data_gradients=training_data_gradients, vec_size=vec_size)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file, alt_safe=alt_safe, slope=slope)
elif kind == 'dc':
    a = TwoDimDescConstMetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
                                 training_data_gradients=training_data_gradients, vec_size=vec_size)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file, alt_p=alt_p, theta=theta, tof=tof)
elif kind == 'dv':
    a = TwoDimDescVarMetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
                               training_data_gradients=training_data_gradients, vec_size=vec_size)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file)
elif kind == 'ds':
    a = TwoDimDescVLandMetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
                                 training_data_gradients=training_data_gradients, vec_size=vec_size)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file, alt_safe=alt_safe, slope=-slope)
elif kind == 'd2p':
    a = TwoDimDescTwoPhasesMetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
                                     training_data_gradients=training_data_gradients, vec_size=vec_size)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file, alt_p=alt_p, alt_switch=alt_safe, theta=theta, tof=tof, fix=fix)
else:
    raise ValueError('kind must be one between ac, av, as or dc, dv, ds, d2p')

