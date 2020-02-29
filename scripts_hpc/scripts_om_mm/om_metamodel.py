"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.surrogate.meta_models import *
from rpfm.utils.primary import Moon


# MetaModel settings
distributed = False  # variables distributed across multiple processes
extrapolate = False  # extrapolation for out-of-bounds inputs
interp_method = 'scipy_cubic'  # interpolation method
training_data_gradients = True  # compute gradients wrt output training data
vec_size = 1  # number of points to evaluate at once
nb_samp = (2, 2)  # number of samples on which the actual solution is computed
rec_file = 'test.pkl'  # name of the file on which the solution is serialized

# ac: ascent constant, av: ascent variable, as: ascent vertical takeoff
# dc: descent constant, dv: descent variable, ds: descent vertical landing
kind = 'ds'

moon = Moon()  # central attracting body

# trajectory
alt = 100e3  # final orbit altitude [m]
alt_p = 15e3  # periselene altitude [m]
alt_safe = 5e3  # minimum safe altitude [m]
slope = 10.  # slope of the constraint on minimum safe altitude [-]
theta = np.pi/2  # guessed spawn angle [rad]
tof = 500  # guessed time of flight [s]
t_bounds = None  # time of flight bounds [-]

# grid limits
isp = [300., 400.]  # specific impulse [s]
twr = [1.5, 3.5]  # initial thrust/weight ratio [-]

# NLP
method = 'gauss-lobatto'
segments = 200
order = 3
solver = 'SNOPT'
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
else:
    raise ValueError('kind must be one between ac, av, as or dc, dv, ds')
