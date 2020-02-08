"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.surrogate.meta_models import *
from rpfm.utils.primary import Moon


# MetaModel settings
rec_file = 'mm.pkl'  # name of the file on which the solution is serialized
interp_method = 'cubic'  # interpolation method
nb_samp = (5, 5)  # number of samples on which the actual solution is computed

# ac: ascent constant, av: ascent variable, as: ascent vertical takeoff
# dc: descent constant, dv: descent variable, ds: descent vertical landing
kind = 'ac'

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
isp = [250., 500.]  # specific impulse [s]
twr = [1.1, 4.0]  # initial thrust/weight ratio [-]

# NLP
method = 'radau-ps'
segments = 20
order = 3
solver = 'SNOPT'
snopt_opts = {'Major feasibility tolerance': 1e-8, 'Major optimality tolerance': 1e-8,
              'Minor feasibility tolerance': 1e-8}

if kind == 'ac':
    a = TwoDimAscConstMetaModel(interp_method=interp_method)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file, theta=theta, tof=tof)
elif kind == 'av':
    a = TwoDimAscVarMetaModel(interp_method=interp_method)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file)
elif kind == 'as':
    a = TwoDimAscVToffMetaModel(interp_method=interp_method)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file, alt_safe=alt_safe, slope=slope)
elif kind == 'dc':
    a = TwoDimDescConstMetaModel(interp_method=interp_method)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file, alt_p=alt_p, theta=theta, tof=tof)
elif kind == 'dv':
    a = TwoDimDescVarMetaModel(interp_method=interp_method)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file)
elif kind == 'ds':
    a = TwoDimDescVLandMetaModel(interp_method=interp_method)
    a.sampling(moon, twr, isp, alt, t_bounds, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
               rec_file=rec_file, alt_safe=alt_safe, slope=-slope)
else:
    raise ValueError('kind must be one between ac, av, as or dc, dv, ds')
