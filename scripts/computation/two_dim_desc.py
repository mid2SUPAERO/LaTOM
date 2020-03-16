"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.primary import Moon
from rpfm.utils.spacecraft import Spacecraft
from rpfm.analyzer.analyzer_2d import TwoDimDescConstAnalyzer, TwoDimDescVarAnalyzer, TwoDimDescVLandAnalyzer


# trajectory
kind = 'v'
moon = Moon()
alt = 100e3  # initial orbit altitude [m]
alt_p = 15e3  # periselene altitude [m]
theta = np.pi/2  # guessed spawn angle [rad]
tof = 1000  # guessed time of flight [s]
t_bounds = (0., 2.)  # time of flight bounds [-]
alt_safe = 5e3  # minimum safe altitude [m]
slope = -5.  # slope of the constraint on minimum safe altitude [-]

# spacecraft
isp = 450.  # specific impulse [s]
twr = 2.1  # initial thrust/weight ratio [-]

sc = Spacecraft(isp, twr, g=moon.g)

# NLP
method = 'gauss-lobatto'
segments = 200
order = 3
solver = 'SNOPT'
snopt_opts = {'Major feasibility tolerance': 1e-8, 'Major optimality tolerance': 1e-8,
              'Minor feasibility tolerance': 1e-8}

# additional settings
u_bound = 'upper'  # upper bound on radial velocity
check_partials = False  # check partial derivatives
run_driver = True  # solve the NLP
exp_sim = run_driver  # perform explicit simulation
rec = False  # record the solution

# record databases
rec_file = '/home/alberto/Downloads/rec.sql'
rec_file_exp = '/home/alberto/Downloads/rec_exp.sql'

# init analyzer
if kind == 'c':
    tr = TwoDimDescConstAnalyzer(moon, sc, alt, alt_p, theta, tof, t_bounds, method, segments, order, solver,
                                 check_partials=check_partials, snopt_opts=snopt_opts, u_bound=u_bound)
elif kind == 'v':
    tr = TwoDimDescVarAnalyzer(moon, sc, alt, t_bounds, method, segments, order, solver, check_partials=check_partials,
                               snopt_opts=snopt_opts, u_bound=u_bound)
elif kind == 's':
    tr = TwoDimDescVLandAnalyzer(moon, sc, alt, alt_safe, slope, t_bounds, method, segments, order, solver,
                                 check_partials=check_partials, snopt_opts=snopt_opts, u_bound=u_bound)
else:
    raise ValueError('kind not recognized')

if run_driver:

    f = tr.run_driver()
    print('Failure: ' + str(f))

    if exp_sim:
        tr.nlp.exp_sim()

tr.get_solutions(explicit=exp_sim)

print(tr)

tr.plot()
