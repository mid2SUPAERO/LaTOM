"""
Two-dimensional LLO to Moon transfer
====================================

This examples computes a two-dimensional descent trajectory from a specified LLO the Moon surface with constant or
variable thrust and optional minimum safe altitude.

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.utils.primary import Moon
from latom.utils.spacecraft import Spacecraft
from latom.analyzer.analyzer_2d import TwoDimDescConstAnalyzer, TwoDimDescVarAnalyzer, TwoDimDescVLandAnalyzer

# trajectory
kind = 's'  # 'c' for constant, 'v' for variable and 's' for variable with minimum safe altitude
moon = Moon()  # central attracting body
alt = 100e3  # initial orbit altitude [m]
alt_p = 15e3  # periselene altitude [m]
theta = np.pi/2  # guessed spawn angle [rad]
tof = 1000  # guessed time of flight [s]
t_bounds = (0., 2.)  # time of flight bounds [-]
alt_safe = 5e3  # minimum safe altitude [m]
slope = -5.  # slope of the constraint on minimum safe altitude [-]

# spacecraft
isp = 400.  # specific impulse [s]
twr = 0.9  # initial thrust/weight ratio [-]
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
rec = True  # record the solution

if rec:  # files IDs in the current working directory where the solutions are serialized if 'rec' is set to 'True'
    rec_file = 'example_imp.sql'  # implicit NLP solution
    rec_file_exp = 'example_exp.sql'  # explicit simulation
else:  # no recording if 'rec' is set to 'False'
    rec_file = rec_file_exp = None

# init analyzer
if kind == 'c':
    tr = TwoDimDescConstAnalyzer(moon, sc, alt, alt_p, theta, tof, t_bounds, method, segments, order, solver,
                                 check_partials=check_partials, snopt_opts=snopt_opts, u_bound=u_bound,
                                 rec_file=rec_file)
elif kind == 'v':
    tr = TwoDimDescVarAnalyzer(moon, sc, alt, t_bounds, method, segments, order, solver, check_partials=check_partials,
                               snopt_opts=snopt_opts, u_bound=u_bound, rec_file=rec_file)
elif kind == 's':
    tr = TwoDimDescVLandAnalyzer(moon, sc, alt, alt_safe, slope, t_bounds, method, segments, order, solver,
                                 check_partials=check_partials, snopt_opts=snopt_opts, u_bound=u_bound,
                                 rec_file=rec_file)
else:
    raise ValueError('kind not recognized')

if run_driver:

    f = tr.run_driver()  # solve the NLP

    if exp_sim:  # explicit simulation with Scipy solve_ivp method
        tr.nlp.exp_sim(rec_file=rec_file_exp)

tr.get_solutions(explicit=exp_sim, scaled=False)  # retrieve solutions
print(tr)  # print summary
tr.plot()  # plot
