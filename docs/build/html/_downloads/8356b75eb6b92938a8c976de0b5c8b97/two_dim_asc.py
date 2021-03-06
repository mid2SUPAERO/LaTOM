"""
Two-dimensional Moon to LLO transfer
====================================

This example computes a two-dimensional ascent trajectory from the Moon surface to a specified LLO with constant or
variable thrust and optional minimum safe altitude.

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.utils.primary import Moon
from latom.utils.spacecraft import Spacecraft
from latom.analyzer.analyzer_2d import TwoDimAscConstAnalyzer, TwoDimAscVarAnalyzer, TwoDimAscVToffAnalyzer

# trajectory
thrust = 's'  # 'c' for constant, 'v' for variable and 's' for variable with minimum safe altitude
moon = Moon()  # central attracting body
alt = 100e3  # final orbit altitude [m]
theta = np.pi/2  # guessed spawn angle [rad]
tof = 2000  # guessed time of flight [s]
t_bounds = (0.0, 2.0)  # time of flight bounds [-]
alt_safe = 5e3  # minimum safe altitude [m]
slope = 10.  # slope of the constraint on minimum safe altitude [-]

# spacecraft
isp = 450.  # specific impulse [s]
twr = 2.1  # initial thrust/weight ratio [-]
sc = Spacecraft(isp, twr, g=moon.g)

# NLP
method = 'gauss-lobatto'
segments = 200
order = 3
solver = 'SNOPT'
snopt_opts = {'Major feasibility tolerance': 1e-12, 'Major optimality tolerance': 1e-12,
              'Minor feasibility tolerance': 1e-12}

# additional settings
u_bound = 'lower'  # lower bound on radial velocity
check_partials = False  # check partial derivatives
run_driver = True  # solve the NLP
exp_sim = run_driver  # perform explicit simulation
rec = False  # record the solution

if rec:  # files IDs in the current working directory where the solutions are serialized if 'rec' is set to 'True'
    rec_file = 'example_imp.sql'  # implicit NLP solution
    rec_file_exp = 'example_exp.sql'  # explicit simulation
else:  # no recording if 'rec' is set to 'False'
    rec_file = rec_file_exp = None

# init analyzer
if thrust == 'c':
    tr = TwoDimAscConstAnalyzer(moon, sc, alt, theta, tof, t_bounds, method, segments, order, solver, u_bound=u_bound,
                                check_partials=check_partials, rec_file=rec_file)
elif thrust == 'v':
    tr = TwoDimAscVarAnalyzer(moon, sc, alt, t_bounds, method, segments, order, solver, u_bound=u_bound,
                              check_partials=check_partials, snopt_opts=snopt_opts, rec_file=rec_file)
elif thrust == 's':
    tr = TwoDimAscVToffAnalyzer(moon, sc, alt, alt_safe, slope, t_bounds, method, segments, order, solver,
                                u_bound=u_bound, check_partials=check_partials, rec_file=rec_file)
else:
    raise ValueError('kind not recognized')

if run_driver:

    f = tr.run_driver()  # solve the NLP

    if exp_sim:  # explicit simulation with Scipy solve_ivp method
        tr.nlp.exp_sim(rec_file=rec_file_exp)

tr.get_solutions(explicit=exp_sim, scaled=False)  # retrieve solutions
print(tr)  # print summary
tr.plot()  # plot
