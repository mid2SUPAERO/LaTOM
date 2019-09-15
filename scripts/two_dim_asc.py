"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.primary import Moon
from rpfm.utils.spacecraft import Spacecraft
from rpfm.analyzer.analyzer_2d import TwoDimAscConstAnalyzer, TwoDimAscVarAnalyzer, TwoDimAscVToffAnalyzer


# trajectory
kind = 'c'
moon = Moon()
alt = 86.87e3  # final orbit altitude [m]
theta = np.pi/2  # guessed spawn angle [rad]
tof = 500  # guessed time of flight [s]
t_bounds = None  # time of flight bounds [-]
alt_safe = 5e3  # minimum safe altitude [m]
slope = 10.  # slope of the constraint on minimum safe altitude [-]

# spacecraft
isp = 450.  # specific impulse [s]
twr = 2.1  # initial thrust/weight ratio [-]

sc = Spacecraft(isp, twr, g=moon.g)

# NLP
method = 'gauss-lobatto'
segments = 10
order = 3
solver = 'SNOPT'

# additional settings
u_bound = True  # lower bound on radial velocity
check_partials = False  # check partial derivatives
run_driver = True  # solve the NLP
exp_sim = True  # perform explicit simulation
rec = False  # record the solution

# record databases
rec_file = '/home/alberto/Downloads/rec.sql'
rec_file_exp = '/home/alberto/Downloads/rec_exp.sql'

# init analyzer
if kind == 'c':
    tr = TwoDimAscConstAnalyzer(moon, sc, alt, theta, tof, t_bounds, method, segments, order, solver, u_bound=u_bound,
                                check_partials=check_partials)
elif kind == 'v':
    tr = TwoDimAscVarAnalyzer(moon, sc, alt, t_bounds, method, segments, order, solver, u_bound=u_bound,
                              check_partials=check_partials)
elif kind == 's':
    tr = TwoDimAscVToffAnalyzer(moon, sc, alt, alt_safe, slope, t_bounds, method, segments, order, solver,
                                u_bound=u_bound, check_partials=check_partials)
else:
    raise ValueError('kind not recognized')

if run_driver:

    tr.run_driver()

    if exp_sim:
        tr.nlp.exp_sim()

tr.get_solutions(explicit=exp_sim)

print(tr)

tr.plot()
