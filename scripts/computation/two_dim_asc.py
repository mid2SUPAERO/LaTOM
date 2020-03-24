"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.utils.primary import Moon
from latom.utils.spacecraft import Spacecraft
from latom.analyzer.analyzer_2d import TwoDimAscConstAnalyzer, TwoDimAscVarAnalyzer, TwoDimAscVToffAnalyzer


# trajectory
thrust = 's'
moon = Moon()
alt = 86.87e3  # final orbit altitude [m]
theta = np.pi/2  # guessed spawn angle [rad]
tof = 2000  # guessed time of flight [s]
t_bounds = None  # time of flight bounds [-]
alt_safe = 5e3  # minimum safe altitude [m]
slope = 100.  # slope of the constraint on minimum safe altitude [-]

# spacecraft
isp = 450.  # specific impulse [s]
twr = 2.1  # initial thrust/weight ratio [-]

sc = Spacecraft(isp, twr, g=moon.g)

# NLP
method = 'gauss-lobatto'
segments = 120
order = 3
solver = 'SNOPT'
snopt_opts = {'Major feasibility tolerance': 1e-8, 'Major optimality tolerance': 1e-8,
              'Minor feasibility tolerance': 1e-8}

# additional settings
u_bound = 'lower'  # lower bound on radial velocity
check_partials = False  # check partial derivatives
run_driver = True  # solve the NLP
exp_sim = run_driver  # perform explicit simulation
rec = False  # record the solution

# record databases
rec_file = '/home/alberto/Downloads/rec.sql'
rec_file_exp = '/home/alberto/Downloads/rec_exp.sql'

# init analyzer
if thrust == 'c':
    tr = TwoDimAscConstAnalyzer(moon, sc, alt, theta, tof, t_bounds, method, segments, order, solver, u_bound=u_bound,
                                check_partials=check_partials)
elif thrust == 'v':
    tr = TwoDimAscVarAnalyzer(moon, sc, alt, t_bounds, method, segments, order, solver, u_bound=u_bound,
                              check_partials=check_partials, snopt_opts=snopt_opts)
elif thrust == 's':
    tr = TwoDimAscVToffAnalyzer(moon, sc, alt, alt_safe, slope, t_bounds, method, segments, order, solver,
                                u_bound=u_bound, check_partials=check_partials)
else:
    raise ValueError('kind not recognized')

if run_driver:

    f = tr.run_driver()

    if exp_sim:
        tr.nlp.exp_sim()

tr.get_solutions(explicit=exp_sim, scaled=False)

print(tr)

tr.plot()
