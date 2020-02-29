"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.primary import Moon
from rpfm.utils.spacecraft import Spacecraft
from rpfm.analyzer.analyzer_2d import TwoDimDescTwoPhasesAnalyzer


# trajectory
kind = 'v'
moon = Moon()
alt = 100e3  # initial orbit altitude [m]
alt_p = 15e3  # periselene altitude [m]
alt_switch = 4e3  # switch altitude [m]
theta = np.pi  # guessed spawn angle [rad]
tof = (1000, 100)  # guessed time of flight [s]
t_bounds = (0., 2.)  # time of flight bounds [-]
fix = 'alt'

# spacecraft
isp = 250.  # specific impulse [s]
twr = 0.5  # initial thrust/weight ratio [-]
sc = Spacecraft(isp, twr, g=moon.g)

# NLP
method = 'gauss-lobatto'
segments = (100, 20)
order = 3
solver = 'SNOPT'

# additional settings
check_partials = True  # check partial derivatives
run_driver = True  # solve the NLP
exp_sim = False  # perform explicit simulation
rec = False  # record the solution

# init analyzer
if kind == 'v':
    tr = TwoDimDescTwoPhasesAnalyzer(moon, sc, alt, alt_p, alt_switch, theta, tof, t_bounds, method, segments, order,
                                     solver, check_partials=check_partials, fix=fix)
else:
    raise ValueError('kind not recognized')

if run_driver:

    tr.run_driver()

    if exp_sim:
        tr.nlp.exp_sim()

tr.get_solutions(explicit=exp_sim)

print(tr)

tr.plot()
