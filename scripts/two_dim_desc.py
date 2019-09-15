"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.primary import Moon
from rpfm.utils.spacecraft import Spacecraft
from rpfm.analyzer.analyzer_2d import TwoDimDescConstAnalyzer


# trajectory
kind = 'c'
moon = Moon()
alt = 100e3  # initial orbit altitude [m]
switch_alt = 15e3  # switch altitude [m]
theta = np.pi/2  # guessed spawn angle [rad]
tof = 1000  # guessed time of flight [s]
t_bounds = None  # time of flight bounds [-]

# spacecraft
isp = 450.  # specific impulse [s]
twr = 2.1  # initial thrust/weight ratio [-]

sc = Spacecraft(isp, twr, g=moon.g)

# NLP
method = 'gauss-lobatto'
segments = 60
order = 3
solver = 'IPOPT'

# additional settings
check_partials = True  # check partial derivatives
run_driver = True  # solve the NLP
exp_sim = True  # perform explicit simulation
rec = False  # record the solution

# record databases
rec_file = '/home/alberto/Downloads/rec.sql'
rec_file_exp = '/home/alberto/Downloads/rec_exp.sql'

# init analyzer
if kind == 'c':
    tr = TwoDimDescConstAnalyzer(moon, sc, alt, switch_alt, theta, tof, t_bounds, method, segments, order, solver,
                                 check_partials=check_partials)
else:
    raise ValueError('kind not recognized')

if run_driver:

    tr.run_driver()

    if exp_sim:
        tr.nlp.exp_sim()

tr.get_solutions(explicit=exp_sim)

print(tr)

tr.plot()
