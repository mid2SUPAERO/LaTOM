"""
Two-phases LLO to Moon transfer with vertical touchdown
=======================================================

This example computes a two-dimensional descent trajectory from a specified LLO to the Moon surface at constant thrust.
The transfer is divided into four phases for which only the last two are transcribed into an NLP and optimized.
The aforementioned phases are the followings:
1. Impulsive burn to leave the initial LLO and enter in an Hohmann transfer with known periapsis radius
2. Hohmann transfer until the aforementioned periapsis
3. Attitude-free powered phase at constant thrust from the aforementioned periapsis to a fixed altitude or time-to-go
4. Vertical powered descent at constant thrust from a fixed altitude or time-to-go until the final touchdown

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.utils.primary import Moon
from latom.utils.spacecraft import Spacecraft
from latom.analyzer.analyzer_2d import TwoDimDescTwoPhasesAnalyzer

# trajectory
moon = Moon()  # central attracting body
alt = 100e3  # initial orbit altitude [m]
alt_p = 15e3  # periselene altitude [m]
alt_switch = 4e3  # switch altitude [m]
theta = np.pi  # guessed spawn angle [rad]
tof = (1000, 100)  # guessed time of flight [s]
t_bounds = (0., 2.)  # time of flight bounds [-]

# condition to trigger the final vertical descent, 'alt' for fixed altitude equal to 'alt_switch' or 'time' for
# fixed time-to-go equal to the second component of 'tof'
fix = 'alt'

# spacecraft
isp = 310.  # specific impulse [s]
twr = 0.9  # initial thrust/weight ratio [-]
sc = Spacecraft(isp, twr, g=moon.g)

# NLP
method = 'gauss-lobatto'
segments = (100, 20)
order = 3
solver = 'IPOPT'

# additional settings
check_partials = False  # check partial derivatives
run_driver = True  # solve the NLP
exp_sim = True  # perform explicit simulation
rec = False  # record the solution

if rec:  # files IDs in the current working directory where the solutions are serialized if 'rec' is set to 'True'
    rec_file = 'example.sql'  # implicit NLP solution
    rec_file_exp = 'example_exp.sql'  # explicit simulation
else:  # no recording if 'rec' is set to 'False'
    rec_file = rec_file_exp = None

# init analyzer
tr = TwoDimDescTwoPhasesAnalyzer(moon, sc, alt, alt_p, alt_switch, theta, tof, t_bounds, method, segments, order,
                                 solver, check_partials=check_partials, fix=fix, rec_file=rec_file)

if run_driver:

    f = tr.run_driver()  # solve the NLP

    if exp_sim:  # explicit simulation with Scipy solve_ivp method
        tr.nlp.exp_sim(rec_file=rec_file_exp)

tr.get_solutions(explicit=exp_sim, scaled=False)  # retrieve solutions
print(tr)  # print summary
tr.plot()  # plot
