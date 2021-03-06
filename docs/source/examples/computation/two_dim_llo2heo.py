"""
Two-dimensional LLO to HEO transfers
====================================

This example computes a two-dimensional LLO to HEO transfer trajectory using one of the following models:
1. Single-phase transfer with variable thrust, open departure point on the LLO and fixed insertion at the HEO apoapsis
2. Finite escape burn at constant thrust to leave the initial LLO and inject into a ballistic arc whose apoapsis radius
coincides with the HEO one
3. Three-phases transfer composed by a first powered phase at constant thrust to leave the LLO, a ballistic arc to
reach the vicinity of the HEO apoapsis and a final powered phase at constant thrust to inject in the vicinity of the HEO
apoapsis

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.utils.primary import Moon
from latom.utils.spacecraft import Spacecraft
from latom.analyzer.analyzer_heo_2d import TwoDimLLO2HEOAnalyzer, TwoDimLLO2ApoAnalyzer, TwoDim3PhasesLLO2HEOAnalyzer

# type of transfer among the followings:
# 'full' for single-phase LLO to HEO transfer
# 'first' for LLO to apoapsis transfer
# '3p' for three-phases LLO to HEO transfer
kind = 'first'

# trajectory
moon = Moon()  # central attracting body
llo_alt = 100e3  # initial LLO altitude [m]
heo_rp = 3150e3  # target HEO periselene radius [m]
heo_period = 6.5655*86400  # target HEO period [s]

# spacecraft
isp = 450.  # specific impulse [s]
twr = 2.1  # initial thrust/weight ratio [-]
sc = Spacecraft(isp, twr, g=moon.g)

# NLP
method = 'gauss-lobatto'
segments = 400
order = 3
solver = 'SNOPT'
snopt_opts = {'Major feasibility tolerance': 1e-12, 'Major optimality tolerance': 1e-12,
              'Minor feasibility tolerance': 1e-12}

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

# analyzer
if kind == 'full':
    tr = TwoDimLLO2HEOAnalyzer(moon, sc, llo_alt, heo_rp, heo_period, None, method, segments, order, solver,
                               snopt_opts=snopt_opts, rec_file=rec_file)
elif kind == 'first':
    tr = TwoDimLLO2ApoAnalyzer(moon, sc, llo_alt, heo_rp, heo_period, None, method, segments, order, solver,
                               snopt_opts=snopt_opts, check_partials=False, rec_file=rec_file)
elif kind == '3p':
    segments = (60, 400, 60)  # modified segments for three-phases
    t_bounds = ((0.2, 1.8), (0.2, 1.8), (0.2, 1.8))  # modified time of flight bounds for three-phases [-]

    tr = TwoDim3PhasesLLO2HEOAnalyzer(moon, sc, llo_alt, heo_rp, heo_period, t_bounds, method, segments, order, solver,
                                      snopt_opts=snopt_opts, check_partials=False, rec_file=rec_file)
else:
    raise ValueError('Kind must be either full, first or 3p')

if run_driver:

    f = tr.run_driver()  # solve the NLP

    if exp_sim:  # explicit simulation with Scipy solve_ivp method
        tr.nlp.exp_sim(rec_file=rec_file_exp)

tr.get_solutions(explicit=exp_sim, scaled=False)  # retrieve solutions
print(tr)  # print summary
tr.plot()  # plot
