"""
Two-dimensional LLO to HEO transfers
====================================

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.utils.primary import Moon
from latom.utils.spacecraft import Spacecraft
from latom.analyzer.analyzer_heo_2d import TwoDimLLO2HEOAnalyzer, TwoDimLLO2ApoAnalyzer, TwoDim3PhasesLLO2HEOAnalyzer

# 'full' for single-phase LLO to HEO transfer, 'first' for LLO to apoapsis transfer and
# '3p' for three-phases LLO to HEO transfer
kind = 'first'

# trajectory
moon = Moon()
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
run_driver = True  # solve the NLP
exp_sim = run_driver  # perform explicit simulation

# analyzer
if kind == 'full':
    tr = TwoDimLLO2HEOAnalyzer(moon, sc, llo_alt, heo_rp, heo_period, None, method, segments, order, solver,
                               snopt_opts=snopt_opts)
elif kind == 'first':
    tr = TwoDimLLO2ApoAnalyzer(moon, sc, llo_alt, heo_rp, heo_period, None, method, segments, order, solver,
                               snopt_opts=snopt_opts, check_partials=False)
elif kind == '3p':
    # method = ('gauss-lobatto', 'radau-ps', 'gauss-lobatto')
    segments = (60, 400, 60)
    t_bounds = ((0.2, 1.8), (0.2, 1.8), (0.2, 1.8))

    tr = TwoDim3PhasesLLO2HEOAnalyzer(moon, sc, llo_alt, heo_rp, heo_period, t_bounds, method, segments, order, solver,
                                      snopt_opts=snopt_opts, check_partials=False)
else:
    raise ValueError('Kind must be either full, first or 3p')

if run_driver:

    f = tr.run_driver()

    if exp_sim:
        tr.nlp.exp_sim()

tr.get_solutions(explicit=exp_sim, scaled=False)

print(tr)

tr.plot()