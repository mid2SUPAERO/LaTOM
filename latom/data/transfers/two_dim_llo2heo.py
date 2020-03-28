"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.utils.primary import Moon
from latom.utils.spacecraft import Spacecraft
from latom.analyzer.analyzer_heo_2d import TwoDimLLO2HEOAnalyzer, TwoDimLLO2ApoAnalyzer, TwoDim3PhasesLLO2HEOAnalyzer

kind = '3p'

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
method = 'radau-ps'
segments = 100
order = 3
solver = 'IPOPT'
snopt_opts = {'Major feasibility tolerance': 1e-12, 'Major optimality tolerance': 1e-12,
              'Minor feasibility tolerance': 1e-12}

# additional settings
run_driver = True  # solve the NLP
exp_sim = True  # perform explicit simulation

# analyzer
if kind == 'full':
    tr = TwoDimLLO2HEOAnalyzer(moon, sc, llo_alt, heo_rp, heo_period, None, method, segments, order, solver,
                               snopt_opts=snopt_opts)
elif kind == 'first':
    tr = TwoDimLLO2ApoAnalyzer(moon, sc, llo_alt, heo_rp, heo_period, None, method, segments, order, solver,
                               snopt_opts=snopt_opts, check_partials=False)
elif kind == '3p':
    method = ('gauss-lobatto', 'gauss-lobatto', 'gauss-lobatto')
    segments = (200, 1200, 100)
    t_bounds = ((0.2, 1.8), (0.2, 1.8), (0.2, 1.8))

    tr = TwoDim3PhasesLLO2HEOAnalyzer(moon, sc, llo_alt, heo_rp, heo_period, t_bounds, method, segments, order, solver,
                                      snopt_opts=snopt_opts, check_partials=False, rec_file='llo2heo_ipopt1200.pkl')
else:
    raise ValueError('Kind must be either full, first, 2p, 3p')

if run_driver:

    f = tr.run_driver()

    if exp_sim:
        tr.nlp.exp_sim(rec_file='llo2heo_ipopt1200_exp.pkl')

tr.get_solutions(explicit=exp_sim, scaled=False)

print(tr)

# tr.plot()
