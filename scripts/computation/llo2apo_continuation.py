"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.primary import Moon
from rpfm.utils.spacecraft import Spacecraft
from rpfm.analyzer.analyzer_heo_2d import TwoDimLLO2ApoContinuationAnalyzer
from rpfm.data.data import dirname

# file ID
fid = 'test_log450.pkl'

# trajectory
moon = Moon()
llo_alt = 100e3  # initial LLO altitude [m]
heo_rp = 3150e3  # target HEO periselene radius [m]
heo_period = 6.5655 * 86400  # target HEO period [s]

# spacecraft
isp = 450.  # specific impulse [s]
twr_list = np.linspace(1.5, 1.2, 5)  # range of thrust/weight ratios in logarithmic scale [-]
twr0 = np.exp(twr_list[0])  # maximum thrust/weight ratio in absolute value [-]
log_scale = True
sc = Spacecraft(isp, twr0, g=moon.g)

# NLP
method = 'gauss-lobatto'
segments = 100
order = 3
solver = 'IPOPT'
snopt_opts = {'Major feasibility tolerance': 1e-12, 'Major optimality tolerance': 1e-12,
              'Minor feasibility tolerance': 1e-12}

# additional settings
run_driver = True  # solve the NLP
exp_sim = True  # perform explicit simulation

tr = TwoDimLLO2ApoContinuationAnalyzer(moon, sc, llo_alt, heo_rp, heo_period, None, twr_list, method, segments, order,
                                       solver, snopt_opts=snopt_opts, check_partials=False, log_scale=log_scale)

if run_driver:

    tr.run_continuation()

    if exp_sim:
        tr.nlp.exp_sim()

tr.get_solutions(explicit=exp_sim, scaled=False)

print(tr)

if fid is not None:
    tr.save('/'.join([dirname, 'continuation', fid]))

tr.plot()
