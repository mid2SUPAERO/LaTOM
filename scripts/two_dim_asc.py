"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.primary import Moon
from rpfm.utils.spacecraft import Spacecraft
from rpfm.analyzer.analyzer_2d import TwoDimAscConstAnalyzer, TwoDimAscVarAnalyzer


moon = Moon()
sc = Spacecraft(450., 2.1)
kind = 'v'

if kind == 'c':

    tr_const = TwoDimAscConstAnalyzer(moon, sc, 86.87e3, np.pi/2, 500, None, 'gauss-lobatto', 10, 3, 'IPOPT')

    tr_const.run_driver()
    tr_const.nlp.exp_sim()
    tr_const.get_solutions()
    tr_const.plot()

elif kind == 'v':

    tr_var = TwoDimAscVarAnalyzer(moon, sc, 86.87e3, (0.5, 1.5), 'gauss-lobatto', 150, 3, 'IPOPT')

    tr_var.run_driver()
    tr_var.nlp.exp_sim()
    tr_var.get_solutions()
    tr_var.plot()

else:

    pass
