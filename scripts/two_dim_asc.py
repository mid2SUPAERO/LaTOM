"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.primary import Moon
from rpfm.utils.spacecraft import Spacecraft
from rpfm.analyzer.analyzer_2d import TwoDimAscConstAnalyzer, TwoDimAscVarAnalyzer, TwoDimAscVToffAnalyzer


rec_file = '/home/alberto/Downloads/rec.sql'
rec_file_exp = '/home/alberto/Downloads/rec_exp.sql'

moon = Moon()
sc = Spacecraft(450., 2.1, g=moon.g)
kind = 'c'

if kind == 'c':

    tr_const = TwoDimAscConstAnalyzer(moon, sc, 86.87e3, np.pi/2, 500, None, 'gauss-lobatto', 10, 3, 'SNOPT',
                                      rec_file=rec_file)

    tr_const.run_driver()
    tr_const.nlp.exp_sim(rec_file=rec_file_exp)
    tr_const.get_solutions()

    print(tr_const)

    tr_const.plot()

elif kind == 'v':

    tr_var = TwoDimAscVarAnalyzer(moon, sc, 86.87e3, None, 'gauss-lobatto', 150, 3, 'SNOPT', u_bound=True)

    tr_var.run_driver()
    tr_var.nlp.exp_sim()
    tr_var.get_solutions()

    print(tr_var)

    tr_var.plot()

elif kind == 's':

    tr_safe = TwoDimAscVToffAnalyzer(moon, sc, 86.87e3, 5e3, 10., None, 'gauss-lobatto', 200, 3, 'SNOPT', check_partials=True)

    tr_safe.run_driver()
    # tr_safe.nlp.exp_sim()
    tr_safe.get_solutions(explicit=False)
    tr_safe.plot()

else:
    raise ValueError('kind not recognized')
