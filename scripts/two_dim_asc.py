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
kind = 's'

if kind == 'c':
    tr = TwoDimAscConstAnalyzer(moon, sc, 86.87e3, np.pi/2, 500, None, 'gauss-lobatto', 10, 3, 'SNOPT')
elif kind == 'v':
    tr = TwoDimAscVarAnalyzer(moon, sc, 86.87e3, None, 'gauss-lobatto', 80, 3, 'SNOPT', u_bound=True)
elif kind == 's':
    tr = TwoDimAscVToffAnalyzer(moon, sc, 86.87e3, 5e3, 1., None, 'gauss-lobatto', 100, 3, 'SNOPT', check_partials=True,
                                rec_file=rec_file)
else:
    raise ValueError('kind not recognized')

tr.run_driver()
tr.nlp.exp_sim(rec_file=rec_file_exp)
tr.get_solutions()

print(tr)

tr.plot()
