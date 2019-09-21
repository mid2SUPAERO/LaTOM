
import matplotlib.pyplot as plt

from rpfm.surrogate.surrogate_2d import TwoDimAscSurrogate, TwoDimAscVToffSurrogate, TwoDimDescVertSurrogate
from rpfm.utils.primary import Moon
from rpfm.plots.response_surfaces import RespSurf
import numpy as np


# trajectory
moon = Moon()
alt = 86.87e3  # final orbit altitude [m]
tof = 500  # guessed time of flight [s]
t_bounds = None  # time of flight bounds [-]

# NLP
method = 'gauss-lobatto'
segments = (10, 10)
order = 3
solver = 'IPOPT'
nb_samp = 10
nb_eval = 100

sm = TwoDimDescVertSurrogate(moon, (430., 450.), (1.5, 2.0), alt, 15000, 3e3, 10*np.pi, (1000, 100), t_bounds, method,
                             segments, order, solver, nb_samp, nb_eval, samp_method='lhs')

sm.sampling()
#sm.train(RMTB)
#sm.evaluate()

#p = RespSurf(sm.x_eval, sm.m_eval, sm.tof_eval)
#p.plot()
#plt.show()