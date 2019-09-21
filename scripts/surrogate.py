
from rpfm.surrogate.surrogate_2d import TwoDimAscSurrogate
from rpfm.utils.primary import Moon


# trajectory
moon = Moon()
alt = 86.87e3  # final orbit altitude [m]
tof = 500  # guessed time of flight [s]
t_bounds = None  # time of flight bounds [-]

# NLP
method = 'gauss-lobatto'
segments = 10
order = 3
solver = 'SNOPT'
nb_samp = 10

sm = TwoDimAscSurrogate(moon, (300., 450.), (1.1, 2.1), alt, t_bounds, method, segments, order, solver, nb_samp)
sm.sampling()