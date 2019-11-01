import numpy as np

from rpfm.plots.solutions import TwoDimSolPlot
from rpfm.guess.guess_heo_2d import TwoDim2PhasesLLO2HEOGuess
from rpfm.utils.primary import Moon
from rpfm.utils.spacecraft import Spacecraft

moon = Moon()
sc = Spacecraft(450., 2.1, g=moon.g)

guess = TwoDim2PhasesLLO2HEOGuess(moon.GM, moon.R, 100e3, 3150e3, 6.5655*86400, sc)
guess.compute_trajectory(nb1=100, nb2=50)

time = np.vstack((guess.pow1.t, guess.pow2.t))
states = np.vstack((guess.pow1.states, guess.pow2.states))
controls = np.vstack((guess.pow1.controls, guess.pow2.controls))

print(guess)

plt = TwoDimSolPlot(moon.R, time, states, controls, threshold=None, a=guess.ht.arrOrb.a, e=guess.ht.arrOrb.e)
plt.plot()
