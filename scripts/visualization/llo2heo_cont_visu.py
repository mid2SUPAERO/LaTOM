import numpy as np

from latom.utils.pickle_utils import load
from latom.utils.primary import Moon
from latom.data.continuation.data_continuation import dirname_continuation

# load saved data
filename = 'lin350.pkl'
abspath = '/'.join([dirname_continuation, 'tests', filename])
tr = load(abspath)

# primary
moon = Moon()

# departure orbit
r_llo = tr.guess.ht.depOrb.rp/moon.R

# target orbit
rp_heo = tr.guess.ht.arrOrb.rp/moon.R
ra_heo = tr.guess.ht.arrOrb.ra/moon.R

# spacecraft
twr = tr.sc.twr
ve = tr.sc.w/moon.vc

# time of flight
tof = tr.tof[0]/moon.tc
tof_days = tr.tof[0]/86400

# number of spirals
dtheta = tr.states[0][-1, 1] - tr.states[0][0, 1]
nb_spirals = dtheta/np.pi/2

# print
print(f"Moon radius: 1.0\nGravitational parameter: 1.0")
print(f"LLO radius: {r_llo:.16f}")
print(f"HEO periapsis radius: {rp_heo:.16f}\nHEO apoapsis radius: {ra_heo:.16f}")
print(f"Thrust/weight ratio: {twr:.16f}\nExhaust velocity: {ve:.16f}")
print(f"\nTime of flight: {tof:.16f} or {tof_days:.16f} days")
print(f"Number of spirals: {nb_spirals:.16f}")
print(f"Propellant fraction (excluding insertion): {(1 - tr.states[0][-1, -1]):.16f}")
print(f"Propellant fraction (total): {(1 - tr.states[-1][-1, -1]):.16f}")

print(tr)
# tr.plot()
