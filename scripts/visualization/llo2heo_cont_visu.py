"""
LLO to Apoapsis transfer visualization
======================================

This example loads and displays a series of LLO to Apoapsis transfers obtained using a continuation method for
decreasing thrust/weight ratio values.

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.utils.pickle_utils import load
from latom.utils.primary import Moon
from latom.data.continuation.data_continuation import dirname_continuation

filename = 'isp400_twr01.pkl'  # file ID in latom.data.continuation where the data are serialized
abspath = '/'.join([dirname_continuation, filename])  # absolute path to 'filename'
tr = load(abspath)  # load serialized data

moon = Moon()  # central attracting body

# boundary conditions
r_llo = tr.guess.ht.depOrb.rp/moon.R  # LLO radius [m]
rp_heo = tr.guess.ht.arrOrb.rp/moon.R  # HEO periapsis radius [m]
ra_heo = tr.guess.ht.arrOrb.ra/moon.R  # HEO apoapsis radius [m]

# spacecraft characteristics and NLP solution for lowest twr value
twr = tr.sc.twr  # thrust/weight ratio [-]
ve = tr.sc.w/moon.vc  # exhaust velocity [m/s]
tof = tr.tof[0]/moon.tc  # non-dimensional time of flight [-]
tof_days = tr.tof[0]/86400  # dimensional time of flight [days]
dtheta = tr.states[0][-1, 1] - tr.states[0][0, 1]  # total spawn angle [rad]
nb_spirals = dtheta/np.pi/2  # number of spirals [-]

# print summary
print(f"Moon radius: 1.0\nGravitational parameter: 1.0")
print(f"LLO radius: {r_llo:.16f}")
print(f"HEO periapsis radius: {rp_heo:.16f}\nHEO apoapsis radius: {ra_heo:.16f}")
print(f"Thrust/weight ratio: {twr:.16f}\nExhaust velocity: {ve:.16f}")
print(f"\nTime of flight: {tof:.16f} or {tof_days:.16f} days")
print(f"Number of spirals: {nb_spirals:.16f}")
print(f"Propellant fraction (excluding insertion): {(1 - tr.states[0][-1, -1]):.16f}")
print(f"Propellant fraction (total): {(1 - tr.states[-1][-1, -1]):.16f}")
print(tr)

tr.plot()
