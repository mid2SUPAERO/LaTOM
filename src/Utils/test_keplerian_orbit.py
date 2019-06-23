#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 07:39:24 2019

@author: alberto

Test the class KeplerianOrbit
"""
import numpy as np

from keplerian_orbit import KeplerianOrbit

mu = 3.986e14 #Earth standard gravitational parameter (m^3/s^2)

a = 8000e3 #semimajor axis (m)
e = 0.0 #eccentricity
i = 90. #inclination (deg)
raan = 10. #right ascension of the ascending node (deg)
w = 0. #argument of periapsis (deg)
ta = 0. #true anomaly (deg)

kepOrb = KeplerianOrbit(mu)
kepOrb.set_coe(a, e, i, raan, w, ta, angle_unit='deg')
H = kepOrb.get_angular_momentum()
E = kepOrb.get_eccentricity_vector()
R, V = kepOrb.get_sv()

print("R:", R*1e-3, "km")
print("V:", V*1e-3, "km/s")
print("H:", H*1e-6, "km^2/s")
print("E:", E)