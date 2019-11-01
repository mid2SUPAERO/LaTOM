import numpy as np

from rpfm.utils.keplerian_orbit import TwoDimOrb

gm = 398600.4412

a0 = 8000
e0 = 0.8
h0 = (gm*a0*(1-e0*e0))**0.5
ta0 = np.pi/6

r, u, v, = TwoDimOrb.coe2polar(gm, ta0, a=a0, h=h0)

print('r:', r, 'km')
print('u:', u, 'km/s')
print('v:', v, 'km/s')

a, e, h, ta = TwoDimOrb.polar2coe(gm, r, u, v)

print('\na:', a, 'km')
print('e:', e)
print('h:', h, 'km^2/s')
print('ta:', ta, 'rad')
