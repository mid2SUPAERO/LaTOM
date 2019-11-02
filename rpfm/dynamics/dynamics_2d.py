"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np


def states_rates_coast(states, gm=1.0):

    r, theta, u, v = states

    rdot = u
    thetadot = v/r
    udot = -gm/r**2 + v**2/r
    vdot = -u*v/r

    return [rdot, thetadot, udot, vdot]


def states_rates_pow(states, controls, w, gm=1.0):

    r, theta, u, v, m = states
    thrust, alpha = controls

    rdot, thetadot, udot, vdot = states_rates_coast(states[:4], gm=gm)

    udot = udot + thrust/m*np.sin(alpha)
    vdot = vdot + thrust/m*np.cos(alpha)
    mdot = -thrust/w

    return [rdot, thetadot, udot, vdot, mdot]


def states_rates_partials_coast(states, gm=1.0):

    r, theta, u, v = states

    dr_du = 1.0

    dtheta_dr = -v/r**2
    dtheta_dv = 1/r

    du_dr = 2*gm/r**3 - v/r**2
    du_dv = 2 * v / r

    dv_dr = u*v/r**2
    dv_du = -v/r
    dv_dv = -u/r

    return [dr_du], [dtheta_dr, dtheta_dv], [du_dr, du_dv], [dv_dr, dv_du, dv_dv]


def states_rates_partials_pow(states, controls, w, gm=1.0):

    r, theta, u, v, m = states
    thrust, alpha = controls

    dr, dtheta, du, dv = states_rates_partials_coast(states[:4], gm=gm)

    du_dm = -thrust/m**2*np.sin(alpha)
    du_dalpha = thrust/m*np.cos(alpha)
    du_dthrust = np.sin(alpha)/m
    du.extend([du_dm, du_dalpha, du_dthrust])

    dv_dm = -thrust/m**2*np.cos(alpha)
    dv_dalpha = -thrust/m*np.sin(alpha)
    dv_dthrust = np.cos(alpha)/m
    dv.extend([dv_dm, dv_dalpha, dv_dthrust])

    dm_dthrust = -1/w
    dm_dw = thrust/w**2

    return dr, dtheta, du, dv, [dm_dthrust, dm_dw]

