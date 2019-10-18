"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.const import g0
from rpfm.guess.guess_2d import HohmannTransfer, PowConstRadius


class EllipticOrbParam:

    def __init__(self, gm, rp, t):

        self.rp_nrho = rp
        self.T_nrho = t * 86400  # period in seconds
        self.a_nrho = (gm * self.T_nrho ** 2 / 4 / np.pi ** 2) ** (1 / 3)
        self.e_nrho = 1 - self.rp_nrho / self.a_nrho
        self.ra_nrho = self.a_nrho * (1 + self.e_nrho)
        self.va_nrho = (gm / self.a_nrho * (1 - self.e_nrho) / (1 + self.e_nrho)) ** 0.5


class HohmannTransferEl(HohmannTransfer):

    def __init__(self, gm, rp_nrho, t_nrho, r_moon, alt_llo, kind='ascent'):

        self.ep = EllipticOrbParam(gm, rp_nrho, t_nrho)
        self.r_llo = r_moon + alt_llo

        HohmannTransfer.__init__(self, gm, self.ep.ra_nrho, self.r_llo, kind='ascent')

        self.dvp = self.vp - self.vp_circ
        self.dva = self.ep.va_nrho - self.va

        if kind in ['ascent', 'descent']:
            self.kind = kind
        else:
            raise ValueError('kind must be either ascent or descent')



class TwoDimGuessNRHO:

    def __init__(self, gm, r, alt, rp, t, sc):

        self.GM = gm
        self.R = r
        self.alt_llo = alt
        self.r_llo = r + alt
        self.vc_llo = (gm/self.r_llo)**0.5
        self.ep = EllipticOrbParam(gm, rp, t)

        self.sc = sc

        self.pow = self.ht = None
        self.t = self.states = self.controls = None


class TwoDimAscGuessNRHO(TwoDimGuessNRHO):

    def __init__(self, gm, r, alt, rp, t, sc):

        TwoDimGuessNRHO.__init__(self, gm, r, alt, rp, t, sc)

        self.ht = HohmannTransfer(gm, self.ep.ra_nrho, self.r_llo)

        self.pow = PowConstRadius(gm, (r + alt), self.vc_llo, self.ht.vp, sc.m0, sc.T_max, sc.Isp)
        self.pow.compute_final_time_mass()

    def compute_trajectory(self, fix_final=False, **kwargs):

        if 't' in kwargs:
            self.t = kwargs['t']
        elif 'nb_nodes' in kwargs:
            self.t = np.reshape(np.linspace(0.0, self.pow.tf + self.ht.tof, kwargs['nb_nodes']), (kwargs['nb_nodes'], 1))

        t_pow = self.t[self.t <= self.pow.tf]
        t_ht = self.t[self.t > self.pow.tf]

        self.pow.compute_states(t_pow)
        self.ht.compute_states(t_ht, self.pow.tf, theta0=self.pow.theta[-1, -1])

        nb_ht = len(t_ht)
        states_ht = np.hstack((self.ht.states, self.pow.mf*np.ones((nb_ht, 1))))
        controls_ht = np.zeros((nb_ht, 2))

        self.dva = self.ep.va_nrho - self.ht.va
        self.mf = self.pow.m[-1, -1]*np.exp(-self.dva/self.sc.Isp/g0)

        self.states = np.vstack((self.pow.states, states_ht))
        self.controls = np.vstack((self.pow.controls, controls_ht))
        self.states[-1, 3] = self.ep.va_nrho
        self.states[-1, 4] = self.mf
        self.controls[-1, 0] = self.sc.T_max

"""
class TwoDimDescGuess(TwoDimGuess):

    def __init__(self, gm, r, alt, sc):

        TwoDimGuess.__init__(self, gm, r, alt, sc)

        self.ht = HohmannTransfer(gm, (r + alt), r, kind='descent')

        self.pow1 = PowConstRadius(gm, (r + alt), self.ht.va_circ, self.ht.va, sc.m0, sc.T_max, sc.Isp)
        self.pow1.compute_final_time_mass()

        self.pow2 = PowConstRadius(gm, r, self.ht.vp, 0.0, self.pow1.mf, sc.T_max, sc.Isp,
                                   t0=(self.pow1.tf + self.ht.tof))
        self.pow2.compute_final_time_mass()
"""

if __name__ == '__main__':

    from rpfm.utils.spacecraft import Spacecraft
    from rpfm.utils.primary import Moon
    from rpfm.plots.solutions import TwoDimSolPlot

    case = 'ascent'

    moon = Moon()
    a = 100e3
    rp = 3150e3
    T = 6.5655
    s = Spacecraft(250., 4., g=moon.g)
    nb = (100, 1000)

    if case == 'ascent':
        tr = TwoDimAscGuessNRHO(moon.GM, moon.R, a, rp, T, s)
    elif case == 'descent':
        pass
        # tr = TwoDimDescGuessNRHO(moon.GM, moon.R, a, s)
    else:
        raise ValueError('case must be equal to ascent or descent')

    t1 = np.linspace(0.0, tr.pow.tf, nb[0])
    t2 = np.linspace(tr.pow.tf, tr.pow.tf + tr.ht.tof, nb[1]+1)

    t_all = np.reshape(np.hstack((t1, t2[1:])), (np.sum(nb), 1))

    tr.compute_trajectory(t=t_all, fix_final=False)

    p = TwoDimSolPlot(tr.R, tr.t, tr.states, tr.controls, kind=case)
    p.plot()
