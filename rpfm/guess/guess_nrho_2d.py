"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.const import g0
from rpfm.utils.keplerian_orbit import TwoDimOrb
from rpfm.guess.guess_2d import HohmannTransfer, TwoDimGuess, PowConstRadius


class TwoDimLLO2NRHOGuess(TwoDimGuess):

    def __init__(self, gm, r, alt, rp, t, sc):

        TwoDimGuess.__init__(self, gm, r, sc)

        dep = TwoDimOrb(gm, a=(r + alt), e=0)
        arr = TwoDimOrb(gm, T=t, rp=rp)

        self.ht = HohmannTransfer(gm, dep, arr)

        self.pow = PowConstRadius(gm, dep.a, dep.vp, self.ht.transfer.vp, sc.m0, sc.T_max, sc.Isp)
        self.pow.compute_final_time_mass()

    def compute_trajectory(self, **kwargs):

        if 't' in kwargs:
            self.t = kwargs['t']
        elif 'nb_nodes' in kwargs:
            self.t = np.reshape(np.linspace(0.0, self.pow.tf+self.ht.tof, kwargs['nb_nodes']), (kwargs['nb_nodes'], 1))

        t_pow = self.t[self.t <= self.pow.tf]
        t_ht = self.t[self.t > self.pow.tf]

        self.pow.compute_states(t_pow)
        self.ht.compute_states(t_ht, self.pow.tf, theta0=self.pow.theta[-1, -1], m=self.pow.mf)

        self.states = np.vstack((self.pow.states, self.ht.states))
        self.controls = np.vstack((self.pow.controls, self.ht.controls))

        self.states[:, 1] = self.states[:, 1] - self.pow.theta[-1, -1]

        self.states[-1, 3] = self.ht.arrOrb.va
        self.states[-1, 4] = self.pow.mf*np.exp(-self.ht.dva/self.sc.Isp/g0)
        self.controls[-1, 0] = self.sc.T_max


class TwoDimNRHO2LLOGuess(TwoDimGuess):

    def __init__(self, gm, r, alt, rp, t, sc):

        TwoDimGuess.__init__(self, gm, r, sc)

        arr = TwoDimOrb(gm, a=(r + alt), e=0)
        dep = TwoDimOrb(gm, T=t, rp=rp)

        self.ht = HohmannTransfer(gm, dep, arr)
        self.m_hoh = self.sc.m0*np.exp(-self.ht.dva/self.sc.Isp/g0)

        self.pow = PowConstRadius(gm, arr.a, self.ht.transfer.vp, arr.vp, self.m_hoh, sc.T_max, sc.Isp)
        self.pow.compute_final_time_mass()

    def compute_trajectory(self, **kwargs):

        if 't' in kwargs:
            self.t = kwargs['t']
        elif 'nb_nodes' in kwargs:
            self.t = np.reshape(np.linspace(0.0, self.pow.tf+self.ht.tof, kwargs['nb_nodes']), (kwargs['nb_nodes'], 1))

        t_ht = self.t[self.t < self.ht.tof]
        t_pow = self.t[self.t >= self.ht.tof]

        self.ht.compute_states(t_ht, 0.0, theta0=np.pi, m=self.m_hoh)
        self.pow.compute_states(t_pow)

        self.states = np.vstack((self.ht.states, self.pow.states))
        self.controls = np.vstack((self.ht.controls, self.pow.controls))

        self.states[0, 3] = self.ht.depOrb.va
        self.states[0, 4] = self.sc.m0
        self.controls[0, 0] = self.sc.T_max


if __name__ == '__main__':

    from rpfm.utils.spacecraft import Spacecraft
    from rpfm.utils.primary import Moon
    from rpfm.plots.solutions import TwoDimSolPlot

    case = 'descent'

    moon = Moon()
    h = 100e3
    r_p = 3150e3
    T = 6.5655*86400
    s = Spacecraft(250., 4., g=moon.g)
    nb = (100, 1000)

    if case == 'ascent':
        tr = TwoDimLLO2NRHOGuess(moon.GM, moon.R, h, r_p, T, s)
        a = tr.ht.arrOrb.a
        e = tr.ht.arrOrb.e
        t1 = np.linspace(0.0, tr.pow.tf, nb[0])
        t2 = np.linspace(tr.pow.tf, tr.pow.tf + tr.ht.tof, nb[1] + 1)
    elif case == 'descent':
        pass
        tr = TwoDimNRHO2LLOGuess(moon.GM, moon.R, h, r_p, T, s)
        a = tr.ht.depOrb.a
        e = tr.ht.depOrb.e
        t1 = np.linspace(0.0, tr.ht.tof, nb[0])
        t2 = np.linspace(tr.ht.tof, tr.ht.tof + tr.pow.tf, nb[1] + 1)
    else:
        raise ValueError('case must be equal to ascent or descent')

    t_all = np.reshape(np.hstack((t1, t2[1:])), (np.sum(nb), 1))

    tr.compute_trajectory(t=t_all)

    p = TwoDimSolPlot(tr.R, tr.t, tr.states, tr.controls, kind=case, a=a, e=e)
    p.plot()
