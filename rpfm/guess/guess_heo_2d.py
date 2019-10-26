"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.utils.const import g0
from rpfm.utils.keplerian_orbit import TwoDimOrb
from rpfm.guess.guess_2d import TwoDimGuess, PowConstRadius


class TwoDimHEOGuess(TwoDimGuess):

    def __init__(self, gm, r, dep, arr, sc):

        TwoDimGuess.__init__(self, gm, r, dep, arr, sc)

        self.pow = None

    def compute_trajectory(self, **kwargs):

        if 't_eval' in kwargs:
            self.t = kwargs['t_eval']
        elif 'nb_nodes' in kwargs:
            self.t = np.reshape(np.linspace(0.0, self.pow.tf + self.ht.tof, kwargs['nb_nodes']),
                                (kwargs['nb_nodes'], 1))


class TwoDimLLO2HEOGuess(TwoDimHEOGuess):

    def __init__(self, gm, r, alt, rp, t, sc):

        dep = TwoDimOrb(gm, a=(r + alt), e=0)
        arr = TwoDimOrb(gm, T=t, rp=rp)

        TwoDimHEOGuess.__init__(self, gm, r, dep, arr, sc)

        self.pow = PowConstRadius(gm, dep.a, dep.vp, self.ht.transfer.vp, sc.m0, sc.T_max, sc.Isp)
        self.pow.compute_final_time_states()

        self.tf = self.pow.tf + self.ht.tof

    def compute_trajectory(self, **kwargs):

        TwoDimHEOGuess.compute_trajectory(self, **kwargs)

        t_pow = self.t[self.t <= self.pow.tf]
        t_ht = self.t[self.t > self.pow.tf]

        self.pow.compute_trajectory(t_pow)
        self.ht.compute_trajectory(t_ht, self.pow.tf, theta0=self.pow.thetaf, m=self.pow.mf)

        self.states = np.vstack((self.pow.states, self.ht.states))
        self.controls = np.vstack((self.pow.controls, self.ht.controls))

        self.states[:, 1] = self.states[:, 1] - self.pow.thetaf  # final true anomaly equal to pi

        # injection burn at the NRHO aposelene
        self.states[-1, 3] = self.ht.arrOrb.va
        self.states[-1, 4] = self.pow.mf*np.exp(-self.ht.dva/self.sc.Isp/g0)
        self.controls[-1, 0] = self.sc.T_max

    def __str__(self):

        lines = [TwoDimHEOGuess.__str__(self),
                 '\n{:^50s}'.format('Initial guess:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:',
                                                   (self.sc.m0 - self.states[-1, -1]) / self.sc.m0, ''),
                 '\n{:^50s}'.format('Departure burn:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.pow.dv_inf, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Finite dV:', self.pow.dv, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', (self.pow.m0 - self.pow.mf)/self.sc.m0, ''),
                 '\n{:^50s}'.format('Injection burn:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.ht.dva, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:',
                                                 (self.pow.mf - self.states[-1, -1])/self.sc.m0, '')]

        s = '\n'.join(lines)

        return s


class TwoDimHEO2LLOGuess(TwoDimHEOGuess):

    def __init__(self, gm, r, alt, rp, t, sc):

        arr = TwoDimOrb(gm, a=(r + alt), e=0)
        dep = TwoDimOrb(gm, T=t, rp=rp)

        TwoDimGuess.__init__(self, gm, r, dep, arr, sc)

        self.m_ht = self.sc.m0*np.exp(-self.ht.dva/self.sc.Isp/g0)

        self.pow = PowConstRadius(gm, arr.a, self.ht.transfer.vp, arr.vp, self.m_ht, sc.T_max, sc.Isp, t0=self.ht.tof)
        self.pow.compute_final_time_states()

        self.tf = self.pow.tf

    def compute_trajectory(self, **kwargs):

        TwoDimHEOGuess.compute_trajectory(self, **kwargs)

        t_ht = self.t[self.t < self.ht.tof]
        t_pow = self.t[self.t >= self.ht.tof]

        self.ht.compute_states(t_ht, 0.0, theta0=np.pi, m=self.m_ht)
        self.pow.compute_trajectory(t_pow)

        self.states = np.vstack((self.ht.states, self.pow.states))
        self.controls = np.vstack((self.ht.controls, self.pow.controls))

        # departure burn on initial LLO
        self.states[0, 3] = self.ht.depOrb.va
        self.states[0, 4] = self.sc.m0
        self.controls[0, 0] = self.sc.T_max

    def __str__(self):

        lines = [TwoDimHEOGuess.__str__(self),
                 '\n{:^50s}'.format('Initial guess:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:',
                                                   (self.sc.m0 - self.states[-1, -1]) / self.sc.m0, ''),
                 '\n{:^50s}'.format('Deorbit burn:'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.ht.dva, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', (self.sc.m0 - self.m_ht)/self.sc.m0, ''),
                 '\n{:^50s}'.format('Injection burn:'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.pow.dv_inf, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Finite dV:', self.pow.dv, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', (self.pow.m0 - self.pow.mf)/self.sc.m0, '')]

        s = '\n'.join(lines)

        return s


if __name__ == '__main__':

    from rpfm.utils.spacecraft import Spacecraft
    from rpfm.utils.primary import Moon
    from rpfm.plots.solutions import TwoDimSolPlot

    case = 'ascent'

    moon = Moon()
    h = 0.
    r_p = 3150e3
    T = 6.5655*86400
    sat = Spacecraft(450, 100., g=moon.g)
    nb = (100, 100)

    if case == 'ascent':
        tr = TwoDimLLO2HEOGuess(moon.GM, moon.R, h, r_p, T, sat)
        a = tr.ht.arrOrb.a
        e = tr.ht.arrOrb.e
        t1 = np.linspace(0.0, tr.pow.tf, nb[0])
        t2 = np.linspace(tr.pow.tf, tr.pow.tf + tr.ht.tof, nb[1] + 1)
    elif case == 'descent':
        tr = TwoDimHEO2LLOGuess(moon.GM, moon.R, h, r_p, T, sat)
        a = tr.ht.depOrb.a
        e = tr.ht.depOrb.e
        t1 = np.linspace(0.0, tr.ht.tof, nb[0])
        t2 = np.linspace(tr.ht.tof, tr.pow.tf, nb[1] + 1)
    else:
        raise ValueError('case must be equal to ascent or descent')

    t_all = np.reshape(np.hstack((t1, t2[1:])), (np.sum(nb), 1))
    # t_all = np.reshape(np.linspace(0.0, tr.pow.tf + tr.ht.tof, 401), (401, 1))

    tr.compute_trajectory(t_eval=t_all)

    p = TwoDimSolPlot(tr.R, tr.t, tr.states, tr.controls, kind=case, a=a, e=e)
    p.plot()

    print(tr)
