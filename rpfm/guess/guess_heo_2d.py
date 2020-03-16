"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from copy import deepcopy
from rpfm.utils.keplerian_orbit import TwoDimOrb
from rpfm.guess.guess_2d import TwoDimGuess, PowConstRadius, TwoDimLLOGuess
from rpfm.utils.spacecraft import ImpulsiveBurn


class TwoDimHEOGuess(TwoDimGuess):

    def __init__(self, gm, r, dep, arr, sc):

        TwoDimGuess.__init__(self, gm, r, dep, arr, sc)  # set central body, spacecraft, Hohmann transfer

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

        self.insertion_burn = None

    def compute_trajectory(self, fix_final=True, throttle=True, **kwargs):

        TwoDimHEOGuess.compute_trajectory(self, **kwargs)

        t_pow = self.t[self.t <= self.pow.tf]
        t_ht = self.t[self.t > self.pow.tf]

        self.pow.compute_trajectory(t_pow)

        if np.size(t_ht) > 0:
            self.ht.compute_trajectory(t_ht, self.pow.tf, theta0=self.pow.thetaf, m=self.pow.mf)

            self.states = np.vstack((self.pow.states, self.ht.states))
            self.controls = np.vstack((self.pow.controls, self.ht.controls))
        else:
            self.states = self.pow.states
            self.controls = self.pow.controls

        if fix_final:
            self.states[:, 1] = self.states[:, 1] - self.pow.thetaf  # final true anomaly equal to pi

        # insertion burn at the NRHO aposelene
        sc = deepcopy(self.sc)
        sc.m0 = self.pow.mf

        self.insertion_burn = ImpulsiveBurn(sc, self.ht.dva)

        if throttle:
            self.states[-1, 3] = self.ht.arrOrb.va
            self.states[-1, 4] = self.insertion_burn.mf
            self.controls[-1, 0] = self.sc.T_max

    def __str__(self):

        lines = [TwoDimHEOGuess.__str__(self),
                 '\n{:^50s}'.format('Initial guess:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1.0 - self.insertion_burn.mf/self.sc.m0, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight:', self.tf/86400, 'days'),
                 '\n{:^50s}'.format('Departure burn:'),
                 self.pow.__str__(),
                 '\n{:^50s}'.format('Injection burn:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.ht.dva, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', self.insertion_burn.dm/self.sc.m0, '')]

        s = '\n'.join(lines)

        return s


class TwoDimHEO2LLOGuess(TwoDimHEOGuess):

    def __init__(self, gm, r, alt, rp, t, sc):

        arr = TwoDimOrb(gm, a=(r + alt), e=0)
        dep = TwoDimOrb(gm, T=t, rp=rp)

        TwoDimGuess.__init__(self, gm, r, dep, arr, sc)

        self.deorbit_burn = ImpulsiveBurn(sc, self.ht.dva)
        self.pow = PowConstRadius(gm, arr.a, self.ht.transfer.vp, arr.vp, self.deorbit_burn.mf, sc.T_max, sc.Isp,
                                  t0=self.ht.tof)
        self.pow.compute_final_time_states()

        self.tf = self.pow.tf

    def compute_trajectory(self, **kwargs):

        TwoDimHEOGuess.compute_trajectory(self, **kwargs)

        t_ht = self.t[self.t <= self.ht.tof]
        t_pow = self.t[self.t > self.ht.tof]

        self.ht.compute_trajectory(t_ht, 0.0, theta0=np.pi, m=self.deorbit_burn.mf)
        # self.ht.states[:, 1] = self.ht.states[:, 1] - np.pi  # adjust true anomaly

        self.pow.compute_trajectory(t_pow)
        self.pow.states[:, 1] = self.pow.states[:, 1] + self.ht.states[-1, 1]  # adjust true anomaly

        self.states = np.vstack((self.ht.states, self.pow.states))
        self.controls = np.vstack((self.ht.controls, self.pow.controls))

        # departure burn on initial HEO
        self.states[0, 3] = self.ht.depOrb.va
        self.states[0, 4] = self.sc.m0
        self.controls[0, 0] = self.sc.T_max

    def __str__(self):

        lines = [TwoDimHEOGuess.__str__(self),
                 '\n{:^50s}'.format('Initial guess:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1.0 - self.states[-1, -1]/self.sc.m0, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight:', self.tf/86400, 'days'),
                 '\n{:^50s}'.format('Deorbit burn:'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.ht.dva, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:',
                                                 (self.sc.m0 - self.deorbit_burn.mf)/self.sc.m0, ''),
                 '\n{:^50s}'.format('Injection burn:'),
                 self.pow.__str__()]

        s = '\n'.join(lines)

        return s


class TwoDim2PhasesLLO2HEOGuess(TwoDimLLOGuess):

    def __init__(self, gm, r, alt, rp, t, sc):

        dep = TwoDimOrb(gm, a=(r + alt), e=0)
        arr = TwoDimOrb(gm, T=t, rp=rp)

        TwoDimGuess.__init__(self, gm, r, dep, arr, sc)

        self.pow1 = PowConstRadius(gm, (r + alt), dep.vp, self.ht.transfer.vp, sc.m0, sc.T_max, sc.Isp)
        self.pow1.compute_final_time_states()

        self.pow2 = PowConstRadius(gm, arr.ra, self.ht.transfer.va, arr.va, self.pow1.mf, sc.T_max, sc.Isp)
        self.pow2.compute_final_time_states()

    def compute_trajectory(self, fix_final=False, **kwargs):

        if ('t1' in kwargs) and ('t2' in kwargs):
            t_pow1 = kwargs['t1']
            t_pow2 = kwargs['t2']
        elif ('nb1' in kwargs) and ('nb2' in kwargs):
            t_pow1 = np.linspace(0.0, self.pow1.tf, kwargs['nb1'])
            t_pow2 = np.linspace(0.0, self.pow2.tf, kwargs['nb2'])
        else:
            raise AttributeError('Either t1, t2 or nb1, nb2 must be provided')

        self.pow1.compute_trajectory(t_pow1.flatten())
        self.pow2.compute_trajectory(t_pow2.flatten())

        self.pow2.states[-1, 0] = self.pow2.R
        self.pow2.states[-1, 3] = self.pow2.vf

        if fix_final:
            self.pow1.states[:, 1] = self.pow1.states[:, 1] - self.pow1.thetaf
            self.pow2.states[:, 1] = self.pow2.states[:, 1] - self.pow2.thetaf

    def __str__(self):

        lines = [TwoDimGuess.__str__(self),
                 '\n{:^50s}'.format('Departure burn:'),
                 self.pow1.__str__(),
                 '\n{:^50s}'.format('Arrival burn:'),
                 self.pow2.__str__()]

        s = '\n'.join(lines)

        return s


class TwoDim3PhasesLLO2HEOGuess(TwoDimLLOGuess):

    def __init__(self, gm, r, alt, rp, t, sc):

        dep = TwoDimOrb(gm, a=(r + alt), e=0)
        arr = TwoDimOrb(gm, T=t, rp=rp)

        TwoDimGuess.__init__(self, gm, r, dep, arr, sc)

        self.pow1 = PowConstRadius(gm, (r + alt), dep.vp, self.ht.transfer.vp, sc.m0, sc.T_max, sc.Isp)
        self.pow1.compute_final_time_states()

        self.pow2 = PowConstRadius(gm, arr.ra, self.ht.transfer.va, arr.va, self.pow1.mf, sc.T_max,
                                   sc.Isp, t0=(self.pow1.tf + self.ht.tof), theta0=(self.pow1.thetaf + np.pi))
        self.pow2.compute_final_time_states()

        self.tf = self.pow2.tf


if __name__ == '__main__':

    from rpfm.utils.spacecraft import Spacecraft
    from rpfm.utils.primary import Moon
    from rpfm.plots.solutions import TwoDimSolPlot

    case = 'ascent'
    moon = Moon()
    h = 100e3
    r_p = 3150e3
    T = 6.5655*86400
    sat = Spacecraft(450, 2.1, g=moon.g)
    nb = (20, 1000)

    if case == 'ascent':
        tr = TwoDimLLO2HEOGuess(moon.GM, moon.R, h, r_p, T, sat)
        a = tr.ht.arrOrb.a
        e = tr.ht.arrOrb.e
        t_all = np.reshape(np.hstack((np.linspace(0.0, tr.pow.tf, nb[0]),
                                      np.linspace(tr.pow.tf, tr.pow.tf + tr.ht.tof, nb[1] + 1)[1:])), (np.sum(nb), 1))
    elif case == 'descent':
        tr = TwoDimHEO2LLOGuess(moon.GM, moon.R, h, r_p, T, sat)
        a = tr.ht.depOrb.a
        e = tr.ht.depOrb.e
        t_all = np.reshape(np.hstack((np.linspace(0.0, tr.ht.tof, nb[1]),
                                      np.linspace(tr.ht.tof, tr.pow.tf, nb[0] + 1)[1:])), (np.sum(nb), 1))
    elif case == '3p':
        case = 'ascent'
        tr = TwoDim3PhasesLLO2HEOGuess(moon.GM, moon.R, h, r_p, T, sat)
        a = tr.ht.arrOrb.a
        e = tr.ht.arrOrb.e
        t_all = np.reshape(np.hstack((np.linspace(0.0, tr.pow1.tf, nb[0]),
                                      np.linspace(tr.pow1.tf, tr.pow1.tf + tr.ht.tof, nb[1] + 2)[1:-1],
                                      np.linspace(tr.pow1.tf + tr.ht.tof, tr.pow2.tf, nb[2]))), (np.sum(nb), 1))
    else:
        raise ValueError('case must be equal to ascent or descent')

    tr.compute_trajectory(t_eval=t_all, fix_final=True, throttle=True)

    p = TwoDimSolPlot(tr.R, tr.t, tr.states, tr.controls, kind=case, a=a, e=e)
    p.plot()

    print(tr)
