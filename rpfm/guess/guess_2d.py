"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import root
from copy import deepcopy

from rpfm.utils.keplerian_orbit import KepOrb
from rpfm.utils.const import g0


class DeorbitBurn:

    def __init__(self, sc, dv):

        self.sc = deepcopy(sc)
        self.dv = dv

        self.sc.m0 = self.sc.m0*np.exp(-self.dv/self.sc.Isp/g0)
        self.dm = sc.m0 - self.sc.m0


class HohmannTransfer:

    def __init__(self, gm, ra, rp, kind='ascent'):

        self.GM = gm
        self.ra = ra
        self.rp = rp

        self.a = (ra + rp)/2
        self.e = (ra - rp)/(ra + rp)

        self.h = (gm*self.a*(1 - self.e**2))**0.5
        self.n = (gm/self.a**3)**0.5
        self.tof = np.pi/gm**0.5*self.a**1.5

        self.va_circ = (gm/ra)**0.5
        self.vp_circ = (gm/rp)**0.5

        self.va = (2*gm*rp/(ra*(ra + rp)))**0.5
        self.vp = (2*gm*ra/(rp*(ra + rp)))**0.5

        self.dva = self.va_circ - self.va
        self.dvp = self.vp - self.vp_circ

        if kind in ['ascent', 'descent']:
            self.kind = kind
        else:
            raise ValueError('kind must be either ascent or descent')

        self.r = self.theta = self.u = self.v = None
        self.states = None

    def compute_states(self, t, tp, theta0=0.0):

        nb_nodes = len(t)
        ea0 = np.reshape(np.linspace(0.0, np.pi, nb_nodes), (nb_nodes, 1))

        print("\nSolving Kepler's equation using Scipy root function")

        sol = root(KepOrb.kepler_eqn, ea0, args=(self.e, self.n, t, tp), tol=1e-15)

        print("output:", sol.message)

        ea = np.reshape(sol.x, (nb_nodes, 1))
        theta = 2*np.arctan(((1 + self.e)/(1 - self.e))**0.5*np.tan(ea/2))

        if self.kind == 'ascent':
            self.r = self.a*(1 - self.e**2)/(1 + self.e*np.cos(theta))
            self.u = self.GM/self.h*self.e*np.sin(theta)
            self.v = self.GM/self.h*(1 + self.e*np.cos(theta))
        elif self.kind == 'descent':
            self.r = self.a*(1 - self.e**2)/(1 + self.e*np.cos(theta + np.pi))
            self.u = self.GM/self.h*self.e*np.sin(theta + np.pi)
            self.v = self.GM/self.h*(1 + self.e*np.cos(theta + np.pi))

        self.theta = theta + theta0
        self.states = np.hstack((self.r, self.theta, self.u, self.v))

        return sol


class PowConstRadius:

    def __init__(self, gm, r, v0, vf, m0, thrust, isp, theta0=0.0, t0=0.0):

        self.GM = gm
        self.R = r
        self.v0 = v0
        self.vf = vf
        self.m0 = m0
        self.T = thrust
        self.Isp = isp
        self.theta0 = theta0
        self.t0 = t0

        self.tf = self.mf = None
        self.t = self.r = self.theta = self.u = self.v = self.m = self.alpha = None
        self.states = self.controls = None

    def compute_mass(self, t):

        m = self.m0 - (self.T/self.Isp/g0)*(t - self.t0)

        return m

    def compute_final_time_mass(self):

        print('\nComputing final time and mass for initial powered trajectory at constant R')

        sol = solve_ivp(fun=lambda v, t: self.dt_dv(v, t, self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                        t_span=(self.v0, self.vf), y0=[self.t0], rtol=1e-20, atol=1e-20)

        self.tf = sol.y[-1, -1]
        self.mf = self.compute_mass(self.tf)

        print('output:', sol.message)

        return sol

    def compute_states(self, t_eval):

        nb_nodes = len(t_eval)

        print('\nIntegrating ODEs for initial powered trajectory at constant R ')

        try:
            sol = solve_ivp(fun=lambda t, x: self.dx_dt(t, x, self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                            t_span=(self.t0, self.tf + 1e-6), y0=[self.theta0, self.v0], t_eval=t_eval,
                            rtol=1e-20, atol=1e-20)

            print('using Scipy solve_ivp function')

            self.t = np.reshape(sol.t, (nb_nodes, 1))
            self.theta = np.reshape(sol.y[0], (nb_nodes, 1))
            self.v = np.reshape(sol.y[1], (nb_nodes, 1))

        except ValueError:
            print('time vector not strictly monotonically increasing, using Scipy odeint function')

            y, sol = odeint(self.dx_dt, y0=[self.theta0, self.v0], t=t_eval,
                            args=(self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                            full_output=True, rtol=1e-12, atol=1e-12, tfirst=True)

            self.t = np.reshape(t_eval, (nb_nodes, 1))
            self.theta = np.reshape(y[:, 0], (nb_nodes, 1))
            self.v = np.reshape(y[:, 1], (nb_nodes, 1))

        print('output:', sol['message'])

        self.r = self.R*np.ones((nb_nodes, 1))
        self.u = np.zeros((nb_nodes, 1))
        self.m = self.compute_mass(self.t)

        v_dot = self.dv_dt(self.t, self.v, self.GM, self.R, self.m0, self.t0, self.T, self.Isp)
        num = self.GM/self.R**2 - self.v**2/self.R

        self.alpha = np.arctan2(num, v_dot)  # angles in [-pi, pi]
        self.alpha[self.alpha < -np.pi/2] = self.alpha[self.alpha < -np.pi/2] + 2*np.pi  # angles in [-pi/2, 3/2pi]

        self.states = np.hstack((self.r, self.theta, self.u, self.v, self.m))
        self.controls = np.hstack((self.T*np.ones((nb_nodes, 1)), self.alpha))

        return sol

    def dt_dv(self, v, t, gm, r, m0, t0, thrust, isp):

        dt_dv = 1/self.dv_dt(t, v, gm, r, m0, t0, thrust, isp)

        return dt_dv

    def dv_dt(self, t, v, gm, r, m0, t0, thrust, isp):

        dv_dt = ((thrust/(m0 - (thrust/isp/g0)*(t - t0)))**2 - (gm/r**2 - v**2/r)**2)**0.5

        if self.v0 < self.vf:
            return dv_dt
        else:
            return -dv_dt

    def dx_dt(self, t, x, gm, r, m0, t0, thrust, isp):

        x0_dot = x[1]/r
        x1_dot = self.dv_dt(t, x[1], gm, r, m0, t0, thrust, isp)

        return [x0_dot, x1_dot]


class TwoDimGuess:

    def __init__(self, gm, r, alt, sc):

        self.GM = gm
        self.R = r
        self.alt = alt
        self.sc = sc

        self.pow1 = self.pow2 = self.ht = None
        self.t = self.states = self.controls = None

    def compute_trajectory(self, fix_final=False, **kwargs):

        if 't' in kwargs:
            self.t = kwargs['t']
        elif 'nb_nodes' in kwargs:
            self.t = np.reshape(np.linspace(0.0, self.pow2.tf, kwargs['nb_nodes']), (kwargs['nb_nodes'], 1))

        t_pow1 = self.t[self.t <= self.pow1.tf]
        t_ht = self.t[(self.t > self.pow1.tf) & (self.t < (self.pow1.tf + self.ht.tof))]
        t_pow2 = self.t[self.t >= (self.pow1.tf + self.ht.tof)]

        self.pow1.compute_states(t_pow1)
        self.ht.compute_states(t_ht, self.pow1.tf, theta0=self.pow1.theta[-1, -1])

        nb_ht = len(t_ht)
        states_ht = np.hstack((self.ht.states, self.pow1.mf*np.ones((nb_ht, 1))))

        if self.ht.kind == 'ascent':
            controls_ht = np.zeros((nb_ht, 2))
        elif self.ht.kind == 'descent':
            controls_ht = np.hstack((np.zeros((nb_ht, 1)), np.pi*np.ones((nb_ht, 1))))
        else:
            raise ValueError('kind must be either ascent or descent')

        self.pow2.theta0 = self.ht.theta[-1, -1]
        self.pow2.compute_states(t_pow2)
        self.pow2.states[-1, 3] = self.pow2.vf

        self.states = np.vstack((self.pow1.states, states_ht, self.pow2.states))
        self.controls = np.vstack((self.pow1.controls, controls_ht, self.pow2.controls))

        if fix_final:
            self.states[:, 1] = self.states[:, 1] - self.states[-1, 1]


class TwoDimAscGuess(TwoDimGuess):

    def __init__(self, gm, r, alt, sc):

        TwoDimGuess.__init__(self, gm, r, alt, sc)

        self.ht = HohmannTransfer(gm, (r + alt), r)

        self.pow1 = PowConstRadius(gm, r, 0.0, self.ht.vp, sc.m0, sc.T_max, sc.Isp)
        self.pow1.compute_final_time_mass()

        self.pow2 = PowConstRadius(gm, (r + alt), self.ht.va, self.ht.va_circ, self.pow1.mf, sc.T_max, sc.Isp,
                                   t0=(self.pow1.tf + self.ht.tof))
        self.pow2.compute_final_time_mass()


class TwoDimDescGuess(TwoDimGuess):

    def __init__(self, gm, r, alt, sc):

        TwoDimGuess.__init__(self, gm, r, alt, sc)

        self.ht = HohmannTransfer(gm, (r + alt), r, kind='descent')

        self.pow1 = PowConstRadius(gm, (r + alt), self.ht.va_circ, self.ht.va, sc.m0, sc.T_max, sc.Isp)
        self.pow1.compute_final_time_mass()

        self.pow2 = PowConstRadius(gm, r, self.ht.vp, 0.0, self.pow1.mf, sc.T_max, sc.Isp,
                                   t0=(self.pow1.tf + self.ht.tof))
        self.pow2.compute_final_time_mass()


if __name__ == '__main__':

    from rpfm.utils.spacecraft import Spacecraft
    from rpfm.utils.primary import Moon
    from rpfm.plots.solutions import TwoDimSolPlot

    case = 'descent'

    moon = Moon()
    a = 15e3
    s = Spacecraft(450., 2.1, g=moon.g)
    nb = (100, 100, 50)

    if case == 'ascent':
        tr = TwoDimAscGuess(moon.GM, moon.R, a, s)
    elif case == 'descent':
        tr = TwoDimDescGuess(moon.GM, moon.R, a, s)
    else:
        raise ValueError('case must be equal to ascent or descent')

    t1 = np.linspace(0.0, tr.pow1.tf, nb[0])
    t2 = np.linspace(tr.pow1.tf, tr.pow1.tf + tr.ht.tof, nb[1] + 2)
    t3 = np.linspace(tr.pow1.tf + tr.ht.tof, tr.pow2.tf, nb[2])

    t_all = np.reshape(np.hstack((t1, t2[1:-1], t3)), (np.sum(nb), 1))

    tr.compute_trajectory(t=t_all, fix_final=True)

    p = TwoDimSolPlot(tr.R, tr.t, tr.states, tr.controls, kind=case)
    p.plot()
