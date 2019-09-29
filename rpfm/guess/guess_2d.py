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

    def __init__(self, gm, ra, rp):

        self.GM = gm
        self.ra = ra
        self.rp = rp

        self.a = (ra + rp)/2
        self.e = (ra - rp)/(ra + rp)

        self.h = (gm*self.a*(1 - self.e**2))**0.5
        self.tof = np.pi/gm**0.5*self.a**1.5

        self.va_circ = (gm/ra)**0.5
        self.vp_circ = (gm/rp)**0.5

        self.va = (2*gm*rp/(ra*(ra + rp)))**0.5
        self.vp = (2*gm*ra/(rp*(ra + rp)))**0.5

        self.dva = self.va_circ - self.va
        self.dvp = self.vp - self.vp_circ

        self.ea = self.theta = self.r = self.u = self.v = None

    def compute_states(self, t):

        nb_nodes = len(t)
        n = (self.GM/self.a**3)**0.5
        ea0 = np.reshape(np.linspace(0, np.pi, nb_nodes), (nb_nodes, 1))

        print("\nSolving Kepler's equation using Scipy root function")

        sol = root(KepOrb.kepler_eqn, ea0, args=(self.e, n, t, 0.0), tol=1e-20)

        print("output:", sol['message'])

        self.ea = np.reshape(sol.x, (nb_nodes, 1))
        self.theta = 2*np.arctan(((1 + self.e)/(1 - self.e))**0.5*np.tan(self.ea/2))
        self.r = self.a*(1 - self.e**2)/(1 + self.e*np.cos(self.theta))
        self.u = self.GM/self.h*self.e*np.sin(self.theta)
        self.v = self.GM/self.h*(1 + self.e*np.cos(self.theta))

        return sol


class PowConstRadius:

    def __init__(self, gm, r, v0, vf, m0, thrust, isp):

        self.GM = gm
        self.R = r
        self.v0 = v0
        self.vf = vf
        self.m0 = m0
        self.T = thrust
        self.Isp = isp

        self.tof = self.t = self.theta = self.v = self.m = self.alpha = None

    def compute_tof(self):

        print('\nComputing time of flight for initial powered trajectory at constant R using Scipy solve_ivp function')

        sol = solve_ivp(fun=lambda v, t: self.dt_dv(v, t, self.GM, self.R, self.m0, self.T, self.Isp),
                        t_span=(self.v0, self.vf), y0=[0], rtol=1e-20, atol=1e-20)

        self.tof = sol.y[-1, -1]

        print('output:', sol['message'])

        return sol

    def compute_states(self, t_eval):

        nb_nodes = len(t_eval)

        print('\nIntegrating ODEs for initial powered trajectory at constant R...')

        try:
            sol = solve_ivp(fun=lambda t, x: self.dx_dt(t, x, self.GM, self.R, self.m0, self.T, self.Isp),
                            t_span=(0, self.tof + 1e-6), y0=[0, self.v0], t_eval=t_eval, rtol=1e-20, atol=1e-20)

            print('using Scipy solve_ivp function')

            self.t = np.reshape(sol.t, (nb_nodes, 1))
            self.theta = np.reshape(sol.y[0], (nb_nodes, 1))
            self.v = np.reshape(sol.y[1], (nb_nodes, 1))

        except:
            print('time vector not strictly monotonically increasing, using Scipy odeint function')

            y, sol = odeint(self.dx_dt, y0=[0, self.v0], t=t_eval, args=(self.GM, self.R, self.m0, self.T, self.Isp),
                            full_output=True, rtol=1e-20, atol=1e-20, tfirst=True)

            self.t = np.reshape(t_eval, (nb_nodes, 1))
            self.theta = y[:, 0]
            self.v = y[:, 1]

        print('output:', sol['message'])

        m_flow = - self.T/self.Isp/g0
        self.m = self.m0 + m_flow*self.t

        v_dot = self.dv_dt(self.t, self.v, self.GM, self.R, self.m0, self.T, self.Isp)
        num = self.GM/self.R**2 - self.v**2/self.R
        self.alpha = np.arctan2(num, v_dot)  # angles in [-pi, pi]
        self.alpha[self.alpha < -np.pi/2] = self.alpha[self.alpha < -np.pi/2] + 2*np.pi  # angles in [-pi/2, 3/2pi]

        return sol

    def dt_dv(self, v, t, gm, r, m0, thrust, isp):

        dt_dv = 1/self.dv_dt(t, v, gm, r, m0, thrust, isp)

        return dt_dv

    def dv_dt(self, t, v, gm, r, m0, thrust, isp):

        dv_dt = ((thrust/(m0 - (thrust/isp/g0)*t))**2 - (gm/r**2 - v**2/r)**2)**0.5

        if self.v0 < self.vf:
            return dv_dt
        else:
            return -dv_dt

    def dx_dt(self, t, x, gm, r, m0, thrust, isp):

        x0_dot = x[1]/r
        x1_dot = self.dv_dt(t, x[1], gm, r, m0, thrust, isp)

        return [x0_dot, x1_dot]


class TwoDimGuess:

    def __init__(self, gm, r, alt, sc):

        self.GM = gm
        self.R = r
        self.alt = alt
        self.sc = sc

        self.ht = HohmannTransfer(gm, (r + alt), r)
        self.vp = self.ht.vp

        self.t = self.tof = None
        self.r = self.theta = self.u = self.v = self.m = None
        self.T = self.alpha = None
        self.states = self.controls = None

    def t_phases(self, t_switch, **kwargs):

        if 't' in kwargs:
            self.t = kwargs['t']
        elif 'nb_nodes' in kwargs:
            self.t = np.reshape(np.linspace(0.0, self.tof, kwargs['nb_nodes']), (kwargs['nb_nodes'], 1))

        t1 = self.t[self.t <= t_switch]
        t2 = self.t[self.t > t_switch]

        nb1 = len(t1)
        nb2 = len(t2)

        return t1, t2, nb1, nb2


class TwoDimAscGuess(TwoDimGuess):

    def __init__(self, gm, r, alt, sc):

        TwoDimGuess.__init__(self, gm, r, alt, sc)

        self.pcr = PowConstRadius(gm, r, 0.0, self.vp, sc.m0, sc.T_max, sc.Isp)
        self.pcr.compute_tof()

        self.tof = self.ht.tof + self.pcr.tof

    def compute_trajectory(self, **kwargs):

        t_pcr, t_ht, nb_pcr, nb_ht = self.t_phases(self.pcr.tof, **kwargs)

        self.pcr.compute_states(t_pcr)
        self.ht.compute_states(t_ht - self.pcr.tof)

        self.r = np.vstack((self.R*np.ones((nb_pcr, 1)), self.ht.r))
        self.theta = np.vstack((self.pcr.theta, (self.ht.theta + self.pcr.theta[-1])))
        self.u = np.vstack((np.zeros((nb_pcr, 1)), self.ht.u))
        self.v = np.vstack((self.pcr.v, self.ht.v))

        m_ht = self.pcr.m[-1, -1]
        m_final = m_ht*np.exp(-self.ht.dva/self.sc.Isp/g0)

        self.m = np.vstack((self.pcr.m, m_ht*np.ones(((nb_ht - 1), 1)), [m_final]))
        self.alpha = np.vstack((self.pcr.alpha, np.zeros((nb_ht, 1))))

        throttle = np.vstack((np.ones((nb_pcr, 1)), np.zeros(((nb_ht - 1), 1)), [1]))

        self.T = self.sc.T_max*throttle

        self.states = np.hstack((self.r, self.theta, self.u, self.v, self.m))
        self.controls = np.hstack((self.T, self.alpha))


class TwoDimDescGuess(TwoDimGuess):

    def __init__(self, gm, r, alt, sc):

        TwoDimGuess.__init__(self, gm, r, alt, sc)

        self.deorbit_burn = DeorbitBurn(sc, self.ht.dva)

        self.pcr = PowConstRadius(gm, r, self.vp, 0.0, self.deorbit_burn.sc.m0, sc.T_max, sc.Isp)
        self.pcr.compute_tof()

        self.tof = self.ht.tof + self.pcr.tof

    def compute_trajectory(self, **kwargs):

        t_ht, t_pcr, nb_ht, nb_pcr = self.t_phases(self.ht.tof, **kwargs)

        self.ht.compute_states(t_ht)
        self.pcr.compute_states(t_pcr - self.ht.tof)

        self.r = np.vstack((np.flip(self.ht.r), self.R*np.ones((nb_pcr, 1))))
        self.theta = np.vstack((self.ht.theta, (self.pcr.theta + self.ht.theta[-1])))
        self.u = np.vstack((np.flip(-1.0*self.ht.u), np.zeros((nb_pcr, 1))))
        self.v = np.vstack((np.flip(self.ht.v), self.pcr.v))

        self.m = np.vstack(([self.sc.m0], self.deorbit_burn.sc.m0*np.ones(((nb_ht - 1), 1)), self.pcr.m))
        self.alpha = np.vstack((np.pi*np.ones((nb_ht, 1)), self.pcr.alpha))

        throttle = np.vstack(([1.], np.zeros(((nb_ht - 1), 1)), np.ones((nb_pcr, 1))))

        self.T = self.sc.T_max*throttle

        self.states = np.hstack((self.r, self.theta, self.u, self.v, self.m))
        self.controls = np.hstack((self.T, self.alpha))


if __name__ == '__main__':

    from rpfm.utils.spacecraft import Spacecraft
    from rpfm.utils.primary import Moon
    from rpfm.plots.solutions import TwoDimSolPlot

    case = 'ascent'

    moon = Moon()
    a = 100e3
    s = Spacecraft(450., 2.1, g=moon.g)
    nb = 10

    if case == 'ascent':
        tr = TwoDimAscGuess(moon.GM, moon.R, a, s)
        t_all = np.reshape(np.hstack((np.linspace(0.0, tr.pcr.tof, nb),
                                      np.linspace(tr.pcr.tof, tr.ht.tof + tr.pcr.tof, nb)[1:])), (2*nb - 1, 1))
    elif case == 'descent':
        tr = TwoDimDescGuess(moon.GM, moon.R, a, s)
        t_all = np.reshape(np.hstack((np.linspace(0.0, tr.ht.tof, nb),
                                      np.linspace(tr.ht.tof, tr.pcr.tof + tr.ht.tof, nb)[1:])), (2*nb - 1, 1))
    else:
        raise ValueError('case must be equal to ascent or descent')

    tr.compute_trajectory(t=t_all)

    p = TwoDimSolPlot(tr.R, tr.t, tr.states, tr.controls, kind=case)
    p.plot()
