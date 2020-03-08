"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import warnings
import numpy as np
from copy import deepcopy

from scipy.integrate import solve_ivp, odeint
from scipy.optimize import root

from rpfm.utils.keplerian_orbit import KepOrb, TwoDimOrb
from rpfm.utils.const import g0


class HohmannTransfer:

    def __init__(self, gm, dep, arr):

        self.GM = gm
        self.depOrb = dep
        self.arrOrb = arr

        if self.depOrb.a < self.arrOrb.a:  # ascent
            self.ra = self.arrOrb.ra
            self.rp = self.depOrb.rp
        else:  # descent
            self.ra = self.depOrb.ra
            self.rp = self.arrOrb.rp

        self.transfer = TwoDimOrb(self.GM, ra=self.ra, rp=self.rp)
        self.tof = self.transfer.T/2

        if self.depOrb.a < self.arrOrb.a:  # ascent
            self.dvp = self.transfer.vp - self.depOrb.vp
            self.dva = self.arrOrb.va - self.transfer.va
        else:  # descent
            self.dva = self.depOrb.va - self.transfer.va
            self.dvp = self.transfer.vp - self.arrOrb.vp

        self.r = self.theta = self.u = self.v = None
        self.states = self.controls = None

    def compute_trajectory(self, t, tp, theta0=0.0, m=1.0):

        nb_nodes = len(t)
        ea0 = np.reshape(np.linspace(0.0, np.pi, nb_nodes), (nb_nodes, 1))

        print("\nSolving Kepler's equation using Scipy root function")

        sol = root(KepOrb.kepler_eqn, ea0, args=(self.transfer.e, self.transfer.n, t, tp), tol=1e-12)

        print("output:", sol.message)

        ea = np.reshape(sol.x, (nb_nodes, 1))
        theta = 2*np.arctan(((1 + self.transfer.e)/(1 - self.transfer.e))**0.5*np.tan(ea/2))

        if self.depOrb.a < self.arrOrb.a:  # ascent
            self.r = self.transfer.a*(1 - self.transfer.e**2)/(1 + self.transfer.e*np.cos(theta))
            self.u = self.GM/self.transfer.h*self.transfer.e*np.sin(theta)
            self.v = self.GM/self.transfer.h*(1 + self.transfer.e*np.cos(theta))
            alpha = np.zeros((nb_nodes, 1))
        else:  # descent
            self.r = self.transfer.a*(1 - self.transfer.e**2)/(1 + self.transfer.e*np.cos(theta + np.pi))
            self.u = self.GM/self.transfer.h*self.transfer.e*np.sin(theta + np.pi)
            self.v = self.GM/self.transfer.h*(1 + self.transfer.e*np.cos(theta + np.pi))
            alpha = np.pi*np.ones((nb_nodes, 1))

        self.theta = theta + theta0
        self.states = np.hstack((self.r, self.theta, self.u, self.v, m*np.ones((nb_nodes, 1))))
        self.controls = np.hstack((np.zeros((nb_nodes, 1)), alpha))

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

        self.dv_inf = np.fabs(self.vf - self.v0)  # impulsive dV [m/s]

        self.tf = self.thetaf = self.mf = self.dv = None
        self.t = self.r = self.theta = self.u = self.v = self.m = self.alpha = None
        self.states = self.controls = None

    def compute_mass(self, t):

        m = self.m0 - (self.T/self.Isp/g0)*(t - self.t0)

        return m

    def compute_final_time_states(self):

        print('\nComputing final time for powered trajectory at constant R')

        sol_time = solve_ivp(fun=lambda v, t: self.dt_dv(v, t, self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                             t_span=(self.v0, self.vf), y0=[self.t0], rtol=1e-12, atol=1e-20)

        print('output:', sol_time.message)

        self.tf = sol_time.y[-1, -1]

        print('\nComputing final states for powered trajectory at constant R')

        sol_states = solve_ivp(fun=lambda t, x: self.dx_dt(t, x, self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                               t_span=(self.t0, self.tf), y0=[self.theta0, self.v0], rtol=1e-12, atol=1e-20)

        print('output:', sol_states.message)

        print('\nAchieved target speed: ', np.isclose(self.vf, sol_states.y[1, -1], rtol=1e-10, atol=1e-10))

        self.thetaf = sol_states.y[0, -1]
        self.mf = self.compute_mass(self.tf)
        self.dv = self.Isp * g0 * np.log(self.m0 / self.mf)

        return sol_time, sol_states

    def compute_trajectory(self, t_eval):

        nb_nodes = len(t_eval)

        print('\nIntegrating ODEs for initial powered trajectory at constant R ')

        try:
            sol = solve_ivp(fun=lambda t, x: self.dx_dt(t, x, self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                            t_span=(self.t0, self.tf + 1e-6), y0=[self.theta0, self.v0], t_eval=t_eval,
                            rtol=1e-12, atol=1e-20)

            print('using Scipy solve_ivp function')

            self.t = np.reshape(sol.t, (nb_nodes, 1))
            self.theta = np.reshape(sol.y[0], (nb_nodes, 1))
            self.v = np.reshape(sol.y[1], (nb_nodes, 1))

        except ValueError:
            print('time vector not strictly monotonically increasing, using Scipy odeint function')

            y, sol = odeint(self.dx_dt, y0=[self.theta0, self.v0], t=t_eval,
                            args=(self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                            full_output=True, rtol=1e-12, atol=1e-20, tfirst=True)

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

    def __str__(self):

        lines = ['\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.dv_inf, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Finite dV:', self.dv, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Burn time:', self.tf - self.t0, 's'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1.0 - self.mf/self.m0, '')]

        s = '\n'.join(lines)

        return s


class TwoDimGuess:

    def __init__(self, gm, r, dep, arr, sc):

        self.GM = gm
        self.R = r
        self.sc = sc

        self.ht = HohmannTransfer(gm, dep, arr)

        self.t = self.states = self.controls = None

    def __str__(self):

        lines = ['\n{:^50s}'.format('Departure Orbit:'),
                 self.ht.depOrb.__str__(),
                 '\n{:^50s}'.format('Arrival Orbit:'),
                 self.ht.arrOrb.__str__(),
                 '\n{:^50s}'.format('Hohmann transfer:'),
                 self.ht.transfer.__str__()]

        s = '\n'.join(lines)

        return s


class TwoDimLLOGuess(TwoDimGuess):

    def __init__(self, gm, r, dep, arr, sc):

        TwoDimGuess.__init__(self, gm, r, dep, arr, sc)

        self.pow1 = self.pow2 = None

    def compute_trajectory(self, fix_final=False, **kwargs):

        if 't_eval' in kwargs:
            self.t = kwargs['t_eval']
        elif 'nb_nodes' in kwargs:
            self.t = np.reshape(np.linspace(0.0, self.pow2.tf, kwargs['nb_nodes']), (kwargs['nb_nodes'], 1))

        t_pow1 = self.t[self.t <= self.pow1.tf]
        t_ht = self.t[(self.t > self.pow1.tf) & (self.t < (self.pow1.tf + self.ht.tof))]
        t_pow2 = self.t[self.t >= (self.pow1.tf + self.ht.tof)]

        self.pow1.compute_trajectory(t_pow1)
        self.ht.compute_trajectory(t_ht, self.pow1.tf, theta0=self.pow1.thetaf, m=self.pow1.mf)

        self.pow2.compute_trajectory(t_pow2)
        self.pow2.states[-1, 3] = self.pow2.vf

        self.states = np.vstack((self.pow1.states, self.ht.states, self.pow2.states))
        self.controls = np.vstack((self.pow1.controls, self.ht.controls, self.pow2.controls))

        if fix_final:
            self.states[:, 1] = self.states[:, 1] - self.pow2.thetaf

        if 'theta' in kwargs:
            self.states[:, 1] = self.states[:, 1] + kwargs['theta']

    def __str__(self):

        lines = [TwoDimGuess.__str__(self),
                 '\n{:^50s}'.format('Initial guess:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1.0 - self.pow2.mf/self.sc.m0, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight:', self.pow2.tf, 's'),
                 '\n{:^50s}'.format('Departure burn:'),
                 self.pow1.__str__(),
                 '\n{:^50s}'.format('Arrival burn:'),
                 self.pow2.__str__()]

        s = '\n'.join(lines)

        return s


class TwoDimAscGuess(TwoDimLLOGuess):

    def __init__(self, gm, r, alt, sc):

        dep = TwoDimOrb(gm, a=r, e=0)
        arr = TwoDimOrb(gm, a=(r + alt), e=0)

        TwoDimGuess.__init__(self, gm, r, dep, arr, sc)

        self.pow1 = PowConstRadius(gm, r, 0.0, self.ht.transfer.vp, sc.m0, sc.T_max, sc.Isp)
        self.pow1.compute_final_time_states()

        self.pow2 = PowConstRadius(gm, (r + alt), self.ht.transfer.va, self.ht.arrOrb.va, self.pow1.mf, sc.T_max,
                                   sc.Isp, t0=(self.pow1.tf + self.ht.tof), theta0=(self.pow1.thetaf + np.pi))
        self.pow2.compute_final_time_states()

        self.tf = self.pow2.tf


class TwoDimDescGuess(TwoDimLLOGuess):

    def __init__(self, gm, r, alt, sc):

        arr = TwoDimOrb(gm, a=r, e=0)
        dep = TwoDimOrb(gm, a=(r + alt), e=0)

        TwoDimGuess.__init__(self, gm, r, dep, arr, sc)

        self.pow1 = PowConstRadius(gm, (r + alt), self.ht.depOrb.va, self.ht.transfer.va, sc.m0, sc.T_max, sc.Isp)
        self.pow1.compute_final_time_states()

        self.pow2 = PowConstRadius(gm, r, self.ht.transfer.vp, 0.0, self.pow1.mf, sc.T_max, sc.Isp,
                                   t0=(self.pow1.tf + self.ht.tof), theta0=(self.pow1.thetaf + np.pi))
        self.pow2.compute_final_time_states()

        self.tf = self.pow2.tf


if __name__ == '__main__':

    from rpfm.utils.spacecraft import Spacecraft
    from rpfm.utils.primary import Moon
    from rpfm.plots.solutions import TwoDimSolPlot

    case = 'ascent'

    moon = Moon()
    sma = 100e3
    sat = Spacecraft(450., 2., g=moon.g)
    nb = (100, 100, 100)

    if case == 'ascent':
        tr = TwoDimAscGuess(moon.GM, moon.R, sma, sat)
    elif case == 'descent':
        tr = TwoDimDescGuess(moon.GM, moon.R, sma, sat)
    else:
        raise ValueError('case must be equal to ascent or descent')

    tht = np.linspace(tr.pow1.tf, tr.pow1.tf + tr.ht.tof, nb[1] + 2)
    t_all = np.reshape(np.hstack((np.linspace(0.0, tr.pow1.tf, nb[0]), tht[1:-1],
                                  np.linspace(tr.pow1.tf + tr.ht.tof, tr.pow2.tf, nb[2]))), (np.sum(nb), 1))

    tr.compute_trajectory(t_eval=t_all, fix_final=False)

    p = TwoDimSolPlot(tr.R, tr.t, tr.states, tr.controls, kind=case)
    p.plot()

    print(tr)
