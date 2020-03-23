"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from scipy.integrate import solve_ivp, odeint
from scipy.optimize import newton

from latom.utils.keplerian_orbit import KepOrb, TwoDimOrb
from latom.utils.const import g0


class HohmannTransfer:
    """`HohmannTransfer` class implements a two-dimensional Hohmann transfer between two coplanar, circular orbits in
    the Keplerian two-body approximation.

    Parameters
    ----------
    gm : float
        Standard gravitational parameter of the central attracting body [m^3/s^2]
    dep : TwoDimOrb
        `TwoDimOrb` class instance representing the departure orbit
    arr : TwoDimOrb
        `TwoDimOrb` class instance representing the arrival orbit

    Attributes
    ----------
    GM : float
        Standard gravitational parameter of the central attracting body [m^3/s^2]
    dep : TwoDimOrb
        `TwoDimOrb` class instance representing the departure orbit
    arr : TwoDimOrb
        `TwoDimOrb` class instance representing the arrival orbit
    ra : float
        Transfer orbit apoapsis radius [m]
    rp : float
        Transfer orbit periapsis radius [m]
    transfer : TwoDimOrb
        `TwoDimOrb` class instance representing the transfer orbit
    tof : float
        Hohmann transfer time of flight corresponding to half of the `transfer` period [s]
    dvp : float
        Impulsive delta-V at periapsis [m/s]
    dva : float
        Impulsive delta-V at apoapsis [m/s]
    r : ndarray
        Radius time series for the transfer arc [m]
    theta : ndarray
        Angle time series for the transfer arc [rad]
    u : ndarray
        Radial velocity time series for the transfer arc [m/s]
    v : ndarray
        Tangential velocity time series for the transfer arc [m/s]
    states : ndarray
        States variables time series for the transfer arc as [r, theta, u, v, m]
    controls : ndarray
        Control variables time series for the transfer arc as [thrust, alpha]

    """

    def __init__(self, gm, dep, arr):
        """Initializes `HohmannTransfer` class. """

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
        self.tof = self.transfer.T / 2

        if self.depOrb.a < self.arrOrb.a:  # ascent
            self.dvp = self.transfer.vp - self.depOrb.vp
            self.dva = self.arrOrb.va - self.transfer.va
        else:  # descent
            self.dva = self.depOrb.va - self.transfer.va
            self.dvp = self.transfer.vp - self.arrOrb.vp

        self.r = self.theta = self.u = self.v = None
        self.states = self.controls = None

    def compute_trajectory(self, t, t0=0.0, theta0=0.0, m=1.0):
        """Computes the states and control time series along the transfer trajectory.

        Parameters
        ----------
        t : ndarray
            Time vector [s]
        t0 : float
            Initial time [s]
        theta0 : float, optional
            Initial angle. Default is ``0.0`` [rad]
        m : float
            Spacecraft mass. Default is ``1.0`` [kg]

        Returns
        -------
        root : ndarray
            Solution of the Kepler time of flight equation for the eccentric anomaly in function of the true anomaly

        """

        t = t.flatten() - t0  # flattened time array [s]
        nb_nodes = np.size(t)  # number of points in which the states are computed

        if self.depOrb.a < self.arrOrb.a:  # ascent
            ea0 = np.linspace(0.0, np.pi, nb_nodes)
            tp = 0.0  # time at periapsis passage [s]
        else:  # descent
            ea0 = np.linspace(-np.pi, 0.0, nb_nodes)
            tp = self.tof  # time at periapsis passage [s]

        root, converged, zero_der = newton(KepOrb.kepler_eqn, ea0, fprime=KepOrb.kepler_eqn_prime,
                                           args=(self.transfer.e, self.transfer.n, t, tp), tol=1e-12, maxiter=100,
                                           fprime2=KepOrb.kepler_eqn_second, rtol=0.0, full_output=True, disp=True)

        ea = np.reshape(root, (nb_nodes, 1))  # eccentric anomaly in [-pi, pi]
        theta = 2 * np.arctan(((1 + self.transfer.e) / (1 - self.transfer.e)) ** 0.5 * np.tan(ea / 2))  # [-pi, pi]

        self.r = self.transfer.a * (1 - self.transfer.e ** 2) / (1 + self.transfer.e * np.cos(theta))
        self.u = self.GM / self.transfer.h * self.transfer.e * np.sin(theta)
        self.v = self.GM / self.transfer.h * (1 + self.transfer.e * np.cos(theta))

        if self.depOrb.a < self.arrOrb.a:  # ascent
            alpha = np.zeros((nb_nodes, 1))
        else:  # descent
            theta = theta - theta[0, 0]  # true anomaly in [0, 2pi]
            alpha = np.pi * np.ones((nb_nodes, 1))

        self.theta = theta + theta0
        self.states = np.hstack((self.r, self.theta, self.u, self.v, m * np.ones((nb_nodes, 1))))
        self.controls = np.hstack((np.zeros((nb_nodes, 1)), alpha))

        return root


class PowConstRadius:
    """`PowConstRadius` class implements a two-dimensional powered phase at constant radius from the central attractive
    body.

    The trajectory is modeled in the restricted two-body problem framework and the spacecraft engines are supposed to
    deliver a constant thrust magnitude across the whole phase. Since the radius `r` is constant, the radial velocity
    `u` is always null.

    Parameters
    ----------
    gm : float
        Central body standard gravitational parameter [m^3/s^2]
    r0 : float
        Radius [m]
    v0 : float
        Initial tangential velocity [m/s]
    vf : float
        Final tangential velocity [m/s]
    m0 : float
        Initial spacecraft mass [kg]
    thrust : float
        Thrust magnitude [N]
    isp : float
        Specific impulse [s]
    theta0 : float
        Initial angle [rad]
    t0 : float
        Initial time [rad]

    Attributes
    ----------
    GM : float
        Central body standard gravitational parameter [m^3/s^2]
    R : float
        Radius [m]
    v0 : float
        Initial tangential velocity [m/s]
    vf : float
        Final tangential velocity [m/s]
    m0 : float
        Initial spacecraft mass [kg]
    T : float
        Thrust magnitude [N]
    Isp : float
        Specific impulse [s]
    theta0 : float
        Initial angle [rad]
    t0 : float
        Initial time [rad]
    dv_inf : float
        Delta-V required to accelerate the spacecraft from `v0` to `vf` assuming an impulsive burn [m/s]
    tf : float
        Final time [s]
    thetaf : float
        Final angle [rad]
    mf : float
        Final spacecraft mass [kg]
    dv : float
        Delta-V required to accelerate the spacecraft from `v0` to `vf` taking into account a finite thrust magnitude
        equal to `T` [m/s]
    t : ndarray
        Time vector [s]
    r : ndarray
        Radius time series [m]
    theta : ndarray
        Angle time series [rad]
    u : ndarray
        Radial velocity time series [m/s]
    v : ndarray
        Tangential velocity time series [m/s]
    m : ndarray
        Spacecraft mass time series [kg]
    alpha : ndarray
        Thrust direction time series [rad]
    states : ndarray
        States variables time series as `[r, theta, u, v, m]`
    controls : ndarray
        Controls variables time series as `[thrust, alpha]`

    """

    def __init__(self, gm, r0, v0, vf, m0, thrust, isp, theta0=0.0, t0=0.0):

        self.GM = gm
        self.R = r0
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
        """Compute the spacecraft mass time series.

        Parameters
        ----------
        t : ndarray
            Time vector [s]

        Returns
        -------
        m : ndarray
            Spacecraft mass time series [kg]

        """

        m = self.m0 - (self.T / self.Isp / g0) * (t - self.t0)

        return m

    def compute_final_time_states(self):
        """Compute the required time of flight and the final spacecraft states.

        Returns
        -------
        sol_time : tuple
            Solution of the Initial Value Problem (IVP) for the final time.
            See scipy.integrate.solve_ivp for more details
        sol_states : tuple
            Solution of the Initial Value Problem (IVP) for the final angle and speed.
            See scipy.integrate.solve_ivp for more details

        """

        print('\nComputing final time for powered trajectory at constant R')

        sol_time = solve_ivp(fun=lambda v, t: self.dt_dv(v, t, self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                             t_span=(self.v0, self.vf), y0=[self.t0], rtol=1e-12, atol=1e-20)

        print('output:', sol_time.message)

        self.tf = sol_time.y[-1, -1]

        print('\nComputing final states for powered trajectory at constant R')

        sol_states = solve_ivp(fun=lambda t, x: self.dx_dt(t, x, self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                               t_span=(self.t0, self.tf), y0=[self.theta0, self.v0], rtol=1e-12, atol=1e-20)

        print('output:', sol_states.message)

        print('\nAchieved target speed: ', np.isclose(self.vf, sol_states.y[1, -1], rtol=1e-8, atol=1e-8))

        self.thetaf = sol_states.y[0, -1]  # final angle [rad]
        self.mf = self.compute_mass(self.tf)  # final spacecraft mass [kg]
        self.dv = self.Isp * g0 * np.log(self.m0 / self.mf)  # finite Delta-V [m/s]

        return sol_time, sol_states

    def compute_trajectory(self, t_eval):
        """Compute the spacecraft states and controls variables time series across the powered phase.

        Parameters
        ----------
        t_eval : ndarray
            Time vector in which the states and controls are provided [s]

        Returns
        -------
        sol : tuple or dict
            Solution of the Initial Value Problem (IVP) for the final time.
            See scipy.integrate.solve_ivp or scipy.integrate.odeint for more details

        """

        nb_nodes = len(t_eval)

        print('\nIntegrating ODEs for initial powered trajectory at constant R ')

        try:
            print('using Scipy solve_ivp function')

            sol = solve_ivp(fun=lambda t, x: self.dx_dt(t, x, self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                            t_span=(self.t0, self.tf + 1e-6), y0=[self.theta0, self.v0], t_eval=t_eval,
                            rtol=1e-12, atol=1e-20)

            self.t = np.reshape(sol.t, (nb_nodes, 1))
            self.theta = np.reshape(sol.y[0], (nb_nodes, 1))
            self.v = np.reshape(sol.y[1], (nb_nodes, 1))

        except ValueError:
            print('time vector not strictly monotonically increasing, using Scipy odeint function')

            y, sol = odeint(self.dx_dt, y0=[self.theta0, self.v0], t=np.hstack(([self.t0], t_eval)),
                            args=(self.GM, self.R, self.m0, self.t0, self.T, self.Isp),
                            full_output=True, rtol=1e-12, atol=1e-20, tfirst=True)

            self.t = np.reshape(t_eval, (nb_nodes, 1))
            self.theta = np.reshape(y[1:, 0], (nb_nodes, 1))
            self.v = np.reshape(y[1:, 1], (nb_nodes, 1))

        print('output:', sol['message'])

        self.r = self.R * np.ones((nb_nodes, 1))
        self.u = np.zeros((nb_nodes, 1))
        self.m = self.compute_mass(self.t)

        v_dot = self.dv_dt(self.t, self.v, self.GM, self.R, self.m0, self.t0, self.T, self.Isp)
        num = self.GM / self.R ** 2 - self.v ** 2 / self.R

        self.alpha = np.arctan2(num, v_dot)  # angles in [-pi, pi]
        self.alpha[self.alpha < -np.pi / 2] =\
            self.alpha[self.alpha < -np.pi / 2] + 2 * np.pi  # angles in [-pi/2, 3/2pi]

        self.states = np.hstack((self.r, self.theta, self.u, self.v, self.m))
        self.controls = np.hstack((self.T * np.ones((nb_nodes, 1)), self.alpha))

        return sol

    def dt_dv(self, v, t, gm, r, m0, t0, thrust, isp):
        """ODE for the first derivative of the time `t` wrt the tangential velocity `v`.

        Parameters
        ----------
        v : float or ndarray
            Tangential velocity time series [m/s]
        t : float or ndarray
            Time vector [s]
        gm : float
            Central body standard gravitational parameter [m^3/s^2]
        r : float or ndarray
            Radius time series [m]
        m0 : float
            Initial spacecraft mass [kg]
        t0 : float
            Initial time [s]
        thrust : float
            Thrust magnitude [N]
        isp : float
            Specific impulse [s]

        Returns
        -------
        dt_dv : float or ndarray
            First derivative of `t` wrt `v` [s^2/m]

        """

        dt_dv = 1 / self.dv_dt(t, v, gm, r, m0, t0, thrust, isp)

        return dt_dv

    def dv_dt(self, t, v, gm, r, m0, t0, thrust, isp):
        """ODE for first time derivative of the tangential velocity `v`.

        Parameters
        ----------
        t : float or ndarray
            Time vector [s]
        v : float or ndarray
            Tangential velocity time series [m/s]
        gm : float
            Central body standard gravitational parameter [m^3/s^2]
        r : float or ndarray
            Radius time series [m]
        m0 : float
            Initial spacecraft mass [kg]
        t0 : float
            Initial time [s]
        thrust : float
            Thrust magnitude [N]
        isp : float
            Specific impulse [s]

        Returns
        -------
        dv_dt : float or ndarray
            First time derivative of `v` [m/s^2]

        """

        dv_dt = ((thrust / (m0 - (thrust / isp / g0) * (t - t0))) ** 2 - (gm / r ** 2 - v ** 2 / r) ** 2) ** 0.5

        if self.v0 < self.vf:  # positive acceleration
            return dv_dt
        else:  # negative acceleration
            return -dv_dt

    def dx_dt(self, t, x, gm, r, m0, t0, thrust, isp):
        """Systems of ODEs for the first time derivatives of the angle `theta` and the tangential velocity `v`.

        Parameters
        ----------
        t : float or ndarray
            Time vector [s]
        x : ndarray
            Angle and tangential velocity time series as `[theta, v]` [rad, m/s]
        gm : float
            Central body standard gravitational parameter [m^3/s^2]
        r : float or ndarray
            Radius time series [m]
        m0 : float
            Initial spacecraft mass [kg]
        t0 : float
            Initial time [s]
        thrust : float
            Thrust magnitude [N]
        isp : float
            Specific impulse [s]

        Returns
        -------
        x_dot : ndarray
            First time derivatives of `theta` and `v` [rad/s, m/s^2]

        """

        x0_dot = x[1] / r
        x1_dot = self.dv_dt(t, x[1], gm, r, m0, t0, thrust, isp)
        x_dot = [x0_dot, x1_dot]

        return x_dot

    def __str__(self):
        """Prints info on `PowConstRadius` class instance. """

        lines = ['\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.dv_inf, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Finite dV:', self.dv, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Burn time:', self.tf - self.t0, 's'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1.0 - self.mf / self.m0, '')]

        s = '\n'.join(lines)

        return s


class TwoDimGuess:
    """`TwoDimGuess` provides an initial guess for a two-dimensional transfer trajectory combining powered phases at
    constant radius with Hohmann transfer arcs.

    Parameters
    ----------
    gm : float
        Central body standard gravitational parameter [m^3/s^2]
    r : float
        Central body equatorial radius [m]
    dep : TwoDimOrb
        Departure orbit object
    arr : TwoDimOrb
        Arrival orbit object
    sc : Spacecraft
        `Spacecraft` object

    Attributes
    ----------
    GM : float
        Central body standard gravitational parameter [m^3/s^2]
    R : float
        Central body equatorial radius [m]
    dep : TwoDimOrb
        Departure orbit object
    ht : HohmannTransfer
        `HohmannTransfer` object
    t : ndarray
        Time vector [s]
    states : ndarray
        States time series as `[r, theta, u, v, m]`
    controls : ndarray
        Controls time series as `[thrust, alpha]`

    """

    def __init__(self, gm, r, dep, arr, sc):
        """Initializes `TwoDimGuess` class. """

        self.GM = gm
        self.R = r
        self.sc = sc

        self.ht = HohmannTransfer(gm, dep, arr)

        self.t = self.states = self.controls = None

    def __str__(self):
        """Prints infos on `TwoDimGuess`.

        Returns
        -------
        s : str
            Infos on `TwoDimGuess`

        """

        lines = ['\n{:^50s}'.format('Departure Orbit:'),
                 self.ht.depOrb.__str__(),
                 '\n{:^50s}'.format('Arrival Orbit:'),
                 self.ht.arrOrb.__str__(),
                 '\n{:^50s}'.format('Hohmann transfer:'),
                 self.ht.transfer.__str__()]

        s = '\n'.join(lines)

        return s


class TwoDimLLOGuess(TwoDimGuess):
    """`TwoDimLLOGuess` provides an initial guess for a two-dimensional ascent or descent trajectory from the Moon
    surface to a circular Low Lunar Orbit.

    The approximate transfer consists into two powered phases at constant radius and an Hohmann transfer.

    Parameters
    ----------
    gm : float
        Central body standard gravitational parameter [m^3/s^2]
    r : float
        Central body equatorial radius [m]
    dep : TwoDimOrb
        Departure orbit object
    arr : TwoDimOrb
        Arrival orbit object
    sc : Spacecraft
        `Spacecraft` object

    Attributes
    ----------
    pow1 : PowConstRadius
        First powered phase at constant radius
    pow2 : PowConstRadius
        Second powered phase at constant radius

    """

    def __init__(self, gm, r, dep, arr, sc):
        """Initializes `TwoDimLLOGuess` class. """

        TwoDimGuess.__init__(self, gm, r, dep, arr, sc)

        self.pow1 = self.pow2 = None

    def compute_trajectory(self, fix_final=False, **kwargs):
        """Computes the states and controls time series for a given time vector or number of equally space nodes.

        Parameters
        ----------
        fix_final : bool, optional
            ``True`` if the final angle is fixed, ``False`` otherwise. Default is ``False``
        kwargs :
            t_eval : ndarray
                Time vector in which states and controls are computed [s]
            nb_nodes : int
                Number of equally space nodes in time in which states and controls are computed

        """

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
            self.states[:, 1] = self.states[:, 1] - self.states[-1, 1]

        if 'theta' in kwargs:
            self.states[:, 1] = self.states[:, 1] + kwargs['theta']

    def __str__(self):
        """Prints info on `TwoDimLLOGuess`.

        Returns
        -------
        s : str
            Info on `TwoDimLLOGuess`

        """

        lines = [TwoDimGuess.__str__(self),
                 '\n{:^50s}'.format('Initial guess:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1.0 - self.pow2.mf / self.sc.m0, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight:', self.pow2.tf, 's'),
                 '\n{:^50s}'.format('Departure burn:'),
                 self.pow1.__str__(),
                 '\n{:^50s}'.format('Arrival burn:'),
                 self.pow2.__str__()]

        s = '\n'.join(lines)

        return s


class TwoDimAscGuess(TwoDimLLOGuess):
    """`TwoDimAscGuess` provides an initial guess for a two-dimensional ascent trajectory from the Moon surface to a
    circular LLO.

    The approximate transfer comprises a first powered phase at constant radius equal to the Moon one, an Hohmann
    transfer and a second powered phase at constant radius equal to the LLO one.

    Parameters
    ----------
    gm : float
        Central body standard gravitational parameter [m^3/s^2]
    r : float
        Central body equatorial radius [m]
    alt : float
        LLO altitude [m]
    sc : Spacecraft
        `Spacecraft` object

    Attributes
    ----------
    pow1 : PowConstRadius
        First powered phase at constant radius
    pow2 : PowConstRadius
        Second powered phase at constant radius
    tf : float
        Final time [s]

    """

    def __init__(self, gm, r, alt, sc):
        """Initializes `TwoDimAscGuess` class. """

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
    """`TwoDimDescGuess` provides an initial guess for a two-dimensional descent trajectory from a circular LLO to the
    Moon surface.

    The approximate transfer comprises a first powered phase at constant radius equal to the LLO one, an Hohmann
    transfer and a second powered phase at constant radius equal to the Moon one.

    Parameters
    ----------
    gm : float
        Central body standard gravitational parameter [m^3/s^2]
    r : float
        Central body equatorial radius [m]
    alt : float
        LLO altitude [m]
    sc : Spacecraft
        `Spacecraft` object

    Attributes
    ----------
    pow1 : PowConstRadius
        First powered phase at constant radius
    pow2 : PowConstRadius
        Second powered phase at constant radius
    tf : float
        Final time [s]

    """

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

    from latom.utils.primary import Moon
    from latom.utils.spacecraft import Spacecraft
    from copy import deepcopy

    moon = Moon()
    sat = Spacecraft(450., 2.1, g=moon.g)

    nb = 50
    ff = False

    dg = TwoDimDescGuess(moon.GM, moon.R, 100e3, sat)
    dg2 = deepcopy(dg)

    tvec = np.reshape(np.linspace(0.0, dg.pow2.tf, nb), (nb, 1))
    tvec2 = np.sort(np.vstack((tvec, tvec[1:-1])), axis=0)

    dg.compute_trajectory(fix_final=ff, t_eval=tvec)
    dg2.compute_trajectory(fix_final=ff, t_eval=tvec2)
