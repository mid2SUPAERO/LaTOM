"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
from scipy.optimize import root

from rpfm.utils.cr3bp import Cr3bpEM
from rpfm.utils.coc import per2eq, coe2sv_vec


class TwoDimOrb:

    def __init__(self, gm, **kwargs):

        self.GM = gm

        if 'T' in kwargs:
            self.T = kwargs['T']
            self.a = (self.GM*self.T**2/4/np.pi**2)**(1/3)
        elif 'a' in kwargs:
            self.a = kwargs['a']

        if hasattr(self, 'a'):
            if 'e' in kwargs:
                self.e = kwargs['e']
                self.ra = self.a*(1 + self.e)
                self.rp = self.a*(1 - self.e)
            elif 'ra' in kwargs:
                self.ra = kwargs['ra']
                self.rp = 2*self.a - self.ra
                self.e = self.ra/self.a - 1
            elif 'rp' in kwargs:
                self.rp = kwargs['rp']
                self.ra = 2*self.a - self.rp
                self.e = 1 - self.rp/self.a
            else:
                raise AttributeError('T or a must be provided with e, ra or rp')

        elif ('e' in kwargs) and ('ra' in kwargs):
            self.e = kwargs['e']
            self.ra = kwargs['ra']
            self.a = self.ra/(1 + self.e)
            self.rp = self.a*(1 - self.e)
        elif ('e' in kwargs) and ('rp' in kwargs):
            self.e = kwargs['e']
            self.rp = kwargs['rp']
            self.a = self.rp/(1 - self.e)
            self.ra = self.a*(1 + self.e)
        elif ('ra' in kwargs) and ('rp' in kwargs):
            self.ra = kwargs['ra']
            self.rp = kwargs['rp']
            self.a = (self.ra + self.rp)/2
            self.e = (self.ra - self.rp)/(self.ra + self.rp)
        else:
            raise AttributeError('invalid kwargs combination')

        if not hasattr(self, 'T'):
            self.T = 2*np.pi/gm**0.5*self.a**1.5

        if self.e == 0.0:
            self.va = self.vp = (gm/self.a)**0.5
        else:
            self.va = (2*gm*self.rp/(self.ra*(self.ra + self.rp)))**0.5
            self.vp = (2*gm*self.ra/(self.rp*(self.ra + self.rp)))**0.5

        self.h = (gm * self.a * (1 - self.e ** 2)) ** 0.5
        self.n = (gm / self.a ** 3) ** 0.5

        self.E = - gm / 2 / self.a  # specific energy in 2BP [m^2/s^2]

        cr3bp = Cr3bpEM()
        self.J = 3 - cr3bp.mu*(1 + cr3bp.mu) - 2*self.E*cr3bp.T**2/cr3bp.L**2  # Jacobi constant approximation

    @staticmethod
    def coe2polar(gm, ta, **kwargs):

        if len(kwargs) != 2:
            raise AttributeError('Exactly two kwargs between a,e,h must be provided')
        elif ('a' in kwargs) and ('e' in kwargs):
            a = kwargs['a']
            e = kwargs['e']
            h = (gm*a*(1-e*e))**0.5
        elif ('a' in kwargs) and ('h' in kwargs):
            a = kwargs['a']
            h = kwargs['h']
            e = (1-h*h/gm/a)**0.5
        elif ('h' in kwargs) and ('e' in kwargs):
            e = kwargs['e']
            h = kwargs['h']
        else:
            raise AttributeError('Two kwargs between a,e,h must be provided')

        r = h*h/gm/(1+e*np.cos(ta))
        u = gm/h*e*np.sin(ta)
        v = gm/h*(1 + e*np.cos(ta))

        return r, u, v

    @staticmethod
    def polar2coe(gm, r, u, v):

        a = gm*r/(2*gm - r*(u*u + v*v))
        h = r*v
        e = (((r*v*v - gm)**2 + (r*u*v)**2)**0.5)/gm
        ta = np.arctan2(r*u*v, (r*v*v - gm))

        return a, e, h, ta

    @staticmethod
    def propagate(gm, a, e, tai, taf, nb, tp=0.0):

        ta = np.reshape(np.linspace(tai, taf, nb), (nb, 1))  # true anomaly [rad]
        ea = 2*np.arctan(((1 - e)/(1 + e))**0.5*np.tan(ta/2))  # eccentric anomaly [rad]
        me = ea - e*np.sin(ea)  # mean anomaly [rad]
        t = me*a**1.5/gm**0.5 + tp  # time [s]

        r, u, v = TwoDimOrb.coe2polar(gm, ta, a=a, e=e)  # states [m, m/s, m/s]

        states = np.hstack((r, ta, u, v))

        return t, states

    def __str__(self):

        lines = ['\n{:<25s}{:>20.6f}{:>5s}'.format('Semimajor axis:', self.a/1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Eccentricity:', self.e, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Periapsis radius:', self.rp/1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Apoapsis radius:', self.ra/1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Orbital period:', self.T, 's'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Periapsis velocity:', self.vp/1e3, 'km/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Apoapsis velocity:', self.va/1e3, 'km/s')]

        s = '\n'.join(lines)

        return s


class KepOrb:
    """KepOrb defines a Keplerian Orbit.

    Parameters
    ----------
    a : float
        Semi-major axis [m]
    e : float
        Eccentricity [-]
    i : float
        Inclination [rad]
    raan : float
        Right Ascension of the Ascending Node [rad]
    w : float
        Argument of Periapsis [rad]
    ta : float
        True anomaly [rad]
    gm : float
        Central body standard gravitational parameter [m^3/s^2]

    Attributes
    ----------
    eps : float
        Smallest number such that ``1.0 + eps != 1.0``
    a : float
        Semi-major axis [m]
    e : float
        Eccentricity [-]
    i : float
        Inclination [rad]
    W : float
        Right Ascension of the Ascending Node [rad]
    w : float
        Argument of Periapsis [rad]
    ta : float
        True anomaly [rad]
    GM : float, optional
        Central body standard gravitational parameter [m^3/s^2]
    n : float
        Mean motion [rad/s]
    R : ndarray
        Position vector [m]
    V : ndarray
        Velocity vector [m/s]
    H : ndarray
        Specific angular momentum vector [m^2/s]
    E : ndarray
        Eccentricity vector [-]
    h : float
        Specific angular momentum magnitude [m^2/s]
    r : float
        Position vector magnitude [m]
    vr : float
        Radial velocity [m/s]

    """

    def __init__(self, a, e, i, raan, w, ta, gm):
        """Initializes KepOrb class. """

        self.eps = np.finfo(float).eps

        self.a = a
        self.e = e
        self.i = i
        self.W = raan
        self.w = w
        self.ta = ta

        self.GM = gm

        # initialization
        self.n = None
        self.R = None
        self.V = None
        self.H = None
        self.E = None
        self.h = None
        self.r = None
        self.vr = None

        self.compute_state_vector()

    def set_state_vector(self, r, v):
        """Set the spacecraft state vector and compute the corresponding COE, H, E.

        Parameters
        ----------
        r : ndarray
            Position vector [m]
        v : ndarray
            Velocity vector [m/s]

        """

        self.R = r
        self.V = v

        self.compute_classical_orbital_elements()

    def set_classical_orbital_elements(self, a, e, i, raan, w, ta):
        """Set the spacecraft COE and compute the corresponding state vector, H, E.

        Parameters
        ----------
        a : float
            Semi-major axis [km]
        e : float
            Eccentricity [-]
        i : float
            Inclination [rad]
        raan : float
            Right Ascension of the Ascending Node [rad]
        w : float
            Argument of Periapsis [rad]
        ta : float
            True anomaly [rad]

        """

        self.a = a
        self.e = e
        self.i = i
        self.W = raan
        self.w = w
        self.ta = ta

        self.compute_state_vector()

    def set_true_anomaly(self, ta):
        """Set the spacecraft true anomaly and update the corresponding state vector.

        Parameters
        ----------
        ta : float
            True anomaly [rad]

        """

        self.ta = ta
        self.compute_state_vector()

    def compute_mean_motion(self):
        """Computes the spacecraft mean motion. """

        self.n = (self.GM * self.a ** -3) ** 0.5

    def compute_angular_momentum(self):
        """Computes the specific angular momentum vector. """

        self.H = np.cross(self.R, self.V)

    def compute_eccentricity(self):
        """Compute the spacecraft eccentricity vector. """

        self.E = np.cross(self.V, self.H) / self.GM - self.R / self.r

    def compute_classical_orbital_elements(self):
        """Computes the spacecraft classical orbital elements, specific angular momentum vector and eccentricity vector
        from its state vector. """

        self.r = np.linalg.norm(self.R, 2)  # orbit radius
        self.vr = np.dot(self.R, self.V) / self.r  # radial velocity

        self.compute_angular_momentum()  # angular momentum vector
        self.h = np.linalg.norm(self.H, 2)  # angular momentum magnitude

        self.i = np.arccos(self.H[2] / self.h)  # inclination

        k = np.array([0, 0, 1])  # unit vector along inertial Z axis
        node_line = np.cross(k, self.H)  # node line vector
        n = np.linalg.norm(node_line, 2)  # node line vector magnitude

        # right ascension of the ascending node
        if self.i >= self.eps:  # inclined orbit
            self.W = np.arccos(node_line[0] / n)
            if node_line[1] < 0.0:
                self.W = 2 * np.pi - self.W
        else:  # equatorial orbit
            self.W = 0.0

        self.compute_eccentricity()  # eccentricity vector
        self.e = np.linalg.norm(self.E, 2)  # eccentricity

        # argument of periapsis
        if self.e >= self.eps:  # elliptical orbit
            if self.i >= self.eps:  # inclined orbit
                self.w = np.arccos(np.dot(node_line, self.E) / (n * self.e))
                if self.E[2] < 0.0:
                    self.w = 2 * np.pi - self.w
            else:  # equatorial orbit
                self.w = np.arccos(self.E[0] / self.e)
                if self.E[1] < 0.0:
                    self.w = 2 * np.pi - self.w
        else:  # circular orbit
            self.w = 0.0

        # true anomaly
        if self.e >= self.eps:  # elliptical orbit
            self.ta = np.arccos(np.dot(self.E, self.R) / (self.e * self.r))
            if self.vr < 0.0:
                self.ta = 2 * np.pi - self.ta
        else:  # circular orbit
            if self.i >= self.eps:  # inclined orbit
                self.ta = np.arccos(np.dot(node_line, self.R) / (n * self.r))
                if self.R[2] < 0.0:
                    self.ta = 2 * np.pi - self.ta
            else:  # equatorial orbit
                self.ta = np.arccos(self.R[0] / self.r)
                if self.R[1] < 0.0:
                    self.ta = 2 * np.pi - self.ta

        self.a = (self.h ** 2 / self.GM) / (1 - self.e ** 2)  # semi-major axis
        self.compute_mean_motion()

    def compute_state_vector(self):
        """Computes the spacecraft state vector, specific angular momentum vector and eccentricity vector
        from its COE. """

        self.h = (self.GM * self.a * (1 - self.e ** 2)) ** 0.5  # angular momentum magnitude
        self.r = (self.h ** 2 / self.GM) / (1 + self.e * np.cos(self.ta))  # distance from central body

        # position and velocity vectors in perifocal reference frame
        r_per = self.r * np.array([np.cos(self.ta), np.sin(self.ta), 0.0])
        v_per = (self.GM / self.h) * np.array([-np.sin(self.ta), self.e + np.cos(self.ta), 0.0])

        q_per2eq = per2eq(self.W, self.i, self.w)  # rotation matrix from perifocal to equatorial reference frame

        self.R = q_per2eq @ r_per  # position vector in equatorial reference frame
        self.V = q_per2eq @ v_per  # velocity vector in equatorial reference frame

        self.vr = np.dot(self.R, self.V) / self.r  # radial velocity

        self.compute_angular_momentum()
        self.compute_eccentricity()
        self.compute_mean_motion()

    def compute_eccentric_anomaly(self, ta):
        """Compute the eccentric anomaly from a given true anomaly.

        Parameters
        ----------
        ta : float
            True anomaly [rad]

        Returns
        -------
        ea : float
            Eccentric anomaly [rad]

        """

        ea = 2 * np.arctan(((1 - self.e) / (1 + self.e)) ** 0.5 * np.tan(ta / 2))

        return ea

    def compute_true_anomaly(self, ea):
        """Compute the true anomaly from a given eccentric anomaly.

        Parameters
        ----------
        ea : float
            Eccentric anomaly [rad]

        Returns
        -------
        ta : float
            True anomaly [rad]

        """

        ta = 2 * np.arctan(((1 + self.e) / (1 - self.e)) ** 0.5 * np.tan(ea / 2))

        return ta

    def compute_periapsis_passage(self, ta, t):
        """Compute the time at periapsis passage given the current time and true anomaly.

        Parameters
        ----------
        t : float
            Time [s]
        ta : float
            True anomaly [rad]

        Returns
        -------
        tp : float
            Time at periapsis passage [s]

        """

        ea = self.compute_eccentric_anomaly(ta)  # eccentric anomaly
        me = ea - self.e * np.sin(ea)  # mean anomaly
        tp = t - me / self.n  # time at periapsis passage

        return tp

    def propagate(self, ta, t_vec, mode):
        """Propagate the orbit forward or backward in time solving the Kepler's time of flight equation.

        Parameters
        ----------
        ta : float
            Initial true anomaly [rad]
        t_vec : ndarray
            Time vector [s]
        mode : str
            ``fwd`` for forward propagation or ``back`` for backward propagation

        Returns
        -------
        r_vec : ndarray
            Position vector time series [m]
        v_vec : ndarray
            Velocity vector time series [m/s]

        """

        nb = len(t_vec)

        if mode == 'fwd':
            tp = self.compute_periapsis_passage(ta, t_vec[0])
        elif mode == 'back':
            tp = self.compute_periapsis_passage(ta, t_vec[-1])
        else:
            raise ValueError("Mode must be either 'fwd' or 'back'")

        ea0 = np.ones(nb)  # eccentric anomaly initial guess

        sol = root(self.kepler_eqn, ea0, args=(self.e, self.n, t_vec, tp), tol=1e-20)

        ea_vec = sol.x  # eccentric anomaly time series
        ta_vec = self.compute_true_anomaly(ea_vec)  # true anomaly time series

        r_vec, v_vec = coe2sv_vec(self.a, self.e, self.i, self.W, self.w, ta_vec, self.GM)  # state vector time series

        print('{:<50s}{:<30s}'.format("Solving Kepler's time of flight equation", sol.message))

        return r_vec, v_vec

    @staticmethod
    def kepler_eqn(ea, e, n, t, tp):
        """Kepler's Time of Flight equation.

        Parameters
        ----------
        ea : float
            Eccentric anomaly [rad]
        e : float
            Eccentricity [-]
        n : float
            Mean motion [rad/s]
        t : float
            Time [s]
        tp : float
            Time at periapsis passage [s]

        """

        return ea - e * np.sin(ea) - n * (t - tp)

    def __str__(self):
        """Prints the orbit Classical Orbital Elements. """

        d = {'a': (self.a, 'km'), 'e': (self.e, ''), 'i': (self.i * 180 / np.pi, 'deg'),
             'W': (self.W * 180 / np.pi, 'deg'),
             'w': (self.w * 180 / np.pi, 'deg'), 'ta': (self.ta * 180 / np.pi, 'deg')}

        lines = []

        for j in d.keys():
            lines.append('{:<6s}{:>20.6f} {:<4s}'.format(j, d[j][0], d[j][1]))

        s = '\n'.join(lines)

        return s


if __name__ == '__main__':

    from rpfm.utils.primary import Moon
    from rpfm.utils.cr3bp import Cr3bpEM

    moon = Moon()
    cr3bp = Cr3bpEM()
    J = 3.049

    gm_adim = moon.GM*cr3bp.T**2/cr3bp.L**3
    a_adim = gm_adim/(J - 3 + cr3bp.mu*(1 + cr3bp.mu))
    a_km = a_adim*cr3bp.L/1000

    o = TwoDimOrb(moon.GM, rp=3150e3, ra=65227.39e3)
    print(o)
