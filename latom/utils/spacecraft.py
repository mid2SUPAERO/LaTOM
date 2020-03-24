"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from warnings import warn
from latom.utils.const import g0


class Spacecraft:
    """Spacecraft class defines the spacecraft characteristics.

    Parameters
    ----------
    isp : float
        Specific impulse [s]
    twr : float
        Thrust over initial weight ratio [-]
    throttle_min : float, optional
        Minimum throttle level [-]. Default is 0.0
    m0 : float, optional
        Initial mass [kg]. Default is 1.0
    m_dry : float or None, optional
        Dry mass [kg]. Default is ``None`` for which `m_dry` is set equal to ``m0/100``
    g : float, optional
        Central body surface gravity [m/s^2]. Default is `g0`

    Attributes
    ----------
    Isp : float
        Specific impulse [s]
    twr : float
        Thrust over initial weight ratio [-]
    m0 : float
        Initial mass [kg]
    m_dry : float
        Dry mass [kg]
    w : float
        Exhaust velocity [m/s]
    T_max : float
        Maximum thrust [N]
    T_min : float
        Minimum thrust [N]

    """

    def __init__(self, isp, twr, throttle_min=0.0, m0=1.0, m_dry=None, g=g0):
        """Init Spacecraft class. """

        self.Isp = isp
        self.m0 = m0
        self.throttle_min = throttle_min
        self.g = g

        self.twr = self.T_max = self.T_min = 0.0

        if m_dry is not None:
            self.m_dry = float(m_dry)
        else:
            self.m_dry = self.m0 / 100

        self.w = isp * g0
        self.update_twr(twr)

    def update_twr(self, twr):

        self.twr = twr
        self.T_max = self.twr * self.m0 * self.g
        self.T_min = self.T_max * self.throttle_min

    def __str__(self):
        """Prints the Spacecraft class attributes. """

        lines = ['\n{:^40s}'.format('Spacecraft characteristics:'),
                 '\n{:<20s}{:>15.3f}{:>5s}'.format('Initial mass:', self.m0, 'kg'),
                 '{:<20s}{:>15.3f}{:>5s}'.format('Dry mass:', self.m_dry, 'kg'),
                 '{:<20s}{:>15.3f}{:>5s}'.format('Thrust/weight ratio:', self.twr, ''),
                 '{:<20s}{:>15.3f}{:>5s}'.format('Max thrust:', self.T_max, 'N'),
                 '{:<20s}{:>15.3f}{:>5s}'.format('Min thrust:', self.T_min, 'N'),
                 '{:<20s}{:>15.3f}{:>5s}'.format('Specific impulse:', self.Isp, 's'),
                 '{:<20s}{:>15.3f}{:>5s}'.format('Exhaust velocity:', self.w, 'm/s')]

        s = '\n'.join(lines)

        return s


class ImpulsiveBurn:
    """ImpulsiveBurn class describes an impulsive burn.

    Parameters
    ----------
    sc : Spacecraft
        Instant of `Spacecraft` class
    dv : float
        Change in velocity corresponding to the impulsive burn [m/s]

    Attributes
    ----------
    sc : Spacecraft
        Instant of `Spacecraft` class
    dv : float
        Change in velocity corresponding to the impulsive burn [m/s]
    mf : float
        Spacecraft final mass after the impulsive burn [kg]
    dm : float
        Propellant mass required for the impulsive burn [kg]

    """

    def __init__(self, sc, dv):
        """Initializes `ImpulsiveBurn` class. """

        self.sc = sc
        self.dv = dv

        self.mf = self.tsiolkovsky_mf(self.sc.m0, dv, self.sc.Isp)
        self.dm = self.sc.m0 - self.mf

    @staticmethod
    def tsiolkovsky_mf(m0, dv, isp):
        """Computes the final spacecraft mass for a given velocity change using the Tsiolkovsky rocket equation.

        Parameters
        ----------
        m0 : float
            Initial spacecraft mass [kg]
        dv : float
            Change in velocity [m/s]
        isp : float
            Specific impulse of the spacecraft rocket engine [s]

        Returns
        -------
        mf : float
            Final spacecraft mass [kg]

        """

        mf = m0 * np.exp(-abs(dv) / isp / g0)

        return mf

    @staticmethod
    def tsiolkovsky_dv(m0, mf, isp):
        """Computes the velocity change for a given initial and final spacecraft masses using the Tsiolkovsky rocket
        equation.

        Parameters
        ----------
        m0 : float
            Initial spacecraft mass [kg]
        mf : float
            Final spacecraft mass [kg]
        isp : float
            Specific impulse of the spacecraft rocket engine [s]

        Returns
        -------
        dv : float
            Change in velocity [m/s]

        """

        dv = isp * g0 * np.log(m0 / mf)

        return dv

    def __str__(self):
        """Prints the ImpulsiveBurn class attributes. """

        lines = [self.sc.__str__(),
                 '\n{:^40s}'.format('Impulsive Burn:'),
                 '\n{:<20s}{:>15.3f}{:>5s}'.format('Velocity change:', self.dv, 'm/s'),
                 '{:<20s}{:>15.3f}{:>5s}'.format('Propellant mass:', self.dm, 'kg'),
                 '{:<20s}{:>15.3f}{:>5s}'.format('Final mass:', self.mf, 'kg')]

        s = '\n'.join(lines)

        return s
