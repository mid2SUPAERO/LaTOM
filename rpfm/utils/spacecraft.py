"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.utils.const import g0


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
        self.twr = twr
        self.m0 = m0

        if m_dry is not None:
            self.m_dry = float(m_dry)
        else:
            self.m_dry = self.m0/100

        self.w = isp*g0
        self.T_max = self.twr*self.m0*g
        self.T_min = self.T_max*throttle_min

    def __str__(self):
        """Prints the Spacecraft object attributes. """

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


if __name__ == '__main__':

    from rpfm.utils.primary import Moon

    moon = Moon()
    sc = Spacecraft(450., 2., throttle_min=0.5, g=moon.g)
    print(sc)
