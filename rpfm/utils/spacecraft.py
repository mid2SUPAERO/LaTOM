"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.utils.primary import Moon


class Spacecraft:
    """Spacecraft class defines the spacecraft characteristics.

    Parameters
    ----------
    isp : int or float
        Specific impulse [s]
    twr : int or float
        Thrust over initial weight ratio [-]
    throttle_min : int or float, optional
        Minimum throttle level [-]. Default is 0.0
    m0 : int or float, optional
        Initial mass [kg]. Default is 1.0
    m_dry : int, float or None, optional
        Dry mass [kg]. Default is ``None`` for which `m_dry` is set equal to ``m0/100``

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
    T_max : float
        Maximum thrust [N]
    T_min : float
        Minimum thrust [N]

    """

    def __init__(self, isp, twr, throttle_min=0.0, m0=1.0, m_dry=None):

        self.Isp = float(isp)
        self.twr = float(twr)
        self.m0 = float(m0)

        if m_dry is not None:
            self.m_dry = float(m_dry)
        else:
            self.m_dry = self.m0/100

        moon = Moon()
        self.T_max = self.twr*self.m0*moon.g
        self.T_min = self.T_max*throttle_min


if __name__ == '__main__':

    sc = Spacecraft(450, 2, throttle_min=0.5)
    print(vars(sc))
