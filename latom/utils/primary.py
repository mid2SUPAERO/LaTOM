"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np


class Primary:
    """Primary class defines the characteristic quantities for a given primary body.

    Attributes
    ----------
    R : float
        Radius [m]
    GM : float
        Standard gravitational parameter [m^3/s^2]
    g : float
        Surface gravity [m/s]
    tc : float
        Characteristic time [s]
    vc : float
        Characteristic speed [m/s]
    T_circ : float
        Orbital period in a circular orbit with radius equal to `R` [s]

    """

    def __init__(self, r, gm):
        """Init Primary class. """

        self.R = r
        self.GM = gm

        self.g = self.GM*self.R**-2.0
        self.tc = self.R**1.5*self.GM**-0.5
        self.vc = self.GM**0.5*self.R**-0.5
        self.T_circ = 2*np.pi*self.tc


class Moon(Primary):
    """ Defines the Moon"""

    def __init__(self):
        """Init Moon class. """

        Primary.__init__(self, 1737.4e3, 4.902800066163796e12)


if __name__ == '__main__':

    moon = Moon()

    for k, v in vars(moon).items():
        print(k, v)
