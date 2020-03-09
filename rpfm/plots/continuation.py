"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt

from rpfm.plots.trajectories import TwoDimTrajectory


class MassEnergyContinuation:
    """Plots the propellant fraction and spacecraft specific energy as function of the thrust/weight ratio.

    Parameters
    ----------
    twr : ndarray
        List of thrust/weight ratios [-]
    m_prop_frac : ndarray
        List of propellant fractions [-]
    en : ndarray
        List of spacecraft specific energies [m^2/s^2]

    Attributes
    ----------
    twr : ndarray
        List of thrust/weight ratios [-]
    m_prop_frac : ndarray
        List of propellant fractions [-]
    energy : ndarray
        List of spacecraft specific energies [m^2/s^2]

    """

    def __init__(self, twr, m_prop_frac, en):
        """Initializes `MassEnergyContinuation` class. """

        self.twr = twr
        self.m_prop_frac = m_prop_frac
        self.energy = en

    def plot(self):
        """Plots the propellant fraction and spacecraft specific energy as function of the thrust/weight ratio. """

        fig, axs = plt.subplots(2, 1, constrained_layout=True)

        axs[0].plot(self.twr, self.m_prop_frac, color='b')
        axs[1].plot(self.twr, self.energy, color='b')

        axs[0].set_ylabel('propellant fraction (-)')
        axs[0].set_title('Propellant fraction')

        axs[1].set_ylabel('spacecraft energy (m^2/s^2)')
        axs[1].set_title('Spacecraft energy')

        for i in range(2):
            axs[i].set_xlabel('thrust/weight ratio (-)')
            axs[i].grid()


class TwoDimTrajectoryContinuation:
    """Plots the ascent trajectories from an initial Low Lunar Orbit to an intermediate ballistic arc for different
    thrust/weight ratios.

    Parameters
    ----------
    r_moon : float
        Moon radius [m] or [-]
    r_llo : float
        Initial Low Lunar Orbit radius [m] o [-]
    sol : dict
        Dictionary that maps each thrust/weight ratio to the corresponding optimal trajectory
    nb : float, optional
        Number of points in which the Moon surface and the initial orbits are discretized. Default is ``2000``
    log_scale : bool, optional
        ``True`` if `twr_list` is provided in logarithmic scale, ``False`` otherwise. Default is ``False``

    Attributes
    ----------
    scaler : float
        scaler for lengths
    units : str
        Unit of measurement for lengths
    x_moon : ndarray
        x coordinates for the Moon surface [km] or [-]
    y_moon : ndarray
        y coordinates for the Moon surface [km] or [-]
    x_llo : ndarray
        x coordinates for the initial orbit [km] or [-]
    y_llo : ndarray
        y coordinates for the initial orbit [km] or [-]
    x : dict
        x coordinates for the ascent trajectories [km] or [-]
    y : dict
        y coordinates for the ascent trajectories [km] or [-]

    """

    def __init__(self, r_moon, r_llo, sol, nb=2000, log_scale=False):
        """Initializes `TwoDimTrajectoryContinuation` class. """

        self.scaler, self.units = TwoDimTrajectory.get_scalers(r_moon)
        self.log_scale = log_scale

        self.x_moon, self.y_moon = TwoDimTrajectory.polar2cartesian(r_moon, scaler=self.scaler, nb=nb)
        self.x_llo, self.y_llo = TwoDimTrajectory.polar2cartesian(r_llo, scaler=self.scaler, nb=nb)

        self.x = {}
        self.y = {}

        for twr in sol.keys():
            self.x[twr], self.y[twr] = TwoDimTrajectory.polar2cartesian(sol[twr]['states'][:, 0], scaler=self.scaler,
                                                                        angle=sol[twr]['states'][:, 1])

    def plot(self):
        """Plots the ascent trajectories from an initial Low Lunar Orbit to an intermediate ballistic arc for different
        thrust/weight ratios.

        """

        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(self.x_moon, self.y_moon, label='Moon surface')
        ax.plot(self.x_llo, self.y_llo, label='departure orbit')

        if self.log_scale:
            for twr in self.x.keys():
                ax.plot(self.x[twr], self.y[twr], label=('log(twr): ' + str(twr)))
        else:
            for twr in self.x.keys():
                ax.plot(self.x[twr], self.y[twr], label=('twr: ' + str(twr)))

        TwoDimTrajectory.set_axes_decorators(ax, 'Ascent trajectories for different thrust/weight ratios', self.units)
