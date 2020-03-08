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

    def __init__(self, r_moon, r_llo, sol, nb=2000):

        self.scaler, self.units = TwoDimTrajectory.get_scalers(r_moon)

        self.x_moon, self.y_moon = TwoDimTrajectory.polar2cartesian(r_moon, scaler=self.scaler, nb=nb)
        self.x_llo, self.y_llo = TwoDimTrajectory.polar2cartesian(r_llo, scaler=self.scaler, nb=nb)

        self.x = {}
        self.y = {}

        for twr in sol.keys():
            self.x[twr], self.y[twr] = TwoDimTrajectory.polar2cartesian(sol[twr]['states'][:, 0], scaler=self.scaler,
                                                                        angle=sol[twr]['states'][:, 1])

    def plot(self):

        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(self.x_moon, self.y_moon, label='Moon surface')
        ax.plot(self.x_llo, self.y_llo, label='departure orbit')

        for twr in self.x.keys():
            ax.plot(self.x[twr], self.y[twr], label=f"twr {twr:.4f}")

        TwoDimTrajectory.set_axes_decorators(ax, 'Ascent trajectories for different thrust/weight ratios', self.units)
