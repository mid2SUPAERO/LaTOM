"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt


class TwoDimStatesTimeSeries:
    """ Plot the two-dimensional simulation's states in time

     Parameters
     ----------
     r : ndarray
        Position along the trajectory [m] or [-]
     time : ndarray
        Simulation time interval [s] or [-]
     states : ndarray
        List of the states values obtained from the simulation
     time_exp : bool
        Defines if the time scale is exponential
     states_exp : bool
        Defines if the states values scale is exponential
     thrust : ndarray
        List of the thrust values
     r_safe : float
        Value of the minimum safe altitude [m] or [-]
     threshold : float
        The threshold for the thrust values
     labels : str
        Defines the kind of phase. The possible values are ['powered', 'coast']

     Attributes
     ----------
     R : ndarray
        Position along the trajectory [m] or [-]
     scaler : float
        Value to scale the distances
     units : ndarray
        List of measurement units
     time : ndarray
        Simulation time interval [s] or [-]
     states : ndarray
        List of the states values obtained from the simulation
     time_exp : bool
        Defines if the time scale is exponential
     states_exp : bool
        Defines if the states values scale is exponential
     time_pow : ndarray
        List of time values for the powered phase
     states_pow : ndarray
        List of states values for the powered phase
     time_coast : ndarray
        List of time value for the coasting phase
     states_coast : ndarray
        List of states values for the coasting phase
     r_safe : float
        Value of the minimum safe altitude [m] or [-]
     labels : str
        Defines the kind of phase. The possible values are ['powered', 'coast']
     """

    def __init__(self, r, time, states, time_exp=None, states_exp=None, thrust=None, threshold=1e-6, r_safe=None,
                 labels=('powered', 'coast')):
        """Initializes `TwoDimStatesTimeSeries` class. """

        self.R = r

        if not np.isclose(r, 1.0):
            self.scaler = 1e3
            self.units = ['km', 'km/s', 's']
        else:
            self.scaler = 1.0
            self.units = ['-', '-', '-']

        self.time = time
        self.states = states

        self.time_exp = time_exp
        self.states_exp = states_exp

        if thrust is not None:

            self.time_pow = self.time[(thrust >= threshold).flatten(), :]
            self.states_pow = self.states[(thrust >= threshold).flatten(), :]

            self.time_coast = self.time[(thrust < threshold).flatten(), :]
            self.states_coast = self.states[(thrust < threshold).flatten(), :]

        self.r_safe = r_safe
        self.labels = labels

    def plot(self):
        """ Plot the two-dimensional simulation's states and controls in time"""

        fig, axs = plt.subplots(2, 2, constrained_layout=True)

        if self.r_safe is not None:

            axs[0, 0].plot(self.time, (self.r_safe - self.R)/self.scaler, color='k', label='safe altitude')

        if self.states_exp is not None:  # explicit simulation

            axs[0, 0].plot(self.time_exp, (self.states_exp[:, 0] - self.R)/self.scaler, color='g', label='explicit')
            axs[1, 0].plot(self.time_exp, self.states_exp[:, 1]*180/np.pi, color='g', label='explicit')
            axs[0, 1].plot(self.time_exp, self.states_exp[:, 2]/self.scaler, color='g', label='explicit')
            axs[1, 1].plot(self.time_exp, self.states_exp[:, 3]/self.scaler, color='g', label='explicit')

        if not hasattr(self, 'time_pow'):  # implicit solution with constant thrust

            axs[0, 0].plot(self.time, (self.states[:, 0] - self.R)/self.scaler, 'o', color='b', label='implicit')
            axs[1, 0].plot(self.time, self.states[:, 1]*180/np.pi, 'o', color='b', label='implicit')
            axs[0, 1].plot(self.time, self.states[:, 2]/self.scaler, 'o', color='b', label='implicit')
            axs[1, 1].plot(self.time, self.states[:, 3]/self.scaler, 'o', color='b', label='implicit')

        else:  # implicit solution with variable thrust

            axs[0, 0].plot(self.time_coast, (self.states_coast[:, 0] - self.R)/self.scaler, 'o', color='b',
                           label=self.labels[1])
            axs[1, 0].plot(self.time_coast, self.states_coast[:, 1]*180/np.pi, 'o', color='b', label=self.labels[1])
            axs[0, 1].plot(self.time_coast, self.states_coast[:, 2]/self.scaler, 'o', color='b', label=self.labels[1])
            axs[1, 1].plot(self.time_coast, self.states_coast[:, 3]/self.scaler, 'o', color='b', label=self.labels[1])

            axs[0, 0].plot(self.time_pow, (self.states_pow[:, 0] - self.R)/self.scaler, 'o', color='r',
                           label=self.labels[0])
            axs[1, 0].plot(self.time_pow, self.states_pow[:, 1]*180/np.pi, 'o', color='r', label=self.labels[0])
            axs[0, 1].plot(self.time_pow, self.states_pow[:, 2]/self.scaler, 'o', color='r', label=self.labels[0])
            axs[1, 1].plot(self.time_pow, self.states_pow[:, 3]/self.scaler, 'o', color='r', label=self.labels[0])

        axs[0, 0].set_ylabel(''.join(['h (', self.units[0], ')']))
        axs[0, 0].set_title('Altitude')

        axs[1, 0].set_ylabel('theta (deg)')
        axs[1, 0].set_title('Angle')

        axs[0, 1].set_ylabel(''.join(['u (', self.units[1], ')']))
        axs[0, 1].set_title('Radial velocity')

        axs[1, 1].set_ylabel(''.join(['v (', self.units[1], ')']))
        axs[1, 1].set_title('Tangential velocity')

        for i in range(2):
            for j in range(2):
                axs[i, j].set_xlabel(''.join(['time (', self.units[2], ')']))
                axs[i, j].grid()
                axs[i, j].legend(loc='best')


class TwoDimControlsTimeSeries:
    """ Plot the two-dimensional simulation's controls in time

     Parameters
     ----------
     time : ndarray
        Simulation time interval [s] or [-]
     controls : ndarray
        List of the controls values obtained from the simulation
     threshold : float
        The threshold for the thrust values
     units : str
        Defines the measures units

    Attributes
    ---------
    time : ndarray
        Simulation time interval [s] or [-]
    thrust : ndarray
        List of thrust time series [N] or [-]
    alpha : ndarray
        List of thrust angle time series [rad] or [-]
    threshold : float
        The threshold for the thrust values
    units : str
        Defines the measures units

    """
    def __init__(self, time, controls, threshold=1e-6, units=('N', 's')):
        """Initializes `TwoDimControlsTimeSeries` class. """

        self.time = time
        self.thrust = controls[:, 0]
        self.alpha = controls[:, 1]
        self.threshold = threshold
        self.units = units

        if threshold is not None:
            self.alpha[(self.thrust < threshold).flatten()] = None

    def plot(self):
        """ Plot the two-dimensional simulation's controls in time"""

        if self.threshold is None:

            fig, ax = plt.subplots(1, 1, constrained_layout=True)

            ax.plot(self.time, self.alpha*180/np.pi, 'o', color='r')
            ax.set_xlabel(''.join(['time (', self.units[1], ')']))
            ax.set_ylabel('alpha (deg)')
            ax.set_title('Thrust direction')
            ax.grid()

        else:

            fig, axs = plt.subplots(2, 1, constrained_layout=True)

            axs[0].plot(self.time, self.alpha*180/np.pi, 'o', color='r')
            axs[0].set_xlabel(''.join(['time (', self.units[1], ')']))
            axs[0].set_ylabel('alpha (deg)')
            axs[0].set_title('Thrust direction')
            axs[0].grid()

            # throttle
            axs[1].plot(self.time, self.thrust, 'o', color='r')
            axs[1].set_xlabel(''.join(['time (', self.units[1], ')']))
            axs[1].set_ylabel(''.join(['thrust (', self.units[0], ')']))
            axs[1].set_title('Thrust magnitude')
            axs[1].grid()
