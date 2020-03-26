"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt


class TwoDimStatesTimeSeries:
    """Plot the two-dimensional simulation's states in time.

    Parameters
    ----------
    r : float
        Equatorial radius of central attracting body [m] or [-]
    time : ndarray
        Time vector for implicit NLP solution [s] or [-]
    states : ndarray
        States time series for implicit NLP solution as `[r, theta, u, v, m]`
    time_exp : ndarray or ``None``, optional
        Time vector for explicit simulation [s] o [-] or ``None``. Default is ``None``
    states_exp : ndarray or ``None``, optional
        States time series for explicit simulation as `[r, theta, u, v, m]` or ``None``. Default is ``None``
    thrust : ndarray or ``None``
        Thrust time series [N] or [-] or ``None``. Default is ``None``
    threshold : float or ``None``, optional
        Threshold value to determine the on/off control structure or ``None``. Default is ``1e-6``
    r_safe : ndarray or ``None``, optional
        Time series for minimum safe altitude [m] or [-] or ``None``. Default is ``None``
    threshold : float
        The threshold for the thrust values
    labels : iterable, optional
        Labels for the different phases. Default is `('powered', 'coast')`

    Attributes
    ----------
    R : float
        Equatorial radius of central attracting body [m] or [-]
    scaler : float
        Value to scale the distances
    units : list
        List of measurement units for distances, velocities and time
    time : ndarray
        Time vector for implicit NLP solution [s] or [-]
    states : ndarray
        States time series for implicit NLP solution as `[r, theta, u, v, m]`
    time_exp : ndarray or ``None``
        Time vector for explicit simulation [s] o [-] or ``None``
    states_exp : ndarray or ``None``
        States time series for explicit simulation as `[r, theta, u, v, m]` or ``None``
    time_pow : ndarray
        Time vector for implicit NLP solution corresponding to a powered phase [s] or [-]
    states_pow : ndarray
        States time series for implicit NLP solution corresponding to a powered phase as `[r, theta, u, v, m]`
    time_coast : ndarray
        Time vector for implicit NLP solution corresponding to a coasting phase [s] or [-]
    states_coast : ndarray
        States time series for implicit NLP solution corresponding to a coasting phase as `[r, theta, u, v, m]`
    r_safe : ndarray or ``None``
        Time series for minimum safe altitude [m] or [-] or ``None``
    labels : iterable
        Labels for the different phases

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
        """Plot the two-dimensional simulation's states and in time. """

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
    """Plot the two-dimensional simulation's controls in time.

    Parameters
    ----------
    time : ndarray
        Time vector for implicit NLP solution [s] or [-]
    controls : ndarray
        Controls time series for implicit NLP solution as `[thrust, alpha]`
    threshold : float or ``None``, optional
        Threshold value to determine the on/off control structure or ``None``. Default is ``1e-6``
    units : iterable, optional
        Measurements units for thrust, time. Default is `('N', 's')`

    Attributes
    ----------
    time : ndarray
        Time vector for implicit NLP solution [s] or [-]
    thrust : ndarray
        Thrust magnitude time series for implicit NLP solution [N] or [-]
    alpha : ndarray
        Thrust direction time series for implicit NLP solution [rad]
    threshold : float or ``None``
        Threshold value to determine the on/off control structure or ``None``
    units : iterable
        Measurements units for thrust, time

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
        """Plot the two-dimensional simulation's controls in time. """

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
