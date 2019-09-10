"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class TwoDimStatesTimeSeries:

    def __init__(self, r, time, states, time_exp=None, states_exp=None, thrust=None, threshold=1e-5, r_min=None):

        self.R = deepcopy(r)

        self.time = deepcopy(time)
        self.states = deepcopy(states)

        self.time_exp = deepcopy(time_exp)
        self.states_exp = deepcopy(states_exp)

        if thrust is not None:

            self.time_pow = self.time[(thrust >= threshold).flatten(), :]
            self.states_pow = self.states[(thrust >= threshold).flatten(), :]

            self.time_coast = self.time[(thrust < threshold).flatten(), :]
            self.states_coast = self.states[(thrust < threshold).flatten(), :]

        if r_min is not None:

            self.r_min = deepcopy(r_min)

    def plot(self):

        fig, axs = plt.subplots(2, 2, constrained_layout=True)

        if hasattr(self, 'r_min'):

            axs[0, 0].plot(self.time, (self.r_min - self.R)/1e3, color='k', label='safe altitude')

        if self.states_exp is not None:  # explicit simulation

            axs[0, 0].plot(self.time_exp, (self.states_exp[:, 0] - self.R)/1e3, color='g', label='explicit')
            axs[1, 0].plot(self.time_exp, self.states_exp[:, 1]*180/np.pi, color='g', label='explicit')
            axs[0, 1].plot(self.time_exp, self.states_exp[:, 2]/1e3, color='g', label='explicit')
            axs[1, 1].plot(self.time_exp, self.states_exp[:, 3]/1e3, color='g', label='explicit')

        if not hasattr(self, 'time_pow'):  # implicit solution with constant thrust

            axs[0, 0].plot(self.time, (self.states[:, 0] - self.R)/1e3, 'o', color='b', label='implicit')
            axs[1, 0].plot(self.time, self.states[:, 1]*180/np.pi, 'o', color='b', label='implicit')
            axs[0, 1].plot(self.time, self.states[:, 2]/1e3, 'o', color='b', label='implicit')
            axs[1, 1].plot(self.time, self.states[:, 3]/1e3, 'o', color='b', label='implicit')

        else:  # implicit solution with variable thrust

            axs[0, 0].plot(self.time_coast, (self.states_coast[:, 0] - self.R)/1e3, 'o', color='b', label='coast')
            axs[1, 0].plot(self.time_coast, self.states_coast[:, 1]*180/np.pi, 'o', color='b', label='coast')
            axs[0, 1].plot(self.time_coast, self.states_coast[:, 2]/1e3, 'o', color='b', label='coast')
            axs[1, 1].plot(self.time_coast, self.states_coast[:, 3]/1e3, 'o', color='b', label='coast')

            axs[0, 0].plot(self.time_pow, (self.states_pow[:, 0] - self.R)/1e3, 'o', color='r', label='powered')
            axs[1, 0].plot(self.time_pow, self.states_pow[:, 1]*180/np.pi, 'o', color='r', label='powered')
            axs[0, 1].plot(self.time_pow, self.states_pow[:, 2]/1e3, 'o', color='r', label='powered')
            axs[1, 1].plot(self.time_pow, self.states_pow[:, 3]/1e3, 'o', color='r', label='powered')

        axs[0, 0].set_ylabel('h (km)')
        axs[0, 0].set_title('Altitude')

        axs[0, 0].set_ylabel('theta (deg)')
        axs[1, 0].set_title('Angle')

        axs[0, 1].set_ylabel('u (km/s)')
        axs[0, 1].set_title('Radial velocity')

        axs[1, 1].set_ylabel('v (km/s)')
        axs[1, 1].set_title('Tangential velocity')

        for i in range(2):
            for j in range(2):
                axs[i, j].set_xlabel('time (s)')
                axs[i, j].grid()
                axs[i, j].legend(loc='best')


class TwoDimControlsTimeSeries:

    def __init__(self, time, controls, kind, threshold=1e-5):

        self.time = deepcopy(time)
        self.thrust = deepcopy(controls[:, 0])
        self.alpha = deepcopy(controls[:, 1])

        if kind in ['const', 'variable']:
            self.kind = kind
        else:
            raise ValueError('kind must be either const or variable')

        if kind == 'variable':

            self.alpha[(self.thrust < threshold).flatten()] = None

    def plot(self):

        if self.kind == 'const':

            fig, ax = plt.subplots(1, 1, constrained_layout=True)

            ax.plot(self.time, self.alpha*180/np.pi, 'o', color='r')
            ax.set_xlabel('time (s)')
            ax.set_ylabel('alpha (deg)')
            ax.set_title('Thrust direction')
            ax.grid()

        else:

            fig, axs = plt.subplots(2, 1, constrained_layout=True)

            axs[0].plot(self.time, self.alpha*180/np.pi, 'o', color='r')
            axs[0].set_xlabel('time (s)')
            axs[0].set_ylabel('alpha (deg)')
            axs[0].set_title('Thrust direction')
            axs[0].grid()

            # throttle
            axs[1].plot(self.time, self.thrust, 'o', color='r')
            axs[1].set_xlabel('time (s)')
            axs[1].set_ylabel('thrust (N)')
            axs[1].set_title('Thrust magnitude')
            axs[1].grid()
