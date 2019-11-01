"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt


class TwoDimAltProfile:

    def __init__(self, r, states, states_exp=None, thrust=None, threshold=1e-6, r_safe=None,
                 labels=('powered', 'coast')):

        self.R = r

        if not np.isclose(r, 1.0):
            self.scaler = 1e3
            self.units = 'km'
        else:
            self.scaler = 1.0
            self.units = '-'

        self.r = states[:, 0]
        self.theta = states[:, 1]

        if thrust is not None:  # variable thrust

            states_pow = states[(thrust >= threshold).flatten(), :]
            states_coast = states[(thrust < threshold).flatten(), :]

            self.r_pow = states_pow[:, 0]
            self.theta_pow = states_pow[:, 1]
            self.r_coast = states_coast[:, 0]
            self.theta_coast = states_coast[:, 1]

        if states_exp is not None:

            self.r_exp = states_exp[:, 0]
            self.theta_exp = states_exp[:, 1]

        self.r_safe = r_safe
        self.labels = labels

    def plot(self):

        fig, ax = plt.subplots(1, 1, constrained_layout=True)

        if self.r_safe is not None:

            ax.plot(self.theta*180/np.pi, (self.r_safe - self.R)/self.scaler, color='k', label='safe altitude')

        if hasattr(self, 'r_exp'):  # explicit simulation

            ax.plot(self.theta_exp*180/np.pi, (self.r_exp - self.R)/self.scaler, color='g', label='explicit')

        if hasattr(self, 'r_pow'):  # implicit solution with variable thrust

            ax.plot(self.theta_coast * 180 / np.pi, (self.r_coast - self.R)/self.scaler, 'o', color='b',
                    label=self.labels[1])
            ax.plot(self.theta_pow*180/np.pi, (self.r_pow - self.R)/self.scaler, 'o', color='r', label=self.labels[0])

        else:  # implicit solution with constant thrust

            ax.plot(self.theta*180/np.pi, (self.r - self.R)/self.scaler, 'o', color='b', label='implicit')

        ax.set_ylabel(''.join(['h (', self.units, ')']))
        ax.set_xlabel('theta (deg)')
        ax.set_title('Altitude profile')
        ax.grid()
        ax.legend(loc='best')


class TwoDimTrajectory:

    def __init__(self, r_moon, r_llo, states, kind='ascent'):

        if not np.isclose(r_moon, 1.0):
            self.scaler = 1e3
            self.units = 'km'
        else:
            self.scaler = 1.0
            self.units = '-'

        self.kind = kind
        self.ang = np.linspace(0.0, 2 * np.pi, 500)

        # Moon
        self.x_moon = r_moon/self.scaler*np.cos(self.ang)
        self.y_moon = r_moon/self.scaler*np.sin(self.ang)

        # trajectory
        self.x = states[:, 0]/self.scaler*np.cos(states[:, 1])
        self.y = states[:, 0]/self.scaler*np.sin(states[:, 1])

        # LLO
        self.x_llo = r_llo/self.scaler*np.cos(self.ang)
        self.y_llo = r_llo/self.scaler*np.sin(self.ang)

    def plot(self):

        fig, ax = plt.subplots(constrained_layout=True)

        label = ' '.join([self.kind, 'trajectory'])
        title = ' '.join(['Optimal', label])

        ax.plot(self.x_moon, self.y_moon, label='Moon surface')

        if hasattr(self, 'x_nrho') and hasattr(self, 'y_nrho'):
            ax.plot(self.x_llo, self.y_llo, label='departure orbit')
            ax.plot(self.x_nrho, self.y_nrho, label='target orbit')
        else:
            ax.plot(self.x_llo, self.y_llo, label='target orbit')

        ax.plot(self.x, self.y, label=label)

        ax.set_aspect('equal')
        ax.grid()

        ax.tick_params(axis='x', rotation=60)

        ax.set_xlabel(''.join(['x (', self.units, ')']))
        ax.set_ylabel(''.join(['y (', self.units, ')']))
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1, 1), loc=2)


class TwoDimSurface2LLO(TwoDimTrajectory):

    def __init__(self, r_moon, states, kind='ascent'):

        # kind
        if kind == 'ascent':
            r_llo = states[-1, 0]
        elif kind == 'descent':
            r_llo = states[0, 0]
        else:
            raise ValueError('kind must be either ascent or descent')

        TwoDimTrajectory.__init__(self, r_moon, r_llo, states, kind=kind)


class TwoDimLLO2NRHO(TwoDimTrajectory):

    def __init__(self, r_moon, a_nrho, e_nrho, states, kind='ascent'):

        # kind
        if kind == 'ascent':
            r_llo = states[0, 0]
        elif kind == 'descent':
            r_llo = states[-1, 0]
        else:
            raise ValueError('kind must be either ascent or descent')

        TwoDimTrajectory.__init__(self, r_moon, r_llo, states, kind=kind)

        # NRHO
        r_nrho = a_nrho*(1 - e_nrho**2)/(1 + e_nrho*np.cos(self.ang))
        self.x_nrho = r_nrho/self.scaler*np.cos(self.ang)
        self.y_nrho = r_nrho/self.scaler*np.sin(self.ang)

