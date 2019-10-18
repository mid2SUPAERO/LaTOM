"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from rpfm.plots.timeseries import TwoDimStatesTimeSeries, TwoDimControlsTimeSeries
from rpfm.plots.trajectories import TwoDimAltProfile, TwoDimSurface2LLO, TwoDimLLO2NRHO


class TwoDimSolPlot:

    def __init__(self, r, time, states, controls, time_exp=None, states_exp=None, r_safe=None,
                 threshold=1e-6, kind='ascent', a=None, e=None):

        self.R = deepcopy(r)

        self.time = deepcopy(time)
        self.states = deepcopy(states)
        self.controls = deepcopy(controls)

        self.time_exp = deepcopy(time_exp)
        self.states_exp = deepcopy(states_exp)

        self.r_safe = r_safe
        self.threshold = threshold

        if threshold is None:
            thrust = None
        else:
            thrust = self.controls[:, 0]

        if kind in ['ascent', 'descent']:
            self.kind = kind
        else:
            raise ValueError('kind must be either ascent or descent')

        self.states_plot = TwoDimStatesTimeSeries(self.R, self.time, self.states, self.time_exp, self.states_exp,
                                                  thrust=thrust, threshold=threshold, r_safe=self.r_safe)
        self.controls_plot = TwoDimControlsTimeSeries(self.time, self.controls, threshold=threshold)
        self.alt_plot = TwoDimAltProfile(self.R, self.states, self.states_exp, thrust=thrust, threshold=threshold,
                                         r_safe=self.r_safe)

        if (a is not None) and (e is not None):
            self.trajectory_plot = TwoDimLLO2NRHO(self.R, a, e, self.states, self.kind)
        else:
            self.trajectory_plot = TwoDimSurface2LLO(self.R, self.states, self.kind)

    def plot(self):

        self.states_plot.plot()
        self.controls_plot.plot()
        self.alt_plot.plot()
        self.trajectory_plot.plot()

        plt.show()


class TwoDimTwoPhasesSolPlot:

    def __init__(self, r, time, states, controls, time_exp=None, states_exp=None, kind='ascent'):

        self.R = deepcopy(r)

        self.time = np.vstack(time)
        self.states = np.vstack(states)
        self.controls = np.vstack(controls)

        if time_exp is not None:
            self.time_exp = np.vstack(time_exp)
            self.states_exp = np.vstack(states_exp)
        else:
            self.time_exp = self.states_exp = None

        n0 = np.size(time[0])
        n1 = np.size(time[1])

        if kind == 'ascent':
            thrust = np.vstack((np.reshape(controls[0][:, 0], (n0, 1)), np.zeros((n1, 1))))
            self.kind = kind
        elif kind == 'descent':
            thrust = np.vstack((np.zeros((n0, 1)), np.reshape(controls[1][:, 0], (n1, 1))))
            self.kind = kind
        else:
            raise ValueError('kind must be either ascent or descent')

        self.states_plot = TwoDimStatesTimeSeries(self.R, self.time, self.states, self.time_exp, self.states_exp,
                                                  thrust=thrust, labels=('vertical', 'attitude-free'))
        self.controls_plot = TwoDimControlsTimeSeries(self.time, self.controls, threshold=None)
        self.alt_plot = TwoDimAltProfile(self.R, self.states, self.states_exp, thrust=thrust,
                                         labels=('vertical', 'attitude-free'))
        self.trajectory_plot = TwoDimSurface2LLO(self.R, self.states, self.kind)

    def plot(self):

        self.states_plot.plot()
        self.controls_plot.plot()
        self.alt_plot.plot()
        self.trajectory_plot.plot()

        plt.show()
