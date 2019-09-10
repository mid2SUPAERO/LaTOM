"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""


import matplotlib.pyplot as plt
from copy import deepcopy
from rpfm.plots.timeseries import TwoDimStatesTimeSeries, TwoDimControlsTimeSeries
from rpfm.plots.trajectories import TwoDimAltProfile, TwoDimTrajectory


class TwoDimSolPlot:

    def __init__(self, r, time, states, controls, time_exp=None, states_exp=None, controls_exp=None, r_safe=None,
                 threshold=1e-6, kind='ascent'):

        self.R = deepcopy(r)

        self.time = deepcopy(time)
        self.states = deepcopy(states)
        self.controls = deepcopy(controls)

        self.time_exp = deepcopy(time_exp)
        self.states_exp = deepcopy(states_exp)
        self.controls_exp = deepcopy(controls_exp)

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
        self.trajectory_plot = TwoDimTrajectory(self.R, self.states, self.kind)

    def plot(self):

        self.states_plot.plot()
        self.controls_plot.plot()
        self.alt_plot.plot()
        self.trajectory_plot.plot()

        plt.show()
