"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt

from rpfm.reader.reader import Reader
from rpfm.utils.const import states_2d
from rpfm.plots.timeseries import TwoDimControlsTimeSeries, TwoDimStatesTimeSeries
from rpfm.plots.trajectories import TwoDimAltProfile, TwoDimTrajectory


class TwoDimReader(Reader):

    def __init__(self, body, db, kind, case_id='final', db_exp=None):

        Reader.__init__(self, db, case_id, db_exp=db_exp)

        self.body = body
        self.phase_name = 'traj.powered.timeseries'

        self.tof, self.time, self.states, self.controls, self.r_min =\
            self.get_time_series(self.case, kind)

        if db_exp is not None:
            self.tof_exp, self.time_exp, self.states_exp, self.controls_exp, self.r_min_exp =\
                self.get_time_series(self.case_exp, kind)
        else:
            self.tof_exp = self.time_exp = self.states_exp = self.controls_exp = self.r_min_exp = None

    def get_time_series(self, case, kind):

        time = case.outputs.get(self.phase_name + '.time')
        tof = time[-1] - time[0]

        states = np.empty((np.size(time), 0))

        for k in states_2d:
            s = case.outputs.get(self.phase_name + '.states:' + k)
            states = np.append(states, s, axis=1)

        alpha = case.outputs.get(self.phase_name + '.controls:alpha')

        if kind == 'c':
            thrust = case.outputs.get(self.phase_name + '.design_parameters:thrust')
        else:
            thrust = case.outputs.get(self.phase_name + '.controls:thrust')

        controls = np.hstack((thrust, alpha))

        if kind == 's':
            r_min = case.outputs.get(self.phase_name + '.r_safe')
        else:
            r_min = None

        return tof, time, states, controls, r_min

    def plot(self, kind):

        states_plot = TwoDimStatesTimeSeries(self.body.R, self.time, self.states, time_exp=self.time_exp,
                                             states_exp=self.states_exp, thrust=self.controls[:, 0], r_min=self.r_min)

        if kind == 'c':
            controls_plot = TwoDimControlsTimeSeries(self.time, self.controls, 'const')
        else:
            controls_plot = TwoDimControlsTimeSeries(self.time, self.controls, 'variable')

        alt_plot = TwoDimAltProfile(self.body.R, self.states, states_exp=self.states_exp, thrust=self.controls[:, 0],
                                    r_min=self.r_min)
        trajectory_plot = TwoDimTrajectory(self.body.R, self.states[-1, 0], self.states)

        states_plot.plot()
        controls_plot.plot()
        alt_plot.plot()
        trajectory_plot.plot()

        plt.show()
