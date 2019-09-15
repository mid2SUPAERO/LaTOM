"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.reader.reader import Reader
from rpfm.utils.const import states_2d
from rpfm.plots.solutions import TwoDimSolPlot


class TwoDimReader(Reader):

    def __init__(self, kind, body, db, case_id='final', db_exp=None):

        Reader.__init__(self, db, case_id, db_exp=db_exp)

        self.kind = kind  # ('ascent/descent', 'const/variable', 'True/False')
        self.body = body
        self.phase_name = 'traj.powered.timeseries'
        self.states_scalers = np.array([self.body.R, 1.0, self.body.vc, self.body.vc, 1.0])

        self.tof, self.time, self.states, self.controls, self.r_safe =\
            self.get_time_series(self.case, kind)

        if db_exp is not None:
            self.tof_exp, self.time_exp, self.states_exp, self.controls_exp, self.r_safe_exp =\
                self.get_time_series(self.case_exp, kind)
        else:
            self.tof_exp = self.time_exp = self.states_exp = self.controls_exp = self.r_safe_exp = None

    def get_time_series(self, case, kind):

        time = case.outputs.get(self.phase_name + '.time')*self.body.tc
        tof = time[-1] - time[0]

        states = np.empty((np.size(time), 0))

        for k in states_2d:
            s = case.outputs.get(self.phase_name + '.states:' + k)
            states = np.append(states, s, axis=1)

        states = states*self.states_scalers

        alpha = case.outputs.get(self.phase_name + '.controls:alpha')

        if kind[1] == 'const':
            thrust = case.outputs.get(self.phase_name + '.design_parameters:thrust')
        elif kind[1] == 'variable':
            thrust = case.outputs.get(self.phase_name + '.controls:thrust')
        else:
            raise ValueError('the second element of kind must be const or variable')

        controls = np.hstack((thrust*self.body.g*states[-1, -1], alpha))

        if kind[2]:
            r_safe = case.outputs.get(self.phase_name + '.r_safe')*self.body.R
        else:
            r_safe = None

        return tof, time, states, controls, r_safe

    def plot(self):

        if self.kind[1] == 'const':
            threshold = None
        else:
            threshold = 1e-6

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 self.controls_exp, self.r_safe, threshold, self.kind[0])
        sol_plot.plot()
