"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt

from rpfm.analyzer.analyzer import Analyzer
from rpfm.nlp.nlp_2d import TwoDimAscConstNLP, TwoDimAscVarNLP, TwoDimAscVToffNLP
from rpfm.plots.timeseries import TwoDimControlsTimeSeries, TwoDimStatesTimeSeries
from rpfm.plots.trajectories import TwoDimAltProfile, TwoDimTrajectory
from rpfm.utils.const import states_2d


class TwoDimAnalyzer(Analyzer):

    def __init__(self, body, sc):

        Analyzer.__init__(self, body, sc)

        self.states_scalers = np.array([self.body.R, 1.0, self.body.vc, self.body.vc, self.sc.m0])

    def get_states_alpha_time_series(self, p):

        tof = float(p.get_val(self.nlp.phase_name + '.t_duration'))*self.body.tc
        t = p.get_val(self.nlp.phase_name + '.timeseries.time')*self.body.tc

        states = np.empty((np.size(t), 0))

        for k in states_2d:
            s = p.get_val(self.nlp.phase_name + '.timeseries.states:' + k)
            states = np.append(states, s, axis=1)

        states = states*self.states_scalers
        alpha = p.get_val(self.nlp.phase_name + '.timeseries.controls:alpha')

        return tof, t, states, alpha

    def get_time_series(self, p):

        return None, None, None, None

    def get_solutions(self, explicit=True):

        tof, t, states, controls = self.get_time_series(self.nlp.p)

        self.tof = tof
        self.time = t
        self.states = states
        self.controls = controls

        if explicit:

            tof, t, states, controls = self.get_time_series(self.nlp.p_exp)

            self.tof_exp = tof
            self.time_exp = t
            self.states_exp = states
            self.controls_exp = controls


class TwoDimAscConstAnalyzer(TwoDimAnalyzer):

    def __init__(self, body, sc, alt, theta, tof, t_bounds, method, nb_seg, order, solver, snopt_opts=None,
                 rec_file=None):

        TwoDimAnalyzer.__init__(self, body, sc)

        self.nlp = TwoDimAscConstNLP(body, sc, alt, theta, (-np.pi, np.pi), tof, t_bounds, method, nb_seg, order,
                                     solver, 'powered', snopt_opts, rec_file)

    def get_time_series(self, p):

        tof, t, states, alpha = self.get_states_alpha_time_series(p)
        thrust = self.sc.T_max*np.ones((len(t), 1))
        controls = np.hstack((thrust, alpha))

        return tof, t, states, controls

    def plot(self):

        states_plot = TwoDimStatesTimeSeries(self.body.R, self.time, self.states, time_exp=self.time_exp,
                                             states_exp=self.states_exp)
        controls_plot = TwoDimControlsTimeSeries(self.time, self.controls, 'const')
        alt_plot = TwoDimAltProfile(self.body.R, self.states, states_exp=self.states_exp)
        trajectory_plot = TwoDimTrajectory(self.body.R, self.states[-1, 0], self.states)

        states_plot.plot()
        controls_plot.plot()
        alt_plot.plot()
        trajectory_plot.plot()

        plt.show()


class TwoDimAscVarAnalyzer(TwoDimAnalyzer):

    def __init__(self, body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None,
                 rec_file=None):

        TwoDimAnalyzer.__init__(self, body, sc)

        self.nlp = TwoDimAscVarNLP(body, sc, alt, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg, order, solver,
                                   'powered', snopt_opts=snopt_opts, rec_file=rec_file)

    def get_time_series(self, p):

        tof, t, states, alpha = self.get_states_alpha_time_series(p)
        thrust = p.get_val(self.nlp.phase_name + '.timeseries.controls:thrust')
        controls = np.hstack((thrust, alpha))

        return tof, t, states, controls

    def plot(self):

        states_plot = TwoDimStatesTimeSeries(self.body.R, self.time, self.states, time_exp=self.time_exp,
                                             states_exp=self.states_exp, thrust=self.controls[:, 0])
        controls_plot = TwoDimControlsTimeSeries(self.time, self.controls, 'variable')
        alt_plot = TwoDimAltProfile(self.body.R, self.states, states_exp=self.states_exp, thrust=self.controls[:, 0])
        trajectory_plot = TwoDimTrajectory(self.body.R, self.states[-1, 0], self.states)

        states_plot.plot()
        controls_plot.plot()
        alt_plot.plot()
        trajectory_plot.plot()

        plt.show()


class TwoDimAscVToffAnalyzer(TwoDimAscVarAnalyzer):

    pass
