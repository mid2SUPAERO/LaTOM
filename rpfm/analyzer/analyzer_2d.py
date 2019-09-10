"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.analyzer.analyzer import Analyzer
from rpfm.nlp.nlp_2d import TwoDimAscConstNLP, TwoDimAscVarNLP, TwoDimAscVToffNLP
from rpfm.plots.solutions import TwoDimSolPlot
from rpfm.utils.const import states_2d


class TwoDimAnalyzer(Analyzer):

    def __init__(self, body, sc):

        Analyzer.__init__(self, body, sc)

        self.phase_name = 'powered'
        self.states_scalers = np.array([self.body.R, 1.0, self.body.vc, self.body.vc, 1.0])

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

    def __str__(self):

        lines = [self.nlp.__str__(), self.sc.__str__(),
                 '\n{:^50s}'.format('Trajectory:'),
                 '\n{:<25s}{:>20.12f}{:>5s}'.format('Time of flight:', self.tof, 's'),
                 '{:<25s}{:>20.12f}{:>5s}'.format('Propellant fraction:', (1.0 - self.states[-1, -1] / self.sc.m0), '')]

        if self.states_exp is not None:
            err = self.states[-1, :] - self.states_exp[-1, :]

            lines_err = ['\n{:^50s}'.format('Error:'),
                         '\n{:<25s}{:>20.12f}{:>5s}'.format('Radius:', err[0] / 1e3, 'km'),
                         '{:<25s}{:>20.12f}{:>5s}'.format('Angle:', err[1] * np.pi / 180, 'deg'),
                         '{:<25s}{:>20.12f}{:>5s}'.format('Radial velocity:', err[2] / 1e3, 'km/s'),
                         '{:<25s}{:>20.12f}{:>5s}'.format('Tangential velocity:', err[3] / 1e3, 'km/s'),
                         '{:<25s}{:>20.12f}{:>5s}'.format('Mass:', err[4], 'kg')]

            lines.extend(lines_err)

        s = '\n'.join(lines)

        return s


class TwoDimAscConstAnalyzer(TwoDimAnalyzer):

    def __init__(self, body, sc, alt, theta, tof, t_bounds, method, nb_seg, order, solver, snopt_opts=None,
                 rec_file=None, check_partials=False, u_bound=False):

        TwoDimAnalyzer.__init__(self, body, sc)

        self.nlp = TwoDimAscConstNLP(body, sc, alt, theta, (-np.pi, np.pi), tof, t_bounds, method, nb_seg, order,
                                     solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                     check_partials=check_partials, u_bound=u_bound)

    def get_time_series(self, p):

        tof, t, states, alpha = self.get_states_alpha_time_series(p)
        thrust = self.sc.T_max*np.ones((len(t), 1))
        controls = np.hstack((thrust, alpha))

        return tof, t, states, controls

    def plot(self):

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 self.controls_exp, threshold=None)
        sol_plot.plot()


class TwoDimAscVarAnalyzer(TwoDimAnalyzer):

    def __init__(self, body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False, u_bound=False):

        TwoDimAnalyzer.__init__(self, body, sc)

        self.nlp = TwoDimAscVarNLP(body, sc, alt, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg, order, solver,
                                   self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                   check_partials=check_partials, u_bound=u_bound)

    def get_time_series(self, p):

        tof, t, states, alpha = self.get_states_alpha_time_series(p)
        thrust = p.get_val(self.nlp.phase_name + '.timeseries.controls:thrust')*self.body.g*self.sc.m0
        controls = np.hstack((thrust, alpha))

        return tof, t, states, controls

    def plot(self):

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 self.controls_exp)
        sol_plot.plot()


class TwoDimAscVToffAnalyzer(TwoDimAscVarAnalyzer):

    def __init__(self, body, sc, alt, alt_min, slope, t_bounds, method, nb_seg, order, solver, snopt_opts=None,
                 rec_file=None, check_partials=False, u_bound=False):

        TwoDimAnalyzer.__init__(self, body, sc)

        self.nlp = TwoDimAscVToffNLP(body, sc, alt, alt_min, slope, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg,
                                     order, solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                     check_partials=check_partials, u_bound=u_bound)

        self.r_safe = None

    def get_solutions(self, explicit=True):
        
        TwoDimAnalyzer.get_solutions(self, explicit=explicit)

        self.r_safe = self.nlp.p.get_val(self.nlp.phase_name + '.timeseries.r_safe')*self.body.R

    def plot(self):

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 self.controls_exp, r_safe=self.r_safe)
        sol_plot.plot()
