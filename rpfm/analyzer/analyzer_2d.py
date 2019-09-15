"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.analyzer.analyzer import Analyzer
from rpfm.nlp.nlp_2d import TwoDimAscConstNLP, TwoDimAscVarNLP, TwoDimAscVToffNLP, TwoDimDescConstNLP,\
    TwoDimDescTwoPhasesNLP
from rpfm.plots.solutions import TwoDimSolPlot, TwoDimTwoPhasesSolPlot
from rpfm.utils.const import states_2d
from rpfm.guess.guess_2d import HohmannTransfer, DeorbitBurn


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

        self.nlp = TwoDimAscConstNLP(body, sc, alt, theta, (-np.pi/2, np.pi/2), tof, t_bounds, method, nb_seg, order,
                                     solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                     check_partials=check_partials, u_bound=u_bound)

    def get_time_series(self, p):

        tof, t, states, alpha = self.get_states_alpha_time_series(p)
        thrust = self.sc.T_max*np.ones((len(t), 1))
        controls = np.hstack((thrust, alpha))

        return tof, t, states, controls

    def plot(self):

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 threshold=None)
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

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp)
        sol_plot.plot()


class TwoDimAscVToffAnalyzer(TwoDimAscVarAnalyzer):

    def __init__(self, body, sc, alt, alt_safe, slope, t_bounds, method, nb_seg, order, solver, snopt_opts=None,
                 rec_file=None, check_partials=False, u_bound=False):

        TwoDimAnalyzer.__init__(self, body, sc)

        self.nlp = TwoDimAscVToffNLP(body, sc, alt, alt_safe, slope, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg,
                                     order, solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                     check_partials=check_partials, u_bound=u_bound)

        self.r_safe = None

    def get_solutions(self, explicit=True):
        
        TwoDimAnalyzer.get_solutions(self, explicit=explicit)

        self.r_safe = self.nlp.p.get_val(self.nlp.phase_name + '.timeseries.r_safe')*self.body.R

    def plot(self):

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 r_safe=self.r_safe)
        sol_plot.plot()


class TwoDimDescConstAnalyzer(TwoDimAnalyzer):

    def __init__(self, body, sc, alt, alt_p, theta, tof, t_bounds, method, nb_seg, order, solver, snopt_opts=None,
                 rec_file=None, check_partials=False):

        self.ht = HohmannTransfer(body.GM, (body.R + alt), (body.R + alt_p))
        self.deorbit_burn = DeorbitBurn(sc, self.ht.dva)

        TwoDimAnalyzer.__init__(self, body, sc)

        self.nlp = TwoDimDescConstNLP(body, self.deorbit_burn.sc, alt_p, self.ht.vp, theta, (0.0, np.pi), tof,
                                      t_bounds, method, nb_seg, order, solver, self.phase_name, snopt_opts=snopt_opts,
                                      rec_file=rec_file, check_partials=check_partials)

    def get_time_series(self, p):

        tof, t, states, alpha = self.get_states_alpha_time_series(p)
        thrust = self.sc.T_max*np.ones((len(t), 1))
        controls = np.hstack((thrust, alpha))

        return tof, t, states, controls

    def plot(self):

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 threshold=None, kind='descent')
        sol_plot.plot()


class TwoDimDescTwoPhasesAnalyzer(Analyzer):

    def __init__(self, body, sc, alt, alt_p, alt_switch, theta, tof, t_bounds, method, nb_seg, order, solver,
                 snopt_opts=None, rec_file=None, check_partials=False, fix='alt'):

        Analyzer.__init__(self, body, sc)

        self.phase_name = ('free', 'vertical')
        self.states_scalers = np.array([self.body.R, 1.0, self.body.vc, self.body.vc, 1.0])

        self.ht = HohmannTransfer(body.GM, (body.R + alt), (body.R + alt_p))
        self.deorbit_burn = DeorbitBurn(sc, self.ht.dva)

        self.nlp = TwoDimDescTwoPhasesNLP(body, self.deorbit_burn.sc, alt_p, alt_switch, self.ht.vp, theta,
                                          (0.0, np.pi), tof, t_bounds, method, nb_seg, order, solver, self.phase_name,
                                          snopt_opts=snopt_opts, rec_file=rec_file, check_partials=check_partials,
                                          fix=fix)

    def get_time_series(self, p):

        # attitude free
        tof_free = float(p.get_val(self.nlp.phase_name[0] + '.t_duration'))*self.body.tc
        t_free = p.get_val(self.nlp.phase_name[0] + '.timeseries.time')*self.body.tc

        states_free = np.empty((np.size(t_free), 0))

        for k in states_2d:
            s = p.get_val(self.nlp.phase_name[0] + '.timeseries.states:' + k)
            states_free = np.append(states_free, s, axis=1)

        states_free = states_free*self.states_scalers

        alpha_free = p.get_val(self.nlp.phase_name[0] + '.timeseries.controls:alpha')
        controls_free = np.hstack((self.sc.T_max*np.ones((np.size(t_free), 1)), alpha_free))

        # vertical
        tof_vertical = float(p.get_val(self.nlp.phase_name[1] + '.t_duration'))*self.body.tc
        t_vertical = p.get_val(self.nlp.phase_name[1] + '.timeseries.time')*self.body.tc

        r_vertical = p.get_val(self.nlp.phase_name[1] + '.timeseries.states:r')*self.body.R
        theta_vertical = states_free[-1, 1] * np.ones((np.size(t_vertical), 1))
        u_vertical = p.get_val(self.nlp.phase_name[1] + '.timeseries.states:u')*self.body.vc
        m_vertical = p.get_val(self.nlp.phase_name[1] + '.timeseries.states:m')

        states_vertical = np.hstack((r_vertical, theta_vertical, u_vertical, np.zeros((np.size(t_vertical), 1)),
                                     m_vertical))

        controls_vertical = np.hstack((self.sc.T_max*np.ones((np.size(t_vertical), 1)),
                                       np.pi/2*np.ones((np.size(t_vertical), 1))))

        tof = [tof_free, tof_vertical]
        t = [t_free, t_vertical]
        states = [states_free, states_vertical]
        controls = [controls_free, controls_vertical]

        return tof, t, states, controls

    def plot(self):

        sol_plot = TwoDimTwoPhasesSolPlot(self.body.R, self.time, self.states, self.controls, time_exp=self.time_exp,
                                          states_exp=self.states_exp, kind='descent')
        sol_plot.plot()
