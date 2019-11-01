"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
from copy import deepcopy

from rpfm.analyzer.analyzer_2d import TwoDimAscAnalyzer, TwoDimAnalyzer
from rpfm.nlp.nlp_heo_2d import TwoDimLLO2HEONLP, TwoDimLLO2ApoNLP, TwoDim3PhasesLLO2HEONLP, TwoDim2PhasesLLO2HEONLP
from rpfm.plots.solutions import TwoDimSolPlot
from rpfm.utils.keplerian_orbit import TwoDimOrb
from rpfm.guess.guess_2d import ImpulsiveBurn
from rpfm.utils.const import g0


class TwoDimLLO2HEOAnalyzer(TwoDimAscAnalyzer):

    def __init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False, u_bound='lower'):

        TwoDimAscAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimLLO2HEONLP(body, sc, alt, rp, t, (-np.pi / 2, np.pi / 2), t_bounds, method, nb_seg, order,
                                    solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                    check_partials=check_partials, u_bound=u_bound)

    def plot(self):

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 a=self.nlp.guess.ht.arrOrb.a, e=self.nlp.guess.ht.arrOrb.e)
        sol_plot.plot()

    def __str__(self):

        lines = ['\n{:^50s}'.format('2D Transfer trajectory from LLO to HEO:'),
                 self.nlp.guess.__str__(),
                 '\n{:^50s}'.format('Optimal transfer:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1. - self.states[-1, -1]/self.sc.m0, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight:', self.tof, 's'),
                 TwoDimAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s


class TwoDimLLO2ApoAnalyzer(TwoDimAscAnalyzer):

    def __init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False, u_bound='lower'):

        TwoDimAscAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimLLO2ApoNLP(body, sc, alt, rp, t, (-np.pi / 2, np.pi / 2), t_bounds, method, nb_seg, order,
                                    solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                    check_partials=check_partials)

        self.transfer = self.insertion_burn = self.dv = None

    def compute_insertion_burn(self):

        rf = self.states[-1, 0]
        uf = self.states[-1, 2]
        vf = self.states[-1, 3]
        mf = self.states[-1, -1]

        ht = rf*vf
        at = self.body.GM*rf/(2*self.body.GM - rf*(uf**2 + vf**2))
        et = (1 - ht**2/self.body.GM/at)**0.5

        sc = deepcopy(self.sc)
        sc.m0 = mf

        self.dv = self.sc.Isp*g0*np.log(self.sc.m0/mf)
        self.transfer = TwoDimOrb(self.body.GM, a=at, e=et)
        self.insertion_burn = ImpulsiveBurn(sc, self.nlp.guess.ht.arrOrb.va - self.transfer.va)

    def plot(self):

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 threshold=None, a=self.nlp.guess.ht.arrOrb.a, e=self.nlp.guess.ht.arrOrb.e)
        sol_plot.plot()

    def __str__(self):

        lines = ['\n{:^50s}'.format('2D Transfer trajectory from LLO to HEO:'),
                 self.nlp.guess.__str__(),
                 '\n{:^50s}'.format('Coasting orbit:'),
                 self.transfer.__str__(),
                 '\n{:^50s}'.format('Optimal transfer:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:',
                                                   1 - self.insertion_burn.mf/self.sc.m0, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight (burn):', self.tof, 's'),
                 '\n{:^50s}'.format('Departure burn:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.nlp.guess.pow.dv_inf, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Finite dV:', self.dv, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1 - self.states[-1, -1]/self.sc.m0, ''),
                 '\n{:^50s}'.format('Injection burn:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.insertion_burn.dv, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', self.insertion_burn.dm/self.sc.m0, ''),
                 TwoDimAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s


class TwoDim2PhasesLLO2HEOAnalyzer(TwoDimAnalyzer):

    def __init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False):

        TwoDimAnalyzer.__init__(self, body, sc)

        self.phase_name = ('dep', 'arr')
        self.nlp = TwoDim2PhasesLLO2HEONLP(body, sc, alt, rp, t, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg, order,
                                           solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                           check_partials=check_partials)

    def get_time_series(self, p, scaled=False):

        tof = []
        time = []
        states = []
        controls = []

        for i in range(2):

            tof.append(float(p.get_val(self.nlp.phase_name[i] + '.t_duration'))*self.body.tc)
            t = p.get_val(self.nlp.phase_name[i] + '.timeseries.time')*self.body.tc
            time.append(t)

            r = p.get_val(self.nlp.phase_name[i] + '.timeseries.states:r')*self.body.R
            theta = p.get_val(self.nlp.phase_name[i] + '.timeseries.states:theta')
            u = p.get_val(self.nlp.phase_name[i] + '.timeseries.states:u')*self.body.vc
            v = p.get_val(self.nlp.phase_name[i] + '.timeseries.states:v')*self.body.vc
            m = p.get_val(self.nlp.phase_name[i] + '.timeseries.states:m')
            alpha = p.get_val(self.nlp.phase_name[i] + '.timeseries.controls:alpha')
            thrust = self.nlp.sc.T_max*np.ones((len(t), 1))

            s = np.hstack((r, theta, u, v, m))
            c = np.hstack((thrust, alpha))

            states.append(s)
            controls.append(c)

        return tof, time, states, controls

    def plot(self):

        time = np.vstack(self.time)
        states = np.vstack(self.states)

        num = states[-1, 2]*states[-1, 0]*states[-1, 3]
        den = states[-1, 0]*states[-1, 3]**2 - self.body.GM
        theta_i = np.arctan2(num, den)
        states[:, 1] = states[:, 1] - states[-1, 1] + theta_i

        controls = np.vstack(self.controls)

        if self.time_exp is not None:
            time_exp = np.vstack(self.time_exp)
            states_exp = np.vstack(self.states_exp)
        else:
            time_exp = None
            states_exp = None

        sol_plot = TwoDimSolPlot(self.body.R, time, states, controls, time_exp, states_exp, threshold=1e-6,
                                 a=self.nlp.guess.ht.arrOrb.a, e=self.nlp.guess.ht.arrOrb.e)
        sol_plot.plot()


class TwoDim3PhasesLLO2HEOAnalyzer(TwoDimAnalyzer):

    def __init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False):

        TwoDimAnalyzer.__init__(self, body, sc)

        self.phase_name = ('dep', 'coast', 'arr')
        self.nlp = TwoDim3PhasesLLO2HEONLP(body, sc, alt, rp, t, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg, order,
                                           solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                           check_partials=check_partials)

    def get_time_series(self, p):

        tof = []
        time = []
        states = []
        controls = []

        for i in range(3):

            tof.append(float(p.get_val(self.nlp.phase_name[i] + '.t_duration'))*self.body.tc)
            t = p.get_val(self.nlp.phase_name[i] + '.timeseries.time')*self.body.tc
            time.append(t)

            r = p.get_val(self.nlp.phase_name[i] + '.timeseries.states:r')*self.body.R
            theta = p.get_val(self.nlp.phase_name[i] + '.timeseries.states:theta')
            u = p.get_val(self.nlp.phase_name[i] + '.timeseries.states:u')*self.body.vc
            v = p.get_val(self.nlp.phase_name[i] + '.timeseries.states:v')*self.body.vc

            if i == 1:
                m = states[0][-1, -1]*np.ones((len(t), 1))
                alpha = np.zeros((len(t), 1))
                thrust = np.zeros((len(t), 1))
            else:
                m = p.get_val(self.nlp.phase_name[i] + '.timeseries.states:m')
                alpha = p.get_val(self.nlp.phase_name[i] + '.timeseries.controls:alpha')
                thrust = self.nlp.sc.T_max*np.ones((len(t), 1))

            s = np.hstack((r, theta, u, v, m))
            c = np.hstack((thrust, alpha))

            states.append(s)
            controls.append(c)

        return tof, time, states, controls

    def plot(self):

        time = np.vstack(self.time)
        states = np.vstack(self.states)

        num = states[-1, 2]*states[-1, 0]*states[-1, 3]
        den = states[-1, 0]*states[-1, 3]**2 - self.body.GM
        theta_i = np.arctan2(num, den)
        states[:, 1] = states[:, 1] - states[-1, 1] + theta_i

        controls = np.vstack(self.controls)

        if self.time_exp is not None:
            time_exp = np.vstack(self.time_exp)
            states_exp = np.vstack(self.states_exp)
        else:
            time_exp = None
            states_exp = None

        sol_plot = TwoDimSolPlot(self.body.R, time, states, controls, time_exp, states_exp, threshold=1e-6,
                                 a=self.nlp.guess.ht.arrOrb.a, e=self.nlp.guess.ht.arrOrb.e)
        sol_plot.plot()
