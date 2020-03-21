"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from smt.sampling_methods import LHS, FullFactorial
from smt.surrogate_models import IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC

from latom.utils.spacecraft import Spacecraft, ImpulsiveBurn
from latom.utils.keplerian_orbit import TwoDimOrb
from latom.nlp.nlp_2d import TwoDimAscConstNLP, TwoDimAscVarNLP, TwoDimAscVToffNLP, TwoDimDescTwoPhasesNLP,\
    TwoDimDescConstNLP, TwoDimDescVarNLP, TwoDimDescVLandNLP
from latom.guess.guess_2d import HohmannTransfer
from latom.plots.response_surfaces import RespSurf
from latom.data.smt.data_smt import dirname_smt
from latom.utils.pickle_utils import save, load


class SurrogateModel:

    def __init__(self, train_method, rec_file=None):

        self.limits = self.x_samp = self.m_prop = self.failures = self.d = None
        self.trained = None

        if rec_file is not None:
            self.load(rec_file)
            self.train(train_method)

    @staticmethod
    def abs_path(rec_file):

        return '/'.join([dirname_smt, rec_file])

    def load(self, rec_file):

        self.d = load(self.abs_path(rec_file))
        self.limits = self.d['limits']
        self.x_samp = self.d['x_samp']
        self.m_prop = self.d['m_prop']
        self.failures = self.d['failures']

    def save(self, rec_file):

        d = {'limits': self.limits, 'x_samp': self.x_samp, 'm_prop': self.m_prop, 'failures': self.failures}
        save(d, self.abs_path(rec_file))

    def compute_grid(self, isp_lim, twr_lim, nb_samp, samp_method='full', criterion='m'):

        self.limits = np.vstack((np.asarray(isp_lim), np.asarray(twr_lim)))

        if samp_method == 'lhs':
            samp = LHS(xlimits=self.limits, criterion=criterion)
        elif samp_method == 'full':
            samp = FullFactorial(xlimits=self.limits)
        else:
            raise ValueError('samp_method must be either lhs or full')

        self.x_samp = samp(nb_samp)
        self.m_prop = np.zeros((nb_samp, 1))
        self.failures = np.zeros((nb_samp, 1))

    @staticmethod
    def solve(nlp, i):

        print(f"\nIteration {i}\nIsp: {nlp.sc.Isp:.6f} s\ttwr: {nlp.sc.twr:.6f}")
        f = nlp.p.run_driver()
        print("\nFailure: {0}".format(f))

        if isinstance(nlp.phase_name, str):
            phase_name = nlp.phase_name
        else:
            phase_name = nlp.phase_name[-1]

        m_prop = 1.0 - nlp.p.get_val(phase_name + '.timeseries.states:m')[-1, -1]
        nlp.cleanup()

        return m_prop, f

    def train(self, train_method, **kwargs):

        if train_method == 'IDW':
            self.trained = IDW(**kwargs)
        elif train_method == 'KPLS':
            self.trained = KPLS(**kwargs)
        elif train_method == 'KPLSK':
            self.trained = KPLSK(**kwargs)
        elif train_method == 'KRG':
            self.trained = KRG(**kwargs)
        elif train_method == 'LS':
            self.trained = LS(**kwargs)
        elif train_method == 'QP':
            self.trained = QP(**kwargs)
        elif train_method == 'RBF':
            self.trained = RBF(**kwargs)
        elif train_method == 'RMTB':
            self.trained = RMTB(xlimits=self.limits, **kwargs)
        elif train_method == 'RMTC':
            self.trained = RMTC(xlimits=self.limits, **kwargs)
        else:
            raise ValueError('train_method must be one between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC')

        self.trained.set_training_values(self.x_samp, self.m_prop)
        self.trained.train()

    def evaluate(self, isp, twr):

        if isinstance(isp, float):
            isp = [isp]
        if isinstance(twr, float):
            twr = [twr]

        x_eval = np.hstack((np.reshape(isp, (len(isp), 1)), np.reshape(twr, (len(twr), 1))))
        m_eval = self.trained.predict_values(x_eval)

        return m_eval

    def compute_matrix(self, nb_eval=None):

        if nb_eval is not None:  # LHS
            samp_eval = FullFactorial(xlimits=self.limits)
            x_eval = samp_eval(nb_eval)
            m_prop_eval = self.trained.predict_values(x_eval)

        else:  # Full-Factorial
            nb_eval = np.size(self.m_prop)
            x_eval = deepcopy(self.x_samp)
            m_prop_eval = deepcopy(self.m_prop)

        isp = np.unique(x_eval[:, 0])
        twr = np.unique(x_eval[:, 1])
        n = int(np.sqrt(nb_eval))
        m_mat = np.reshape(m_prop_eval, (n, n))

        return isp, twr, m_mat

    def plot(self, nb_eval=None, nb_lines=50, kind='prop'):

        isp, twr, m_mat = self.compute_matrix(nb_eval=nb_eval)

        if kind == 'prop':
            surf_plot = RespSurf(isp, twr, m_mat, 'Propellant fraction', nb_lines=nb_lines)
        elif kind == 'final':
            surf_plot = RespSurf(isp, twr, (1 - m_mat), 'Final/initial mass ratio', nb_lines=nb_lines)
        else:
            raise ValueError('kind must be either prop or final')

        surf_plot.plot()

        plt.show()


class TwoDimAscConstSurrogate(SurrogateModel):

    def sampling(self, body, isp_lim, twr_lim, alt, theta, tof, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='full', criterion='m', snopt_opts=None, u_bound='lower'):

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            nlp = TwoDimAscConstNLP(body, sc, alt, theta, (-np.pi/2, np.pi/2), tof, t_bounds, method, nb_seg, order,
                                    solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimAscVarSurrogate(SurrogateModel):

    def sampling(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='full', criterion='m', snopt_opts=None, u_bound='lower'):

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            nlp = TwoDimAscVarNLP(body, sc, alt, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg, order, solver,
                                  'powered', snopt_opts=snopt_opts, u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimAscVToffSurrogate(SurrogateModel):

    def sampling(self, body, isp_lim, twr_lim, alt, alt_safe, slope, t_bounds, method, nb_seg, order, solver,
                 nb_samp, samp_method='lhs', criterion='m', snopt_opts=None, u_bound='lower'):

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            nlp = TwoDimAscVToffNLP(body, sc, alt, alt_safe, slope, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg,
                                    order, solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimDescConstSurrogate(SurrogateModel):

    def sampling(self, body, isp_lim, twr_lim, alt, alt_p, theta, tof, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='lhs', criterion='m', snopt_opts=None, u_bound='upper'):

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        ht = HohmannTransfer(body.GM, TwoDimOrb(body.GM, a=(body.R + alt), e=0.0),
                             TwoDimOrb(body.GM, a=(body.R + alt_p), e=0.0))

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            deorbit_burn = ImpulsiveBurn(sc, ht.dva)
            nlp = TwoDimDescConstNLP(body, deorbit_burn.sc, alt_p, ht.transfer.vp, theta, (0, 3/2*np.pi),  tof,
                                     t_bounds, method, nb_seg, order, solver, 'powered', snopt_opts=snopt_opts,
                                     u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimDescVarSurrogate(SurrogateModel):

    def sampling(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp, samp_method='lhs',
                 criterion='m', snopt_opts=None, u_bound='upper'):

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            nlp = TwoDimDescVarNLP(body, sc, alt, (0.0, 3/2*np.pi), t_bounds, method, nb_seg, order, solver, 'powered',
                                   snopt_opts=snopt_opts, u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimDescVLandSurrogate(SurrogateModel):

    def sampling(self, body, isp_lim, twr_lim, alt, alt_safe, slope, t_bounds, method, nb_seg, order, solver,
                 nb_samp, samp_method='lhs', criterion='m', snopt_opts=None, u_bound='upper'):

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            nlp = TwoDimDescVLandNLP(body, sc, alt, alt_safe, slope, (0.0, 3/2*np.pi), t_bounds, method, nb_seg, order,
                                     solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimDescVertSurrogate(SurrogateModel):

    def sampling(self, body, isp_lim, twr_lim, alt, alt_p, alt_switch, theta, tof, t_bounds, method, nb_seg, order,
                 solver, nb_samp, samp_method='lhs', criterion='m', snopt_opts=None):

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        ht = HohmannTransfer(body.GM, TwoDimOrb(body.GM, a=(body.R + alt), e=0.0),
                             TwoDimOrb(body.GM, a=(body.R + alt_p), e=0.0))

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            deorbit_burn = ImpulsiveBurn(sc, ht.dva)
            nlp = TwoDimDescTwoPhasesNLP(body, deorbit_burn.sc, alt, alt_switch, ht.transfer.vp, theta, (0.0, np.pi),
                                         tof, t_bounds, method, nb_seg, order, solver, ('free', 'vertical'),
                                         snopt_opts=snopt_opts)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)
