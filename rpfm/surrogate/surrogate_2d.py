"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from smt.sampling_methods import LHS, FullFactorial
from smt.surrogate_models import IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC

from rpfm.utils.spacecraft import Spacecraft
from rpfm.nlp.nlp_2d import TwoDimAscConstNLP, TwoDimAscVarNLP, TwoDimAscVToffNLP, TwoDimDescTwoPhasesNLP,\
    TwoDimDescConstNLP, TwoDimDescVarNLP, TwoDimDescVLandNLP
from rpfm.guess.guess_2d import HohmannTransfer, DeorbitBurn
from rpfm.plots.response_surfaces import RespSurf


class SurrogateModel:

    def __init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='lhs', criterion='m', snopt_opts=None, u_bound=None):

        # parameters
        self.body = body
        self.limits = np.vstack((np.asarray(isp_lim), np.asarray(twr_lim)))
        self.alt = alt
        self.t_bounds = t_bounds
        self.method = method
        self.nb_seg = nb_seg
        self.order = order
        self.solver = solver
        self.nb_samp = nb_samp
        self.snopt_opts = snopt_opts
        self.u_bound = u_bound

        # training models
        self.train_mass = self.train_tof = None

        # sampling, evaluation and matrices values
        self.failures = []
        self.tof_samp = np.zeros((nb_samp, 1))
        self.m_samp = np.zeros((nb_samp, 1))

        self.nb_eval = self.samp_eval = self.x_eval = self.m_eval = self.tof_eval = None
        self.isp = self.twr = self.m_mat = self.tof_mat = None

        # sampling grid
        if samp_method == 'lhs':
            self.samp = LHS(xlimits=self.limits, criterion=criterion)
        elif samp_method == 'full':
            self.samp = FullFactorial(xlimits=self.limits)
            self.nb_eval = self.nb_samp
        else:
            raise ValueError('samp_method must be either lhs or full')

        self.x_samp = self.samp(self.nb_samp)
        self.surf_plot = None

    def solve(self, nlp, i):

        print('\nIteration:', i, '\n')

        f = nlp.p.run_driver()

        print('\nFailure:', f, '\n')

        if isinstance(nlp.phase_name, str):
            phase_name = nlp.phase_name
        else:
            phase_name = nlp.phase_name[-1]

        self.m_samp[i, 0] = nlp.p.get_val(phase_name + '.timeseries.states:m')[-1, -1]
        self.tof_samp[i, 0] = nlp.p.get_val(phase_name + '.time')[-1]*self.body.tc
        self.failures.append(f)

        nlp.cleanup()

    def train(self, train_method, **kwargs):

        if train_method == 'IDW':
            self.train_mass = IDW(**kwargs)
            self.train_tof = IDW(**kwargs)
        elif train_method == 'KPLS':
            self.train_mass = KPLS(**kwargs)
            self.train_tof = KPLS(**kwargs)
        elif train_method == 'KPLSK':
            self.train_mass = KPLSK(**kwargs)
            self.train_tof = KPLSK(**kwargs)
        elif train_method == 'KRG':
            self.train_mass = KRG(**kwargs)
            self.train_tof = KRG(**kwargs)
        elif train_method == 'LS':
            self.train_mass = LS(**kwargs)
            self.train_tof = LS(**kwargs)
        elif train_method == 'QP':
            self.train_mass = QP(**kwargs)
            self.train_tof = QP(**kwargs)
        elif train_method == 'RBF':
            self.train_mass = RBF(**kwargs)
            self.train_tof = RBF(**kwargs)
        elif train_method == 'RMTB':
            self.train_mass = RMTB(xlimits=self.limits, **kwargs)
            self.train_tof = RMTB(xlimits=self.limits, **kwargs)
        elif train_method == 'RMTC':
            self.train_mass = RMTC(xlimits=self.limits, **kwargs)
            self.train_tof = RMTC(xlimits=self.limits, **kwargs)
        else:
            raise ValueError('train_method must be one between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC')

        self.train_mass.set_training_values(self.x_samp, self.m_samp[:, 0])
        self.train_tof.set_training_values(self.x_samp, self.tof_samp[:, 0])

        self.train_mass.train()
        self.train_tof.train()

    def evaluate(self, **kwargs):

        if ('isp' in kwargs) and ('twr' in kwargs):  # single values

            isp = kwargs['isp']
            twr = kwargs['twr']

            if isinstance(isp, float):
                isp = [isp]
            if isinstance(twr, float):
                twr = [twr]

            x_eval = np.hstack((np.reshape(isp, (len(isp), 1)), np.reshape(twr, (len(twr), 1))))
            m_eval = self.train_mass.predict_values(x_eval)
            tof_eval = self.train_tof.predict_values(x_eval)

            return m_eval, tof_eval

        else:  # full grid

            if 'nb_eval' in kwargs:

                self.nb_eval = kwargs['nb_eval']

                self.samp_eval = FullFactorial(xlimits=self.limits)

                self.x_eval = self.samp_eval(self.nb_eval)
                self.m_eval = self.train_mass.predict_values(self.x_eval)
                self.tof_eval = self.train_tof.predict_values(self.x_eval)

            elif self.nb_eval is not None:

                self.x_eval = deepcopy(self.x_samp)
                self.m_eval = deepcopy(self.m_samp)
                self.tof_eval = deepcopy(self.tof_samp)

            else:
                raise ValueError('Surrogate model built with LHS sampling method.'
                                 '\nThe two arrays isp, twr or nb_eval must be provided')

            self.isp = np.unique(self.x_eval[:, 0])
            self.twr = np.unique(self.x_eval[:, 1])

            n = int(np.sqrt(self.nb_eval))

            self.m_mat = np.reshape(self.m_eval, (n, n))
            self.tof_mat = np.reshape(self.tof_eval, (n, n))

    def plot(self):

        self.surf_plot = RespSurf(self.isp, self.twr, self.m_mat, self.tof_mat)
        self.surf_plot.plot()

        plt.show()


class TwoDimAscConstSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, theta, tof, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='lhs', criterion='m', snopt_opts=None, u_bound='lower'):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts, u_bound=u_bound)

        self.theta = theta
        self.tof = tof

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)
            nlp = TwoDimAscConstNLP(self.body, sc, self.alt, self.theta, (-np.pi/2, np.pi/2), self.tof, self.t_bounds,
                                    self.method, self.nb_seg, self.order, self.solver, 'powered',
                                    snopt_opts=self.snopt_opts, u_bound=self.u_bound)

            self.solve(nlp, i)


class TwoDimAscVarSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp, samp_method='lhs',
                 criterion='m', snopt_opts=None, u_bound='lower'):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts, u_bound=u_bound)

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)
            nlp = TwoDimAscVarNLP(self.body, sc, self.alt, (-np.pi/2, np.pi/2), self.t_bounds, self.method, self.nb_seg,
                                  self.order, self.solver, 'powered', snopt_opts=self.snopt_opts, u_bound=self.u_bound)

            self.solve(nlp, i)


class TwoDimAscVToffSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, alt_safe, slope, t_bounds, method, nb_seg, order, solver,
                 nb_samp, samp_method='lhs', criterion='m', snopt_opts=None, u_bound='lower'):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts, u_bound=u_bound)

        self.alt_safe = alt_safe
        self.slope = slope

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)
            nlp = TwoDimAscVToffNLP(self.body, sc, self.alt, self.alt_safe, self.slope, (-np.pi/2, np.pi/2),
                                    self.t_bounds, self.method, self.nb_seg, self.order, self.solver, 'powered',
                                    snopt_opts=self.snopt_opts, u_bound=self.u_bound)

            self.solve(nlp, i)


class TwoDimDescConstSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, alt_p, theta, tof, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='lhs', criterion='m', snopt_opts=None, u_bound='upper'):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts, u_bound=u_bound)

        self.ht = HohmannTransfer(body.GM, (body.R + alt), (body.R + alt_p))
        self.alt_p = alt_p
        self.theta = theta
        self.tof = tof

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)
            deorbit_burn = DeorbitBurn(sc, self.ht.dva)
            nlp = TwoDimDescConstNLP(self.body, deorbit_burn.sc, self.alt_p, self.ht.vp, self.theta, (0, 3/2*np.pi),
                                     self.tof, self.t_bounds, self.method, self.nb_seg, self.order, self.solver,
                                     'powered', snopt_opts=self.snopt_opts, u_bound=self.u_bound)

            self.solve(nlp, i)


class TwoDimDescVarSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp, samp_method='lhs',
                 criterion='m', snopt_opts=None, u_bound='upper'):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts, u_bound=u_bound)

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)
            nlp = TwoDimDescVarNLP(self.body, sc, self.alt, (0.0, 3/2*np.pi), self.t_bounds, self.method, self.nb_seg,
                                   self.order, self.solver, 'powered', snopt_opts=self.snopt_opts, u_bound=self.u_bound)

            self.solve(nlp, i)


class TwoDimDescVLandSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, alt_safe, slope, t_bounds, method, nb_seg, order, solver,
                 nb_samp, samp_method='lhs', criterion='m', snopt_opts=None, u_bound='upper'):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts, u_bound=u_bound)

        self.alt_safe = alt_safe
        self.slope = slope

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)
            nlp = TwoDimDescVLandNLP(self.body, sc, self.alt, self.alt_safe, self.slope, (0.0, 3/2*np.pi),
                                     self.t_bounds, self.method, self.nb_seg, self.order, self.solver, 'powered',
                                     snopt_opts=self.snopt_opts, u_bound=self.u_bound)

            self.solve(nlp, i)


class TwoDimDescVertSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, alt_p, alt_switch, theta, tof, t_bounds, method, nb_seg, order,
                 solver, nb_samp, samp_method='lhs', criterion='m', snopt_opts=None):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt_p, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts)

        self.ht = HohmannTransfer(body.GM, (body.R + alt), (body.R + alt_p))
        self.alt_switch = alt_switch
        self.theta = theta
        self.tof = tof

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)

            deorbit_burn = DeorbitBurn(sc, self.ht.dva)

            nlp = TwoDimDescTwoPhasesNLP(self.body, deorbit_burn.sc, self.alt, self.alt_switch, self.ht.vp, self.theta,
                                         (0.0, np.pi), self.tof, self.t_bounds, self.method, self.nb_seg, self.order,
                                         self.solver, ('free', 'vertical'), snopt_opts=self.snopt_opts)

            self.solve(nlp, i)
