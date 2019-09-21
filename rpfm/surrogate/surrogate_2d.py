"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS, FullFactorial
from smt.surrogate_models import RMTB, RMTC

from rpfm.utils.spacecraft import Spacecraft
from rpfm.nlp.nlp_2d import TwoDimAscConstNLP, TwoDimAscVarNLP, TwoDimAscVToffNLP, TwoDimDescTwoPhasesNLP
from rpfm.guess.guess_2d import HohmannTransfer, DeorbitBurn
from rpfm.plots.response_surfaces import RespSurf


class SurrogateModel:

    def __init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='lhs', criterion='c', nb_eval=100, snopt_opts=None):

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
        self.nb_eval = nb_eval
        self.snopt_opts = snopt_opts

        # training models
        self.train_mass = self.train_time = None

        # sampling, evaluation and matrices values
        self.tof_samp = np.zeros((nb_samp, 1))
        self.m_samp = np.zeros((nb_samp, 1))

        self.x_eval = self.m_eval = self.tof_eval = None
        self.isp = self.twr = self.m_mat = self.tof_mat = None

        # sampling grid
        if samp_method == 'lhs':
            samp = LHS(xlimits=self.limits, criterion=criterion)
            self.x_samp = samp(nb_samp)

        elif samp_method == 'full':
            samp = FullFactorial(xlimits=self.limits)
            self.x_samp = samp(nb_samp**2)
        else:
            raise ValueError('samp_method must be one of rand, lhs, full')

        self.samp_method = samp_method
        self.surf_plot = None

    def solve(self, nlp, i):

        nlp.p.run_driver()

        if isinstance(nlp.phase_name, str):
            phase_name = nlp.phase_name
        else:
            phase_name = nlp.phase_name[-1]

        self.m_samp[i, 0] = nlp.p.get_val(phase_name + '.timeseries.states:m')[-1, -1]
        self.tof_samp[i, 0] = nlp.p.get_val(phase_name + '.time')[-1]*self.body.tc

        nlp.cleanup()

    def train(self, train_method, **kwargs):

        if train_method == 'RMTB':
            self.train_mass = RMTB(xlimits=self.limits, **kwargs)
            self.train_time = RMTB(xlimits=self.limits, **kwargs)
        elif train_method == 'RMTC':
            self.train_mass = RMTC(xlimits=self.limits, **kwargs)
            self.train_time = RMTC(xlimits=self.limits, **kwargs)
        else:
            raise ValueError('train_method must be RMTB or RMTC')

        self.train_mass.set_training_values(self.x_samp, self.m_samp[:, 0])
        self.train_time.set_training_values(self.x_samp, self.tof_samp[:, 0])

        self.train_mass.train()
        self.train_time.train()

    def evaluate(self):

        if self.samp_method == 'full':

            self.x_eval = self.x_samp
            self.m_eval = self.m_samp
            self.tof_eval = self.tof_samp

        else:

            samp_eval = FullFactorial(xlimits=self.limits)

            self.x_eval = samp_eval(self.nb_eval**2)
            self.m_eval = self.train_mass.predict_values(self.x_eval)
            self.tof_eval = self.train_time.predict_values(self.x_eval)

        self.isp = np.unique(self.x_eval[:, 0])
        self.twr = np.unique(self.x_eval[:, 1])
        self.m_mat = np.reshape(self.m_eval, (self.nb_eval, self.nb_eval))
        self.tof_mat = np.reshape(self.tof_eval, (self.nb_eval, self.nb_eval))

    def plot(self):

        self.surf_plot = RespSurf(self.isp, self.twr, self.m_mat, self.tof_mat)
        self.surf_plot.plot()

        plt.show()


class TwoDimAscConstSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, theta, tof, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='lhs', criterion='c', nb_eval=100, snopt_opts=None):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, nb_eval=nb_eval, snopt_opts=snopt_opts)

        self.theta = theta
        self.tof = tof

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)
            nlp = TwoDimAscConstNLP(self.body, sc, self.alt, self.theta, (-np.pi/2, np.pi/2), self.tof, self.t_bounds,
                                    self.method, self.nb_seg, self.order, self.solver, 'powered',
                                    snopt_opts=self.snopt_opts, u_bound=True)

            self.solve(nlp, i)


class TwoDimAscVarSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp, samp_method='lhs',
                 criterion='c', nb_eval=100, snopt_opts=None):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, nb_eval=nb_eval, snopt_opts=snopt_opts)

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)
            nlp = TwoDimAscVarNLP(self.body, sc, self.alt, (-np.pi/2, np.pi/2), self.t_bounds, self.method, self.nb_seg,
                                  self.order, self.solver, 'powered', snopt_opts=self.snopt_opts, u_bound=True)

            self.solve(nlp, i)


class TwoDimAscVToffSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, alt_safe, slope, t_bounds, method, nb_seg, order, solver,
                 nb_samp, samp_method='lhs', criterion='c', nb_eval=100, snopt_opts=None):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, nb_eval=nb_eval, snopt_opts=snopt_opts)

        self.alt_safe = alt_safe
        self.slope = slope

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)
            nlp = TwoDimAscVToffNLP(self.body, sc, self.alt, self.alt_safe, self.slope, (-np.pi/2, np.pi/2),
                                    self.t_bounds, self.method, self.nb_seg, self.order, self.solver, 'powered',
                                    snopt_opts=self.snopt_opts, u_bound=True)

            self.solve(nlp, i)


class TwoDimDescVertSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, alt_p, alt_switch, theta, tof, t_bounds, method, nb_seg, order,
                 solver, nb_samp, samp_method='lhs', criterion='c', nb_eval=100, snopt_opts=None):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt_p, t_bounds, method, nb_seg, order, solver, nb_samp,
                                samp_method=samp_method, criterion=criterion, nb_eval=nb_eval, snopt_opts=snopt_opts)

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
