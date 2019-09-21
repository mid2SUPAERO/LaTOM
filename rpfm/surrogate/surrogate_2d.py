"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
from smt.sampling_methods import LHS, Random, FullFactorial

from rpfm.utils.spacecraft import Spacecraft
from rpfm.nlp.nlp_2d import TwoDimAscVarNLP, TwoDimAscVToffNLP, TwoDimDescTwoPhasesNLP


class SurrogateModel:

    def __init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                 snopt_opts=None, samp_method='lhs', criterion='c'):

        self.body = body
        self.alt = alt
        self.t_bounds = t_bounds
        self.method = method
        self.nb_seg = nb_seg
        self.order = order
        self.solver = solver
        self.snopt_opts = snopt_opts
        self.nb_samp = nb_samp

        limits = np.vstack((np.asarray(isp_lim), np.asarray(twr_lim)))

        if samp_method == 'rand':
            samp = Random(xlimits=limits)
        elif samp_method == 'lhs':
            samp = LHS(xlimits=limits, criterion=criterion)
        elif samp_method == 'full':
            samp = FullFactorial(xlimits=limits)
        else:
            raise ValueError('samp_method must be one of rand, lhs, full')

        self.x_samp = samp(nb_samp)
        self.tof = np.zeros((nb_samp, 1))
        self.mf = np.zeros((nb_samp, 1))

    def solve(self, nlp, i):

        nlp.p.run_driver()

        if isinstance(nlp.phase_name, str):
            phase_name = nlp.phase_name
        else:
            phase_name = nlp.phase_name[-1]

        self.mf[i, 0] = nlp.p.get_val(phase_name + '.timeseries.states:m')[-1, -1]
        self.tof[i, 0] = nlp.p.get_val(phase_name + '.time')[-1]

        nlp.cleanup()


class TwoDimAscSurrogate(SurrogateModel):

    def __init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver,
                 nb_samp, snopt_opts=None, samp_method='lhs', criterion='c'):

        SurrogateModel.__init__(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                                snopt_opts=snopt_opts, samp_method=samp_method, criterion=criterion)

    def sampling(self):

        for i in range(self.nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=self.body.g)
            nlp = TwoDimAscVarNLP(self.body, sc, self.alt, (-np.pi/2, np.pi/2), self.t_bounds, self.method, self.nb_seg,
                                  self.order, self.solver, 'powered', snopt_opts=self.snopt_opts, u_bound=True)

            self.solve(nlp, i)

class TwoDimAscVToffSurrogate(SurrogateModel):

    pass


class TwoDimDescVertSurrogate(SurrogateModel):

    pass