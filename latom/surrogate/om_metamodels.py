"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, MetaModelStructuredComp
from latom.utils.pickle_utils import load, save
from latom.utils.spacecraft import Spacecraft, ImpulsiveBurn
from latom.nlp.nlp_2d import TwoDimAscConstNLP, TwoDimAscVarNLP, TwoDimAscVToffNLP, TwoDimDescConstNLP, \
    TwoDimDescVarNLP, TwoDimDescVLandNLP
from latom.guess.guess_2d import HohmannTransfer
from latom.utils.keplerian_orbit import TwoDimOrb
from latom.data.metamodels.data_mm import dirname_metamodels
from latom.plots.response_surfaces import RespSurf
from latom.analyzer.analyzer_2d import TwoDimDescTwoPhasesAnalyzer


class MetaModel:

    def __init__(self, distributed=False, extrapolate=False, method='scipy_cubic', training_data_gradients=True,
                 vec_size=1, rec_file=None):

        self.mm = MetaModelStructuredComp(distributed=distributed, extrapolate=extrapolate, method=method,
                                          training_data_gradients=training_data_gradients, vec_size=vec_size)

        self.p = Problem()
        self.p.model.add_subsystem('mm', self.mm, promotes=['Isp', 'twr'])
        self.twr = self.Isp = self.m_prop = self.failures = self.limits = self.d = None

        if rec_file is not None:
            self.load(rec_file)
            self.setup()

    @staticmethod
    def abs_path(rec_file):

        return '/'.join([dirname_metamodels, rec_file])

    def load(self, rec_file):

        self.d = load(self.abs_path(rec_file))
        self.twr = self.d['twr']
        self.Isp = self.d['Isp']
        self.m_prop = self.d['m_prop']
        self.failures = self.d['failures']

    def save(self, rec_file):

        d = {'Isp': self.Isp, 'twr': self.twr, 'm_prop': self.m_prop, 'failures': self.failures}
        save(d, self.abs_path(rec_file))

    def compute_grid(self, twr_lim, isp_lim, nb_samp):

        self.twr = np.linspace(twr_lim[0], twr_lim[1], nb_samp[0])
        self.Isp = np.linspace(isp_lim[0], isp_lim[1], nb_samp[1])

        self.m_prop = np.zeros(nb_samp)
        self.failures = np.zeros(nb_samp)

    def setup(self):

        self.limits = np.array([[self.Isp[0], self.Isp[-1]], [self.twr[0], self.twr[-1]]])
        self.mm.add_input('twr', training_data=self.twr)
        self.mm.add_input('Isp', training_data=self.Isp)
        self.mm.add_output('m_prop', training_data=self.m_prop)

        self.p.setup()
        self.p.final_setup()

    def sampling(self, body, twr_lim, isp_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp, snopt_opts=None,
                 u_bound=None, rec_file=None, **kwargs):

        self.compute_grid(twr_lim, isp_lim, nb_samp)

        count = 1

        for i in range(nb_samp[0]):  # loop over thrust/weight ratios
            for j in range(nb_samp[1]):  # loop over specific impulses

                print(f"\nMajor Iteration {j}"
                      f"\nSpecific impulse: {self.Isp[j]:.6f} s"
                      f"\nThrust/weight ratio: {self.twr[i]:.6f}\n")

                sc = Spacecraft(self.Isp[j], self.twr[i], g=body.g)

                try:
                    m, f = self.solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=snopt_opts,
                                      u_bound=u_bound, **kwargs)
                except:
                    m = None
                    f = 1.

                self.m_prop[i, j] = m
                self.failures[i, j] = f

                count += 1

        self.setup()

        if rec_file is not None:
            self.save(rec_file)

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):

        return None, None

    def plot(self, nb_lines=50, kind='prop', log_scale=False):

        if kind == 'prop':
            rs = RespSurf(self.Isp, self.twr, self.m_prop.T, 'Propellant fraction', nb_lines=nb_lines,
                          log_scale=log_scale)
        elif kind == 'final':
            rs = RespSurf(self.Isp, self.twr, (1 - self.m_prop.T), 'Final/initial mass ratio', nb_lines=nb_lines,
                          log_scale=log_scale)
        else:
            raise ValueError('kind must be either prop or final')

        rs.plot()
        plt.show()


class TwoDimAscConstMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        nlp = TwoDimAscConstNLP(body, sc, alt, kwargs['theta'], (-np.pi / 2, np.pi / 2), kwargs['tof'], t_bounds,
                                method, nb_seg, order, solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimAscVarMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        nlp = TwoDimAscVarNLP(body, sc, alt, (-np.pi / 2, np.pi / 2), t_bounds, method, nb_seg, order, solver,
                              'powered', snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimAscVToffMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        nlp = TwoDimAscVToffNLP(body, sc, alt, kwargs['alt_safe'], kwargs['slope'], (-np.pi / 2, np.pi / 2), t_bounds,
                                method, nb_seg, order, solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimDescConstMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):

        if 'alt_p' in kwargs:
            alt_p = kwargs['alt_p']
        else:
            alt_p = alt

        dep = TwoDimOrb(body.GM, a=(body.R + alt), e=0.0)
        arr = TwoDimOrb(body.GM, a=(body.R + alt_p), e=0.0)

        ht = HohmannTransfer(body.GM, dep, arr)
        deorbit_burn = ImpulsiveBurn(sc, ht.dva)

        nlp = TwoDimDescConstNLP(body, deorbit_burn.sc, alt_p, ht.transfer.vp, kwargs['theta'], (0.0, 1.5 * np.pi),
                                 kwargs['tof'], t_bounds, method, nb_seg, order, solver, 'powered',
                                 snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimDescVarMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        nlp = TwoDimDescVarNLP(body, sc, alt, (0.0, 3 / 2 * np.pi), t_bounds, method, nb_seg, order, solver, 'powered',
                               snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimDescVLandMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        nlp = TwoDimDescVLandNLP(body, sc, alt, kwargs['alt_safe'], kwargs['slope'], (0.0, 3 / 2 * np.pi), t_bounds,
                                 method, nb_seg, order, solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimDescTwoPhasesMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        tr = TwoDimDescTwoPhasesAnalyzer(body, sc, alt, kwargs['alt_p'], kwargs['alt_switch'], kwargs['theta'],
                                         kwargs['tof'], t_bounds, method, nb_seg, order, solver, snopt_opts=snopt_opts,
                                         fix=kwargs['fix'])

        f = tr.run_driver()
        tr.get_solutions(explicit=False, scaled=False)
        tr.nlp.cleanup()

        m_prop = 1 - tr.states[-1][-1, -1] / tr.sc.m0

        return m_prop, f
