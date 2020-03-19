"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.nlp.nlp_heo_2d import TwoDim3PhasesLLO2HEONLP
from latom.analyzer.analyzer_heo_2d import TwoDimLLO2ApoAnalyzer, TwoDimLLO2ApoContinuationAnalyzer
from latom.surrogate.meta_models import MetaModel
from latom.utils.pickle_utils import save
from latom.utils.spacecraft import Spacecraft
from latom.plots.response_surfaces import RespSurf


class TwoDimLLO2ApoMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):

        tr = TwoDimLLO2ApoAnalyzer(body, sc, alt, kwargs['rp'], kwargs['t'], t_bounds, method, nb_seg, order, solver,
                                   snopt_opts=snopt_opts)

        f = tr.run_driver()
        tr.get_solutions(explicit=False, scaled=False)
        tr.nlp.cleanup()

        m_prop = 1. - tr.insertion_burn.mf/tr.sc.m0

        return m_prop, f


class TwoDimLLO2ApoContinuationMetaModel(MetaModel):

    def __init__(self, distributed=False, extrapolate=False, method='scipy_cubic', training_data_gradients=True,
                 vec_size=1, rec_file=None):

        self.energy = None

        MetaModel.__init__(self, distributed=distributed, extrapolate=extrapolate, method=method,
                           training_data_gradients=training_data_gradients, vec_size=vec_size, rec_file=rec_file)

    def load(self, rec_file):

        MetaModel.load(self, rec_file)
        self.energy = self.d['energy']

    def save(self, rec_file):

        d = {'Isp': self.Isp, 'twr': self.twr, 'm_prop': self.m_prop, 'failures': self.failures, 'energy': self.energy}
        save(d, self.abs_path(rec_file))

    def compute_grid(self, twr_lim, isp_lim, nb_samp):

        MetaModel.compute_grid(self, twr_lim, isp_lim, nb_samp)
        self.energy = np.zeros(nb_samp)

    def sampling(self, body, twr_lim, isp_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp, snopt_opts=None,
                 u_bound=None, rec_file=None, **kwargs):

        self.compute_grid(twr_lim, isp_lim, nb_samp)
        twr_flip = np.flip(self.twr)

        for j in range(nb_samp[1]):  # loop over specific impulses

            print(f"\nMajor Iteration {j}\nSpecific impulse: {self.Isp[j]:.6f} s\n")

            if kwargs['log_scale']:
                sc = Spacecraft(self.Isp[j], np.exp(twr_flip[0]), g=body.g)
            else:
                sc = Spacecraft(self.Isp[j], twr_flip[0], g=body.g)
            tr = TwoDimLLO2ApoContinuationAnalyzer(body, sc, alt, kwargs['rp'], kwargs['t'], t_bounds, twr_flip,
                                                   method, nb_seg, order, solver, snopt_opts=snopt_opts,
                                                   log_scale=kwargs['log_scale'])
            tr.run_continuation()

            self.m_prop[:, j] = np.flip(tr.m_prop_list)
            self.energy[:, j] = np.flip(tr.energy_list)

        self.setup()
        if rec_file is not None:
            self.save(rec_file)

    def plot(self, nb_lines=50, log_scale=False):

        en = RespSurf(self.Isp, self.twr, self.energy, 'Specific energy [m^2/s^2]', nb_lines=nb_lines,
                      log_scale=log_scale)
        en.plot()
        MetaModel.plot(self, nb_lines=nb_lines, log_scale=log_scale)


class TwoDim3PhasesLLO2HEOMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):

        nlp = TwoDim3PhasesLLO2HEONLP(body, sc, alt, kwargs['rp'], kwargs['t'], (-np.pi/2, np.pi/2), t_bounds, method,
                                      nb_seg, order, solver, kwargs['phase_name'], snopt_opts=snopt_opts)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name[-1] + '.timeseries.states:m')[-1, -1]

        return m_prop, f
