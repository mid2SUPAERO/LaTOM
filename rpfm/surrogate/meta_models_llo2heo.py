"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp_heo_2d import TwoDim3PhasesLLO2HEONLP
from rpfm.analyzer.analyzer_heo_2d import TwoDimLLO2ApoAnalyzer
from rpfm.surrogate.meta_models import MetaModel


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


class TwoDim3PhasesLLO2HEOMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):

        nlp = TwoDim3PhasesLLO2HEONLP(body, sc, alt, kwargs['rp'], kwargs['t'], (-np.pi/2, np.pi/2), t_bounds, method,
                                      nb_seg, order, solver, kwargs['phase_name'], snopt_opts=snopt_opts)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name[-1] + '.timeseries.states:m')[-1, -1]

        return m_prop, f
