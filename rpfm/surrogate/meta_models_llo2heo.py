"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp_heo_2d import TwoDim3PhasesLLO2HEONLP, TwoDimLLO2ApoNLP
from rpfm.surrogate.meta_models import MetaModel


class TwoDimLLO2ApoMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):

        nlp = TwoDimLLO2ApoNLP(body, sc, alt, kwargs['rp'], kwargs['t'], (-np.pi / 2, np.pi / 2), t_bounds, method,
                               nb_seg, order, solver, 'powered', snopt_opts=snopt_opts)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

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


if __name__ == '__main__':

    from rpfm.utils.primary import Moon
    moon = Moon()
    so = {'Major feasibility tolerance': 1e-12, 'Major optimality tolerance': 1e-12,
          'Minor feasibility tolerance': 1e-12}

    # additional settings
    run_driver = False  # solve the NLP
    exp_sim = False  # perform explicit

    # a = TwoDim3PhasesLLO2HEOMetaModel()
    # a.sampling(moon, [1.1, 4.0], [250., 500.], 100e3, ((0.2, 1.8), (0.2, 1.8), (0.2, 1.8)), 'gauss-lobatto',
    # (60, 400, 60), 3, 'SNOPT', nb_samp=(5, 10), snopt_opts=so, rec_file='meta_model_test.pkl',
    # rp=3150e3, t=6.5655*86400, phase_name=('dep', 'coast', 'arr'))

    a = TwoDimLLO2ApoMetaModel()
    a.sampling(moon, [1.1, 4.0], [250., 500.], 100e3, None, 'gauss-lobatto',
               50, 3, 'SNOPT', nb_samp=(5, 5), snopt_opts=so, rec_file='meta_model_test.pkl',
               rp=3150e3, t=6.5655*86400)

