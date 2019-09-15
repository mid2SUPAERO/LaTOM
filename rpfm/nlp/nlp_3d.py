"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.nlp.nlp import SinglePhaseNLP


class ThreeDimNLP(SinglePhaseNLP):

    def __init__(self, body, sc, method, nb_seg, order, solver, ode_class, ode_kwargs, ph_name, snopt_opts=None,
                 rec_file=None):

        SinglePhaseNLP.__init__(self, body, sc, method, nb_seg, order, solver, ode_class, ode_kwargs, ph_name,
                                snopt_opts=snopt_opts, rec_file=rec_file)

