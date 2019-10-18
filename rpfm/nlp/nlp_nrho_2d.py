"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.nlp.nlp_2d import TwoDimVarNLP
from rpfm.guess.guess_nrho_2d import TwoDimAscGuessNRHO


class TwoDimVarNRHO(TwoDimVarNLP):

    def __init__(self, body, sc, alt, rp, t, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False, u_bound='lower', fix_final=False):

        guess = TwoDimAscGuessNRHO(body.GM, body.R, alt, rp, t, sc)

        TwoDimVarNLP.__init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                              guess, snopt_opts=snopt_opts, rec_file=rec_file, check_partials=check_partials,
                              u_bound=u_bound, fix_final=fix_final)
