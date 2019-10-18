"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp_2d import TwoDimVarNLP
from rpfm.guess.guess_nrho_2d import TwoDimAscGuessNRHO


class TwoDimAscVarNRHO(TwoDimVarNLP):

    def __init__(self, body, sc, alt, rp, t, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False, u_bound='lower', fix_final=False):

        guess = TwoDimAscGuessNRHO(body.GM, body.R, alt, rp, t, sc)

        TwoDimVarNLP.__init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                              guess, snopt_opts=snopt_opts, rec_file=rec_file, check_partials=check_partials,
                              u_bound=u_bound, fix_final=fix_final)

    def set_states_alpha_options(self, theta, u_bound=None):

        self.phase.set_state_options('r', fix_initial=True, fix_final=True, lower=self.guess.r_llo/self.body.R,
                                     ref0=self.guess.r_llo/self.body.R, ref=self.guess.ep.ra_nrho/self.body.R)

        self.phase.set_state_options('theta', fix_initial=False, fix_final=True, lower=-np.pi/2,
                                     ref=theta)
        self.phase.set_state_options('u', fix_initial=True, fix_final=True, ref=self.v_circ/self.body.vc)
        self.phase.set_state_options('v', fix_initial=True, fix_final=True, lower=0.0, ref=self.v_circ/self.body.vc)
        self.phase.set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                     ref0=self.sc.m_dry, ref=self.sc.m0)
        self.phase.add_control('alpha', fix_initial=False, fix_final=False, continuity=True, rate_continuity=True,
                               rate2_continuity=False, lower=self.alpha_bounds[0], upper=self.alpha_bounds[1],
                               ref=self.alpha_bounds[1])
        self.phase.add_design_parameter('w', opt=False, val=self.sc.w/self.body.vc)
