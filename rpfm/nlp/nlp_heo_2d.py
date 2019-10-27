"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp_2d import TwoDimVarNLP, TwoDimNLP
from rpfm.guess.guess_heo_2d import TwoDimLLO2HEOGuess
from rpfm.odes.odes_2d import ODE2dLLO2Apo


class TwoDimLLO2HEONLP(TwoDimVarNLP):

    def __init__(self, body, sc, alt, rp, t, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False, u_bound='lower', fix_final=True):

        guess = TwoDimLLO2HEOGuess(body.GM, body.R, alt, rp, t, sc)

        TwoDimVarNLP.__init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                              guess, snopt_opts=snopt_opts, rec_file=rec_file, check_partials=check_partials,
                              u_bound=u_bound, fix_final=fix_final)

    def set_states_alpha_options(self, theta, u_bound=None):

        self.phase.set_state_options('r', fix_initial=True, fix_final=True, lower=1.0,
                                     ref0=self.guess.ht.depOrb.a/self.body.R,
                                     ref=self.guess.ht.arrOrb.ra/self.body.R)

        self.phase.set_state_options('theta', fix_initial=False, fix_final=True, lower=-np.pi/2, ref=theta)
        self.phase.set_state_options('u', fix_initial=True, fix_final=True, ref=self.v_circ/self.body.vc)
        self.phase.set_state_options('v', fix_initial=True, fix_final=True, lower=0.0, ref=self.v_circ/self.body.vc)
        self.phase.set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                     ref0=self.sc.m_dry, ref=self.sc.m0)
        self.phase.add_control('alpha', fix_initial=False, fix_final=False, continuity=True, rate_continuity=True,
                               rate2_continuity=False, lower=self.alpha_bounds[0], upper=self.alpha_bounds[1],
                               ref=self.alpha_bounds[1])
        self.phase.add_design_parameter('w', opt=False, val=self.sc.w/self.body.vc)


class TwoDimLLO2ApoNLP(TwoDimNLP):

    def __init__(self, body, sc, alt, rp, t, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False):

        guess = TwoDimLLO2HEOGuess(body.GM, body.R, alt, rp, t, sc)
        ode_kwargs = {'GM': 1.0, 'T': sc.twr, 'w': sc.w / body.vc, 'ra': guess.ht.arrOrb.ra/body.R}

        TwoDimNLP.__init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ODE2dLLO2Apo,
                           ode_kwargs, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.guess = guess

        # states options
        self.phase.set_state_options('r', fix_initial=True, fix_final=False, lower=1.0, ref0=1.0,
                                     ref=self.r_circ / self.body.R)
        self.phase.set_state_options('theta', fix_initial=True, fix_final=False, lower=0.0, ref=self.guess.pow.thetaf)
        self.phase.set_state_options('u', fix_initial=True, fix_final=False, ref0=0.0, ref=self.v_circ / self.body.vc)
        self.phase.set_state_options('v', fix_initial=True, fix_final=False, lower=0.0, ref=self.v_circ / self.body.vc)
        self.phase.set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                     ref0=self.sc.m_dry, ref=self.sc.m0)

        # control options
        self.phase.add_control('alpha', fix_initial=False, fix_final=False, continuity=True, rate_continuity=True,
                               rate2_continuity=False, lower=self.alpha_bounds[0], upper=self.alpha_bounds[1],
                               ref=self.alpha_bounds[1])

        # design parameters
        self.phase.add_design_parameter('w', opt=False, val=self.sc.w / self.body.vc)
        self.phase.add_design_parameter('thrust', opt=False, val=self.sc.twr)

        # time options
        self.set_time_options(self.guess.pow.tf, t_bounds)

        # constraint on injection
        self.phase.add_boundary_constraint('c', loc='final', equals=0.0, shape=(1,))

        # objective
        self.set_objective()

        # NLP setup
        self.setup()

        # initial guess
        TwoDimNLP.set_initial_guess(self, check_partials=check_partials, throttle=False)
