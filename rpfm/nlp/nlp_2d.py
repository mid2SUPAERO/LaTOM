"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp import SinglePhaseNLP
from rpfm.odes.odes_2d import ODE2dVarThrust, ODE2dConstThrust
from rpfm.guess.guess_2d import TwoDimAscGuess
from rpfm.utils.const import states_2d


class TwoDimNLP(SinglePhaseNLP):

    def __init__(self, sc, alt, theta, delta_theta, alpha_bounds, method, nb_seg, order, solver, ode_class, ph_name,
                 snopt_opts=None, rec_file=None):

        SinglePhaseNLP.__init__(self, sc, method, nb_seg, order, solver, ode_class, ph_name,
                                snopt_opts=snopt_opts, rec_file=rec_file)

        self.alt_adim = alt/self.moon.R
        self.alpha_bounds = alpha_bounds

        r_ref0 = (2 + self.alt_adim)*0.5
        r_ref = 1 + self.alt_adim
        v_ref = 1/r_ref**0.5

        self.phase.add_state('r', rate_source='rdot', targets='r', fix_initial=True, fix_final=False, lower=1.0,
                             ref0=r_ref0, ref=r_ref)
        self.phase.add_state('theta', rate_source='thetadot', fix_initial=True, fix_final=False, lower=0.0,
                             ref0=theta, ref=delta_theta*theta)
        self.phase.add_state('u', rate_source='udot', targets='u', fix_initial=True, fix_final=True)
        self.phase.add_state('v', rate_source='vdot', targets='v', fix_initial=True, fix_final=True, lower=0.0,
                             ref0=v_ref/2, ref=v_ref)
        self.phase.add_state('m', rate_source='mdot', targets='m', fix_initial=True, fix_final=False, lower=sc.m_dry,
                             upper=sc.m0, ref0=(sc.m0 + sc.m_dry)*0.5, ref=sc.m0)

        self.phase.add_control('alpha', fix_initial=False, fix_final=False, targets='alpha', lower=alpha_bounds[0],
                               upper=alpha_bounds[1], ref0=np.mean(alpha_bounds), ref=alpha_bounds[1], continuity=True,
                               rate_continuity=False, rate2_continuity=False)


class TwoDimAscConstNLP(TwoDimNLP):

    def __init__(self, sc, alt, theta, delta_theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                 ode_class, ph_name, snopt_opts=None, rec_file=None):

        TwoDimNLP.__init__(self, sc, alt, theta, delta_theta, alpha_bounds, method, nb_seg, order, solver,
                           ODE2dConstThrust, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.tof_adim = tof/self.moon.tc
        self.t_bounds_adim = t_bounds*self.tof_adim
        self.set_time_options(self.t_bounds_adim)

        # initial guess from linear interpolation


class TwoDimAscVarNLP(TwoDimNLP):

    def __init__(self, sc, alt, delta_theta, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None):

        TwoDimNLP.__init__(self, sc, alt, np.pi, delta_theta, alpha_bounds, method, nb_seg, order, solver,
                           ODE2dVarThrust, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.phase.add_control('thrust', fix_initial=False, fix_final=False, targets='thrust', lower=self.sc.T_min,
                               upper=self.sc.T_max, ref0=(self.sc.T_min + self.sc.T_max)*0.5, ref=self.sc.T_max,
                               continuity=False, rate_continuity=False, rate2_continuity=False)

        self.guess = TwoDimAscGuess(self.moon.GM, self.moon.R, alt, sc)

        self.tof_adim = self.guess.tof/self.moon.tc
        self.t_bounds_adim = t_bounds*self.tof_adim
        self.set_time_options(self.t_bounds_adim)
        self.set_time_guess(self.tof_adim)

        self.guess.compute_trajectory(t=self.t_control*self.moon.tc)

        for i in range(5):
            self.p[ph_name + '.states:' + states_2d[i]] = np.take(self.guess.states[:, i], self.idx_state_control)

        self.p[ph_name + '.controls:thrust'] = self.guess.T
        self.p[ph_name + '.controls:alpha'] = self.guess.alpha


class TwoDimAscVToffNLP(TwoDimAscVarNLP):

    pass
