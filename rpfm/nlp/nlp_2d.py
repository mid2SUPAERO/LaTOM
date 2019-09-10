"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp import SinglePhaseNLP
from rpfm.odes.odes_2d import ODE2dVarThrust, ODE2dConstThrust
from rpfm.guess.guess_2d import TwoDimAscGuess
from rpfm.utils.const import g0


class TwoDimNLP(SinglePhaseNLP):

    def __init__(self, body, sc, alt, theta, alpha_bounds, method, nb_seg, order, solver, ode_class,
                 ode_kwargs, ph_name, snopt_opts=None, rec_file=None):

        SinglePhaseNLP.__init__(self, body, sc, method, nb_seg, order, solver, ode_class, ode_kwargs, ph_name,
                                snopt_opts=snopt_opts, rec_file=rec_file)

        self.alt = alt
        self.alpha_bounds = np.asarray(alpha_bounds)
        self.r_circ = self.body.R + self.alt
        self.v_circ = (self.body.GM/self.r_circ)**0.5

        """
        self.phase.add_state('r', units='m', rate_source='rdot', targets='r', fix_initial=True, fix_final=True,
                             lower=body.R, scaler=np.power(10, -np.floor(np.log10(self.r_circ))))
        self.phase.add_state('theta', units='rad', rate_source='thetadot', fix_initial=True, fix_final=False,
                             lower=0.0)
        self.phase.add_state('u', units='m/s', rate_source='udot', targets='u', fix_initial=True, fix_final=True,
                             scaler=np.power(10, -np.floor(np.log10(self.v_circ))))
        self.phase.add_state('v', units='m/s', rate_source='vdot', targets='v', fix_initial=True, fix_final=True,
                             lower=0.0, scaler=np.power(10, -np.floor(np.log10(self.v_circ))))
        self.phase.add_state('m', units='kg', rate_source='mdot', targets='m', fix_initial=True, fix_final=False,
                             lower=sc.m_dry, upper=sc.m0, scaler=np.power(10, -np.floor(np.log10(sc.m0))))

        self.phase.add_control('alpha', units='rad', targets='alpha', fix_initial=False, fix_final=False,
                               continuity=True, rate_continuity=True, rate2_continuity=False, lower=alpha_bounds[0],
                               upper=alpha_bounds[1])

        self.phase.add_design_parameter('w', units='m/s', opt=False, val=sc.w)
        
        """

        self.phase.add_state('r', units='m', rate_source='rdot', targets='r', fix_initial=True, fix_final=True,
                             lower=1.0, ref0=1.0, ref=self.r_circ/self.body.R)
        self.phase.add_state('theta', units='rad', rate_source='thetadot', fix_initial=True, fix_final=False,
                             lower=0.0, ref=theta)
        self.phase.add_state('u', units='m/s', rate_source='udot', targets='u', fix_initial=True, fix_final=True,
                             lower=0.0, ref=self.v_circ/self.body.vc)
        self.phase.add_state('v', units='m/s', rate_source='vdot', targets='v', fix_initial=True, fix_final=True,
                             lower=0.0, ref=self.v_circ/self.body.vc)
        self.phase.add_state('m', units='kg', rate_source='mdot', targets='m', fix_initial=True, fix_final=False,
                             lower=sc.m_dry, upper=sc.m0, ref0=self.sc.m_dry, ref=self.sc.m0)

        self.phase.add_control('alpha', units='rad', targets='alpha', fix_initial=False, fix_final=False,
                               continuity=True, rate_continuity=True, rate2_continuity=False, lower=alpha_bounds[0],
                               upper=alpha_bounds[1], ref=alpha_bounds[1])

        self.phase.add_design_parameter('w', units='m/s', opt=False, val=sc.w/body.vc)


class TwoDimAscConstNLP(TwoDimNLP):

    def __init__(self, body, sc, alt, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                 ph_name, snopt_opts=None, check_partials=False, rec_file=None):

        # ode_kwargs = {'GM': body.GM, 'T': sc.T_max, 'w': sc.w}
        ode_kwargs = {'GM': 1.0, 'T': sc.twr, 'w': sc.w/body.vc}

        TwoDimNLP.__init__(self, body, sc, alt, theta, alpha_bounds, method, nb_seg, order, solver, ODE2dConstThrust,
                           ode_kwargs, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        # self.phase.add_design_parameter('thrust', units='N', opt=False, val=sc.T_max)
        self.phase.add_design_parameter('thrust', units='N', opt=False, val=sc.twr)

        self.set_time_options(tof, t_bounds)
        self.set_objective()
        self.setup()

        # initial guess from linear interpolation
        self.set_time_guess(self.tof)

        """
        self.p[self.phase_name + '.states:r'] = self.phase.interpolate(ys=(body.R, self.r_circ), nodes='state_input')
        self.p[self.phase_name + '.states:theta'] = self.phase.interpolate(ys=(0.0, theta), nodes='state_input')
        self.p[self.phase_name + '.states:u'] = self.phase.interpolate(ys=(0.0, 0.0), nodes='state_input')
        self.p[self.phase_name + '.states:v'] = self.phase.interpolate(ys=(0.0, self.v_circ), nodes='state_input')
        self.p[self.phase_name + '.states:m'] = self.phase.interpolate(ys=(self.sc.m0, self.sc.m_dry),
                                                                       nodes='state_input')

        self.p[self.phase_name + '.controls:alpha'] = self.phase.interpolate(ys=(0.0, 0.0), nodes='control_input')
        """

        self.p[self.phase_name + '.states:r'] = self.phase.interpolate(ys=(1.0, self.r_circ/self.body.R),
                                                                       nodes='state_input')
        self.p[self.phase_name + '.states:theta'] = self.phase.interpolate(ys=(0.0, theta), nodes='state_input')
        self.p[self.phase_name + '.states:u'] = self.phase.interpolate(ys=(0.0, 0.0), nodes='state_input')
        self.p[self.phase_name + '.states:v'] = self.phase.interpolate(ys=(0.0, self.v_circ/self.body.vc),
                                                                       nodes='state_input')
        self.p[self.phase_name + '.states:m'] = self.phase.interpolate(ys=(self.sc.m0, self.sc.m_dry),
                                                                       nodes='state_input')

        self.p[self.phase_name + '.controls:alpha'] = self.phase.interpolate(ys=(0.0, 0.0), nodes='control_input')

        self.p.run_model()

        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)


class TwoDimAscVarNLP(TwoDimNLP):

    def __init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, check_partials=False, rec_file=None):

        # ode_kwargs = {'GM': body.GM, 'w': sc.w}
        ode_kwargs = {'GM': 1.0, 'w': sc.w/body.vc}

        TwoDimNLP.__init__(self, body, sc, alt, np.pi/2, alpha_bounds, method, nb_seg, order, solver, ODE2dVarThrust,
                           ode_kwargs, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        """
        self.phase.add_control('thrust', units='N', targets='thrust', fix_initial=False, fix_final=False,
                               continuity=False, rate_continuity=False, rate2_continuity=False, lower=self.sc.T_min,
                               upper=self.sc.T_max, scaler=np.power(10, -np.floor(np.log10(self.sc.T_max))))
        """

        self.phase.add_control('thrust', units='N', targets='thrust', fix_initial=False, fix_final=False,
                               continuity=False, rate_continuity=False, rate2_continuity=False, lower=0.0,
                               upper=self.sc.twr, ref=self.sc.twr)

        self.guess = TwoDimAscGuess(self.body.GM, self.body.R, alt, sc)

        self.set_time_options(self.guess.tof, t_bounds)
        self.set_objective()
        self.setup()

        # initial guess from TwoDimAscGuess
        self.set_time_guess(self.tof)

        """
        self.guess.compute_trajectory(t=self.t_control)

        self.p[self.phase_name + '.states:r'] = np.take(self.guess.r, self.idx_state_control)
        self.p[self.phase_name + '.states:theta'] = np.take(self.guess.theta, self.idx_state_control)
        self.p[self.phase_name + '.states:u'] = np.take(self.guess.u, self.idx_state_control)
        self.p[self.phase_name + '.states:v'] = np.take(self.guess.v, self.idx_state_control)
        self.p[self.phase_name + '.states:m'] = np.take(self.guess.m, self.idx_state_control)

        self.p[self.phase_name + '.controls:thrust'] = self.guess.T
        self.p[self.phase_name + '.controls:alpha'] = self.guess.alpha
        """

        self.guess.compute_trajectory(t=self.t_control*body.tc)

        self.p[self.phase_name + '.states:r'] = np.take(self.guess.r/body.R, self.idx_state_control)
        self.p[self.phase_name + '.states:theta'] = np.take(self.guess.theta, self.idx_state_control)
        self.p[self.phase_name + '.states:u'] = np.take(self.guess.u/body.vc, self.idx_state_control)
        self.p[self.phase_name + '.states:v'] = np.take(self.guess.v/body.vc, self.idx_state_control)
        self.p[self.phase_name + '.states:m'] = np.take(self.guess.m, self.idx_state_control)

        self.p[self.phase_name + '.controls:thrust'] = self.guess.T/sc.m0/body.g
        self.p[self.phase_name + '.controls:alpha'] = self.guess.alpha

        self.p.run_model()
        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)


class TwoDimAscVToffNLP(TwoDimAscVarNLP):

    pass
