"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp import SinglePhaseNLP, MultiPhaseNLP
from rpfm.odes.odes_2d import ODE2dVarThrust, ODE2dConstThrust, ODE2dVToff, ODE2dVertical
from rpfm.guess.guess_2d import TwoDimAscGuess, TwoDimDescGuess


class TwoDimNLP(SinglePhaseNLP):

    def __init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ode_class,
                 ode_kwargs, ph_name, snopt_opts=None, rec_file=None):

        SinglePhaseNLP.__init__(self, body, sc, method, nb_seg, order, solver, ode_class, ode_kwargs, ph_name,
                                snopt_opts=snopt_opts, rec_file=rec_file)

        self.alt = alt
        self.alpha_bounds = np.asarray(alpha_bounds)
        self.r_circ = body.R + self.alt
        self.v_circ = (body.GM/self.r_circ)**0.5

    def set_states_alpha_options(self, theta, u_bound=False):

        self.phase.set_state_options('r', fix_initial=True, fix_final=True, lower=1.0, ref0=1.0,
                                     ref=self.r_circ/self.body.R)

        if theta > 0.0:
            self.phase.set_state_options('theta', fix_initial=True, fix_final=False, lower=0.0, ref=theta)
        else:
            self.phase.set_state_options('theta', fix_initial=False, fix_final=True, upper=0.0, adder=-theta,
                                         scaler=-1.0/theta)

        if u_bound:
            self.phase.set_state_options('u', fix_initial=True, fix_final=True, lower=0.0, ref=self.v_circ/self.body.vc)
        else:
            self.phase.set_state_options('u', fix_initial=True, fix_final=True, ref=self.v_circ/self.body.vc)

        self.phase.set_state_options('v', fix_initial=True, fix_final=True, lower=0.0, ref=self.v_circ/self.body.vc)
        self.phase.set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                     ref0=self.sc.m_dry, ref=self.sc.m0)

        self.phase.add_control('alpha', fix_initial=False, fix_final=False, continuity=True, rate_continuity=True,
                               rate2_continuity=False, lower=self.alpha_bounds[0], upper=self.alpha_bounds[1],
                               ref=self.alpha_bounds[1])

        self.phase.add_design_parameter('w', opt=False, val=self.sc.w/self.body.vc)


class TwoDimConstNLP(TwoDimNLP):

    def __init__(self, body, sc, alt, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, u_bound=False):

        ode_kwargs = {'GM': 1.0, 'T': sc.twr, 'w': sc.w / body.vc}

        TwoDimNLP.__init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ODE2dConstThrust,
                           ode_kwargs, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.set_options(theta, tof, t_bounds, u_bound=u_bound)
        self.setup()

    def set_options(self, theta, tof, t_bounds, u_bound=False):

        self.set_states_alpha_options(theta, u_bound=u_bound)
        self.phase.add_design_parameter('thrust', opt=False, val=self.sc.twr)
        self.set_time_options(tof, t_bounds)
        self.set_objective()

    def set_initial_guess(self, bcs, check_partials=False):

        self.set_time_guess(self.tof)

        self.p[self.phase_name + '.states:r'] = self.phase.interpolate(ys=bcs[0], nodes='state_input')
        self.p[self.phase_name + '.states:theta'] = self.phase.interpolate(ys=bcs[1], nodes='state_input')
        self.p[self.phase_name + '.states:u'] = self.phase.interpolate(ys=bcs[2], nodes='state_input')
        self.p[self.phase_name + '.states:v'] = self.phase.interpolate(ys=bcs[3], nodes='state_input')
        self.p[self.phase_name + '.states:m'] = self.phase.interpolate(ys=bcs[4], nodes='state_input')

        self.p[self.phase_name + '.controls:alpha'] = self.phase.interpolate(ys=bcs[5], nodes='control_input')

        self.p.run_model()

        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)


class TwoDimAscConstNLP(TwoDimConstNLP):

    def __init__(self, body, sc, alt, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                 ph_name, snopt_opts=None, rec_file=None, check_partials=False, u_bound=False):

        TwoDimConstNLP.__init__(self, body, sc, alt, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                                ph_name, snopt_opts=snopt_opts, rec_file=rec_file, u_bound=u_bound)

        bcs = np.array([[1.0, self.r_circ/self.body.R], [0.0, theta], [0.0, 0.0], [0.0, self.v_circ/self.body.vc],
                        [self.sc.m0, self.sc.m_dry], [0.0, 0.0]])

        self.set_initial_guess(bcs, check_partials=check_partials)


class TwoDimDescConstNLP(TwoDimConstNLP):

    def __init__(self, body, sc, alt, vp, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                 ph_name, snopt_opts=None, rec_file=None, check_partials=False):

        TwoDimConstNLP.__init__(self, body, sc, alt, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                                ph_name, snopt_opts=snopt_opts, rec_file=rec_file, u_bound=False)

        self.vp = vp

        bcs = np.array([[self.r_circ/self.body.R, 1.0], [0.0, theta], [0.0, 0.0], [self.vp/self.body.vc, 0.0],
                        [self.sc.m0, self.sc.m_dry], [1.5*np.pi, np.pi/2]])

        self.set_initial_guess(bcs, check_partials=check_partials)


class TwoDimVarNLP(TwoDimNLP):

    def __init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name, guess,
                 snopt_opts=None, rec_file=None, check_partials=False, u_bound=False, fix_final=False):

        ode_kwargs = {'GM': 1.0, 'w': sc.w / body.vc}

        TwoDimNLP.__init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ODE2dVarThrust, ode_kwargs,
                           ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.guess = guess

        self.set_options(np.pi, t_bounds, u_bound=u_bound)  # was pi/2
        self.setup()
        self.set_initial_guess(check_partials=check_partials, fix_final=fix_final)

    def set_options(self, theta, t_bounds, u_bound=False):

        self.set_states_alpha_options(theta, u_bound=u_bound)

        twr_min = self.sc.T_min/self.sc.m0/self.body.g

        self.phase.add_control('thrust', fix_initial=False, fix_final=False, continuity=False, rate_continuity=False,
                               rate2_continuity=False, lower=twr_min, upper=self.sc.twr, ref0=twr_min, ref=self.sc.twr)

        self.set_time_options(self.guess.pow2.tf, t_bounds)
        self.set_objective()

    def set_initial_guess(self, check_partials=False, fix_final=False):

        self.set_time_guess(self.tof)

        self.guess.compute_trajectory(t=self.t_control*self.body.tc, fix_final=fix_final)

        self.p[self.phase_name + '.states:r'] = np.take(self.guess.states[:, 0]/self.body.R, self.idx_state_control)
        self.p[self.phase_name + '.states:theta'] = np.take(self.guess.states[:, 1], self.idx_state_control)
        self.p[self.phase_name + '.states:u'] = np.take(self.guess.states[:, 2]/self.body.vc, self.idx_state_control)
        self.p[self.phase_name + '.states:v'] = np.take(self.guess.states[:, 3]/self.body.vc, self.idx_state_control)
        self.p[self.phase_name + '.states:m'] = np.take(self.guess.states[:, 4], self.idx_state_control)

        self.p[self.phase_name + '.controls:thrust'] =\
            np.reshape(self.guess.controls[:, 0]/self.sc.m0/self.body.g, (len(self.t_control), 1))
        self.p[self.phase_name + '.controls:alpha'] = np.reshape(self.guess.controls[:, 1], (len(self.t_control), 1))

        self.p.run_model()

        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)


class TwoDimAscVarNLP(TwoDimVarNLP):

    def __init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name, snopt_opts=None,
                 rec_file=None, check_partials=False, u_bound=False, fix_final=False):

        TwoDimVarNLP.__init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                              TwoDimAscGuess(body.GM, body.R, alt, sc), snopt_opts=snopt_opts, rec_file=rec_file,
                              check_partials=check_partials, u_bound=u_bound, fix_final=fix_final)


class TwoDimDescVarNLP(TwoDimVarNLP):

    def __init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name, snopt_opts=None,
                 rec_file=None, check_partials=False, fix_final=False):

        TwoDimVarNLP.__init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                              TwoDimDescGuess(body.GM, body.R, alt, sc), snopt_opts=snopt_opts, rec_file=rec_file,
                              check_partials=check_partials, u_bound=False, fix_final=fix_final)


class TwoDimVToffNLP(TwoDimVarNLP):

    def __init__(self, body, sc, alt, alt_safe, slope, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 guess, snopt_opts=None, rec_file=None, check_partials=False, u_bound=False, fix_final=False):

        ode_kwargs = {'GM': 1.0, 'w': sc.w/body.vc, 'R': 1.0, 'alt_safe': alt_safe/body.R, 'slope': slope}

        TwoDimNLP.__init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ODE2dVToff, ode_kwargs,
                           ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.alt_safe = alt_safe
        self.slope = slope

        self.guess = guess

        self.set_options(np.sign(slope)*np.pi, t_bounds, u_bound=u_bound)
        self.setup()
        self.set_initial_guess(check_partials=check_partials, fix_final=fix_final)

    def set_options(self, theta, t_bounds, u_bound=False):

        self.phase.add_path_constraint('dist_safe', lower=0.0, ref=self.alt_safe/self.body.R)
        self.phase.add_timeseries_output('r_safe')

        TwoDimVarNLP.set_options(self, theta, t_bounds, u_bound=u_bound)


class TwoDimAscVToffNLP(TwoDimVToffNLP):

    def __init__(self, body, sc, alt, alt_safe, slope, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False, u_bound=False, fix_final=False):

        TwoDimVToffNLP.__init__(self, body, sc, alt, alt_safe, slope, alpha_bounds, t_bounds, method, nb_seg, order,
                                solver, ph_name, TwoDimAscGuess(body.GM, body.R, alt, sc), snopt_opts=snopt_opts,
                                rec_file=rec_file, check_partials=check_partials, u_bound=u_bound, fix_final=fix_final)


class TwoDimDescVToffNLP(TwoDimVToffNLP):

    def __init__(self, body, sc, alt, alt_safe, slope, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False, fix_final=True):

        TwoDimVToffNLP.__init__(self, body, sc, alt, alt_safe, slope, alpha_bounds, t_bounds, method, nb_seg, order,
                                solver, ph_name, TwoDimDescGuess(body.GM, body.R, alt, sc), snopt_opts=snopt_opts,
                                rec_file=rec_file, check_partials=check_partials, u_bound=False, fix_final=fix_final)


class TwoDimDescTwoPhasesNLP(MultiPhaseNLP):

    def __init__(self, body, sc, alt, alt_switch, vp, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                 ph_name, snopt_opts=None, rec_file=None, check_partials=False, fix='alt'):

        ode_kwargs = {'GM': 1.0, 'T': sc.twr, 'w': sc.w/body.vc}

        MultiPhaseNLP.__init__(self, body, sc, method, nb_seg, order, solver, (ODE2dConstThrust, ODE2dVertical),
                               (ode_kwargs, ode_kwargs), ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.alt = alt
        self.alt_switch = alt_switch

        self.rp = alt + body.R
        self.r_switch = alt_switch + body.R

        self.vp = vp

        self.alpha_bounds = np.asarray(alpha_bounds)
        self.tof = np.asarray(tof)

        if fix in ['alt', 'time']:
            self.fix = fix
        else:
            raise ValueError('fix must be either alt or time')

        self.set_options(theta, t_bounds)
        self.trajectory.link_phases(ph_name, vars=['time', 'r', 'u', 'm'])
        self.setup()
        self.set_initial_guess(theta, check_partials=check_partials)

    def set_options(self, theta, t_bounds):

        # time
        if t_bounds is not None:
            t_bounds = np.asarray(t_bounds)*np.reshape(self.tof, (2, 1))
            self.phase[0].set_time_options(fix_initial=True, duration_ref=self.tof[0]/self.body.tc,
                                           duration_bounds=t_bounds[0]/self.body.tc)
            if self.fix == 'time':
                self.phase[1].set_time_options(fix_initial=False, fix_duration=True,
                                               initial_ref=t_bounds[0, 1]/self.body.tc,
                                               initial_bounds=t_bounds[0]/self.body.tc,
                                               duration_ref=self.tof[1]/self.body.tc,
                                               duration_bounds=t_bounds[1]/self.body.tc)
            else:
                self.phase[1].set_time_options(fix_initial=False, fix_duration=False,
                                               initial_ref=t_bounds[0, 1]/self.body.tc,
                                               initial_bounds=t_bounds[0]/self.body.tc,
                                               duration_ref=self.tof[1]/self.body.tc,
                                               duration_bounds=t_bounds[1]/self.body.tc)
        else:
            self.phase[0].set_time_options(fix_initial=True, duration_ref=self.tof[0]/self.body.tc)
            if self.fix == 'time':
                self.phase[1].set_time_options(fix_initial=False, fix_duration=True,
                                               initial_ref=self.tof[0]/self.body.tc,
                                               duration_ref=self.tof[1]/self.body.tc)
            else:
                self.phase[1].set_time_options(fix_initial=False, fix_duration=False,
                                               initial_ref=self.tof[0]/self.body.tc,
                                               duration_ref=self.tof[1]/self.body.tc)

        # first phase
        if self.fix == 'alt':
            self.phase[0].set_state_options('r', fix_initial=True, fix_final=True, lower=1.0, ref0=1.0,
                                            ref=self.rp/self.body.R)
        else:
            self.phase[0].set_state_options('r', fix_initial=True, fix_final=False, lower=1.0, ref0=1.0,
                                            ref=self.rp/self.body.R)

        self.phase[0].set_state_options('theta', fix_initial=True, fix_final=False, lower=0.0, ref=theta)
        self.phase[0].set_state_options('u', fix_initial=True, fix_final=False, ref=self.vp/self.body.vc)
        self.phase[0].set_state_options('v', fix_initial=True, fix_final=True, lower=0.0, ref=self.vp/self.body.vc)
        self.phase[0].set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                        ref0=self.sc.m_dry, ref=self.sc.m0)
        self.phase[0].add_control('alpha', fix_initial=False, fix_final=False, continuity=True, rate_continuity=True,
                                  rate2_continuity=False, lower=self.alpha_bounds[0], upper=self.alpha_bounds[1],
                                  ref=self.alpha_bounds[1])
        self.phase[0].add_design_parameter('w', opt=False, val=self.sc.w/self.body.vc)
        self.phase[0].add_design_parameter('thrust', opt=False, val=self.sc.twr)

        # second phase
        if self.fix == 'alt':
            self.phase[1].set_state_options('r', fix_initial=True, fix_final=True, lower=1.0, ref0=1.0,
                                            ref=self.rp/self.body.R)
        else:
            self.phase[1].set_state_options('r', fix_initial=False, fix_final=True, lower=1.0, ref0=1.0,
                                            ref=self.rp/self.body.R)

        self.phase[1].set_state_options('u', fix_initial=False, fix_final=True, ref=self.vp/self.body.vc)
        self.phase[1].set_state_options('m', fix_initial=False, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                        ref0=self.sc.m_dry, ref=self.sc.m0)
        self.phase[1].add_design_parameter('w', opt=False, val=self.sc.w/self.body.vc)
        self.phase[1].add_design_parameter('thrust', opt=False, val=self.sc.twr)

        # objective
        self.phase[1].add_objective('m', loc='final', scaler=-1.0)

    def set_initial_guess(self, theta, check_partials=False):

        # time
        self.p[self.phase_name[0] + '.t_initial'] = 0.0
        self.p[self.phase_name[0] + '.t_duration'] = self.tof[0]/self.body.tc
        self.p[self.phase_name[1] + '.t_initial'] = self.tof[0]/self.body.tc
        self.p[self.phase_name[1] + '.t_duration'] = self.tof[1]/self.body.tc

        self.p[self.phase_name[0] + '.states:r'] =\
            self.phase[0].interpolate(ys=(self.rp/self.body.R, self.r_switch/self.body.R), nodes='state_input')
        self.p[self.phase_name[0] + '.states:theta'] = self.phase[0].interpolate(ys=(0.0, theta), nodes='state_input')
        self.p[self.phase_name[0] + '.states:u'] = self.phase[0].interpolate(ys=(0.0, -100/self.body.vc),
                                                                             nodes='state_input')
        self.p[self.phase_name[0] + '.states:v'] = self.phase[0].interpolate(ys=(self.vp/self.body.vc, 0.0),
                                                                             nodes='state_input')
        self.p[self.phase_name[0] + '.states:m'] = self.phase[0].interpolate(ys=(self.sc.m0, self.sc.m0/2),
                                                                             nodes='state_input')
        self.p[self.phase_name[0] + '.controls:alpha'] = self.phase[0].interpolate(ys=(np.pi, np.pi/2),
                                                                                   nodes='control_input')

        self.p[self.phase_name[1] + '.states:r'] = self.phase[1].interpolate(ys=(self.r_switch/self.body.R, 1.0),
                                                                             nodes='state_input')
        self.p[self.phase_name[1] + '.states:u'] = self.phase[1].interpolate(ys=(-100/self.body.vc, 0.0),
                                                                             nodes='state_input')
        self.p[self.phase_name[1] + '.states:m'] = self.phase[1].interpolate(ys=(self.sc.m0/2, self.sc.m_dry),
                                                                             nodes='state_input')

        self.p.run_model()

        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)
