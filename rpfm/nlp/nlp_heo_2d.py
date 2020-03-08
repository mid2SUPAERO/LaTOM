"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from dymos.transcriptions.common import PhaseLinkageComp

from rpfm.nlp.nlp import MultiPhaseNLP
from rpfm.nlp.nlp_2d import TwoDimVarNLP, TwoDimNLP
from rpfm.guess.guess_heo_2d import TwoDimLLO2HEOGuess, TwoDim2PhasesLLO2HEOGuess, TwoDim3PhasesLLO2HEOGuess
from rpfm.odes.odes_2d_group import ODE2dLLO2HEO, ODE2dLLO2Apo


class TwoDimLLO2HEONLP(TwoDimVarNLP):

    def __init__(self, body, sc, alt, rp, t, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False, u_bound='lower', fix_final=True):

        guess = TwoDimLLO2HEOGuess(body.GM, body.R, alt, rp, t, sc)

        TwoDimVarNLP.__init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                              guess, snopt_opts=snopt_opts, rec_file=rec_file, check_partials=check_partials,
                              u_bound=u_bound, fix_final=fix_final)

    def set_states_options(self, theta, u_bound=None):

        self.phase.set_state_options('r', fix_initial=True, fix_final=True, lower=1.0,
                                     ref0=self.guess.ht.depOrb.a/self.body.R,
                                     ref=self.guess.ht.arrOrb.ra/self.body.R)

        self.phase.set_state_options('theta', fix_initial=False, fix_final=True, lower=-np.pi/2, ref=theta)
        self.phase.set_state_options('u', fix_initial=True, fix_final=True, ref=self.v_circ/self.body.vc)
        self.phase.set_state_options('v', fix_initial=True, fix_final=True, lower=0.0, ref=self.v_circ/self.body.vc)

        self.phase.set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                     ref0=self.sc.m_dry, ref=self.sc.m0)


class TwoDimLLO2ApoNLP(TwoDimNLP):

    def __init__(self, body, sc, alt, rp, t, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False, params=None):

        if params is None:
            guess = TwoDimLLO2HEOGuess(body.GM, body.R, alt, rp, t, sc)
            ode_kwargs = {'GM': 1.0, 'T': sc.twr, 'w': sc.w / body.vc, 'ra': guess.ht.arrOrb.ra / body.R}
        else:
            guess = None
            ode_kwargs = {'GM': 1.0, 'T': sc.twr, 'w': sc.w / body.vc, 'ra': params['ra_heo'] / body.R}

        TwoDimNLP.__init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ODE2dLLO2Apo,
                           ode_kwargs, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.guess = guess

        if params is None:
            self.set_options(self.guess.ht.depOrb.rp, self.guess.ht.transfer.vp, self.guess.pow.thetaf,
                             self.guess.pow.tf, t_bounds=t_bounds)
            self.set_initial_guess(check_partials=check_partials)
        else:
            self.set_options(params['rp_llo'], params['vp_hoh'], params['thetaf_pow'], params['tof'],
                             t_bounds=t_bounds)
            self.set_continuation_guess(params['tof'], params['states'], params['controls'],
                                        check_partials=check_partials)

    def set_options(self, rp, vp, thetaf, tof, t_bounds=None):

        # states options
        self.phase.set_state_options('r', fix_initial=True, fix_final=False, lower=1.0, ref0=1.0, ref=rp/self.body.R)
        self.phase.set_state_options('theta', fix_initial=True, fix_final=False, lower=0.0, ref=thetaf)
        self.phase.set_state_options('u', fix_initial=True, fix_final=False, ref0=0.0, ref=vp/self.body.vc)
        self.phase.set_state_options('v', fix_initial=True, fix_final=False, lower=0.0, ref=vp/self.body.vc)
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
        self.set_time_options(tof, t_bounds)

        # constraint on injection
        self.phase.add_boundary_constraint('c', loc='final', equals=0.0, shape=(1,))

        # objective
        self.set_objective()

        # NLP setup
        self.setup()

    def set_initial_guess(self, check_partials=False, fix_final=False, throttle=False):
        TwoDimNLP.set_initial_guess(self, check_partials=check_partials, fix_final=fix_final, throttle=throttle)

    def set_continuation_guess(self, tof, states, controls, check_partials=False):

        self.p[self.phase_name + '.t_initial'] = 0.0
        self.p[self.phase_name + '.t_duration'] = tof/self.body.tc

        self.p[self.phase_name + '.states:r'] = np.reshape(states[:, 0]/self.body.R, (np.size(states[:, 0]), 1))
        self.p[self.phase_name + '.states:theta'] = np.reshape(states[:, 1], (np.size(states[:, 0]), 1))
        self.p[self.phase_name + '.states:u'] = np.reshape(states[:, 2]/self.body.vc, (np.size(states[:, 0]), 1))
        self.p[self.phase_name + '.states:v'] = np.reshape(states[:, 3]/self.body.vc, (np.size(states[:, 0]), 1))
        self.p[self.phase_name + '.states:m'] = np.reshape(states[:, 4], (np.size(states[:, 0]), 1))
        self.p[self.phase_name + '.controls:alpha'] = np.reshape(controls[:, 1], (np.size(controls[:, 1]), 1))

        self.p.run_model()

        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)


class TwoDim2PhasesLLO2HEONLP(MultiPhaseNLP):

    def __init__(self, body, sc, alt, rp, t, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=None):

        self.guess = TwoDim2PhasesLLO2HEOGuess(body.GM, body.R, alt, rp, t, sc)
        self.tof_adim = np.reshape(np.array([self.guess.pow1.tf, self.guess.pow2.tf]), (2, 1))/body.tc

        ode_kwargs = {'GM': 1.0, 'T': sc.twr, 'w': sc.w / body.vc}

        MultiPhaseNLP.__init__(self, body, sc, method, nb_seg, order, solver, (ODE2dLLO2HEO, ODE2dLLO2HEO),
                               (ode_kwargs, ode_kwargs), ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        # time options
        if t_bounds is None:
            for i in range(2):
                self.phase[i].set_time_options(fix_initial=True, duration_ref=self.tof_adim[i, 0])
        else:
            t_bounds = np.asarray(t_bounds) * self.tof_adim
            for i in range(2):
                self.phase[i].set_time_options(fix_initial=True, duration_ref=self.tof_adim[i, 0],
                                               duration_bounds=t_bounds[i])

        # states options

        # first phase (departure)
        self.phase[0].set_state_options('r', fix_initial=True, fix_final=False, lower=1.0, ref0=1.0,
                                        ref=self.guess.ht.depOrb.rp/self.body.R)
        self.phase[0].set_state_options('theta', fix_initial=True, fix_final=False, lower=0.0,
                                        ref=self.guess.pow1.thetaf)
        self.phase[0].set_state_options('u', fix_initial=True, fix_final=False,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)
        self.phase[0].set_state_options('v', fix_initial=True, fix_final=False, lower=0.0,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)
        self.phase[0].set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                        ref0=self.sc.m_dry, ref=self.sc.m0)

        # second phase (injection)
        self.phase[1].set_state_options('r', fix_initial=False, fix_final=False, lower=1.0,
                                        ref0=1.0,
                                        ref=self.guess.ht.depOrb.rp/self.body.R)
        self.phase[1].set_state_options('theta', fix_initial=True, fix_final=False, lower=0.0,
                                        ref0=self.guess.pow2.theta0,
                                        ref=self.guess.pow2.thetaf)
        self.phase[1].set_state_options('u', fix_initial=False, fix_final=False,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)
        self.phase[1].set_state_options('v', fix_initial=False, fix_final=False,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)
        self.phase[1].set_state_options('m', fix_initial=False, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                        ref0=self.sc.m_dry, ref=self.sc.m0)

        # controls and design parameters options
        for i in range(2):
            self.phase[i].add_control('alpha', fix_initial=False, fix_final=False, continuity=True,
                                      rate_continuity=True, rate2_continuity=False, lower=alpha_bounds[0],
                                      upper=alpha_bounds[1], ref=alpha_bounds[1])

            self.phase[i].add_design_parameter('w', opt=False, val=self.sc.w/self.body.vc)
            self.phase[i].add_design_parameter('thrust', opt=False, val=self.sc.twr)

        # constraints on injection
        self.phase[1].add_boundary_constraint('a', loc='final', equals=self.guess.ht.arrOrb.a/self.body.R, shape=(1,))
        self.phase[1].add_boundary_constraint('h', loc='final', equals=self.guess.ht.arrOrb.h/self.body.R/self.body.vc,
                                              shape=(1,))

        # objective
        self.phase[1].add_objective('m', loc='final', scaler=-1.0)

        # linkages
        self.trajectory.link_phases(ph_name, vars=['m'])

        linkage_comp = PhaseLinkageComp()
        self.trajectory.add_subsystem('linkage_comp', subsys=linkage_comp)

        for s in ['a', 'h']:

            linkage_comp.add_linkage(s, vars=(s,), equals=0.0, shape=(1,))
            var_name = []

            for i in range(2):
                self.phase[i].add_timeseries_output(s)
                ph_name_list = self.phase_name[i].split('.')
                var_name.append('.'.join([ph_name_list[1], 'rhs_disc', s]))

            link_name = ''.join(['linkage_comp.', s, '_', s])
            self.trajectory.connect(var_name[0], link_name + ':lhs', src_indices=[-1], flat_src_indices=True)
            self.trajectory.connect(var_name[1], link_name + ':rhs', src_indices=[0], flat_src_indices=True)

        # setup
        self.setup()

        # time grid and initial guess
        sn1, cn1, ts1, tc1, ta1 = self.set_time_phase(0.0, self.tof_adim[0, 0], self.phase[0], self.phase_name[0])
        sn2, cn2, ts2, tc2, ta2 = self.set_time_phase(0.0, self.tof_adim[1, 0], self.phase[1], self.phase_name[1])

        self.guess.compute_trajectory(t1=ta1*self.body.tc, t2=ta2*self.body.tc)
        self.set_initial_guess_phase(sn1, cn1, self.guess.pow1, self.phase_name[0])
        self.set_initial_guess_phase(sn2, cn2, self.guess.pow2, self.phase_name[1])

        self.p.run_model()

        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)

    def set_initial_guess_phase(self, state_nodes, control_nodes, guess, phase_name):

        self.p[phase_name + '.states:r'] = np.take(guess.states[:, 0]/self.body.R, state_nodes)
        self.p[phase_name + '.states:theta'] = np.take(guess.states[:, 1], state_nodes)
        self.p[phase_name + '.states:u'] = np.take(guess.states[:, 2]/self.body.vc, state_nodes)
        self.p[phase_name + '.states:v'] = np.take(guess.states[:, 3]/self.body.vc, state_nodes)
        self.p[phase_name + '.states:m'] = np.take(guess.states[:, 4], state_nodes)
        self.p[phase_name + '.controls:alpha'] = np.take(guess.controls[:, 1], control_nodes)


class TwoDim3PhasesLLO2HEONLP(MultiPhaseNLP):

    def __init__(self, body, sc, alt, rp, t, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False):

        self.guess = TwoDim3PhasesLLO2HEOGuess(body.GM, body.R, alt, rp, t, sc)

        self.tof_adim = np.reshape(np.array([self.guess.pow1.tf, self.guess.ht.tof,
                                             (self.guess.pow2.tf - self.guess.pow2.t0)]), (3, 1))/body.tc

        ode_kwargs = {'GM': 1.0, 'T': sc.twr, 'w': sc.w / body.vc}

        MultiPhaseNLP.__init__(self, body, sc, method, nb_seg, order, solver,
                               (ODE2dLLO2HEO, ODE2dLLO2HEO, ODE2dLLO2HEO),
                               (ode_kwargs, ode_kwargs, ode_kwargs), ph_name,
                               snopt_opts=snopt_opts, rec_file=rec_file)

        # time options
        if t_bounds is not None:

            t_bounds = np.asarray(t_bounds)*self.tof_adim
            duration_ref0 = np.mean(t_bounds, axis=1)
            duration_ref = t_bounds[:, 1]

            self.phase[0].set_time_options(fix_initial=True, duration_ref0=duration_ref0[0],
                                           duration_ref=duration_ref[0], duration_bounds=t_bounds[0])

            for i in range(1, 3):

                initial_ref0 = np.sum(duration_ref0[:i])
                initial_ref = np.sum(duration_ref[:i])
                initial_bounds = np.sum(t_bounds[:i, :], axis=0)

                self.phase[i].set_time_options(initial_ref0=initial_ref0, initial_ref=initial_ref,
                                               initial_bounds=initial_bounds, duration_ref0=duration_ref0[i],
                                               duration_ref=duration_ref[i], duration_bounds=t_bounds[i])

        else:

            self.phase[0].set_time_options(fix_initial=True, duration_ref0=0.0, duration_ref=self.tof_adim[0, 0])
            self.phase[1].set_time_options(initial_ref0=0.0, initial_ref=self.tof_adim[0, 0],
                                           duration_ref0=0.0, duration_ref=self.tof_adim[1, 0])
            self.phase[2].set_time_options(initial_ref0=0.0, initial_ref=(self.tof_adim[0, 0] + self.tof_adim[1, 0]),
                                           duration_ref0=0.0, duration_ref=self.tof_adim[2, 0])

        # states options - first phase
        self.phase[0].set_state_options('r', fix_initial=True, fix_final=False, lower=1.0, ref0=1.0,
                                        ref=self.guess.ht.rp/self.body.R)
        self.phase[0].set_state_options('theta', fix_initial=True, fix_final=False, lower=0.0,
                                        ref=self.guess.pow1.thetaf)
        self.phase[0].set_state_options('u', fix_initial=True, fix_final=False,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)
        self.phase[0].set_state_options('v', fix_initial=True, fix_final=False, lower=0.0,
                                        ref=self.guess.ht.transfer.vp/self.body.vc)
        self.phase[0].set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                        ref0=self.sc.m_dry, ref=self.sc.m0)

        # states options - second phase
        self.phase[1].set_state_options('r', fix_initial=False, fix_final=False, lower=1.0, ref0=1.0,
                                        ref=self.guess.ht.arrOrb.ra/self.body.R)
        self.phase[1].set_state_options('theta', fix_initial=False, fix_final=False, lower=0.0, ref=np.pi)
        self.phase[1].set_state_options('u', fix_initial=False, fix_final=False,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)
        self.phase[1].set_state_options('v', fix_initial=False, fix_final=False,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)
        self.phase[1].set_state_options('m', fix_initial=False, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                        ref0=self.sc.m_dry, ref=self.sc.m0)

        # states options - third phase
        self.phase[2].set_state_options('r', fix_initial=False, fix_final=False, lower=1.0, ref0=1.0,
                                        ref=self.guess.ht.arrOrb.ra/self.body.R)
        self.phase[2].set_state_options('theta', fix_initial=False, fix_final=False, lower=0.0, ref=np.pi)
        self.phase[2].set_state_options('u', fix_initial=False, fix_final=False,
                                        ref=self.guess.ht.arrOrb.va/self.body.vc)
        self.phase[2].set_state_options('v', fix_initial=False, fix_final=False,
                                        ref=self.guess.ht.arrOrb.va/self.body.vc)
        self.phase[2].set_state_options('m', fix_initial=False, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                        ref0=self.sc.m_dry, ref=self.sc.m0)

        # controls options and design parameters
        for i in range(3):
            self.phase[i].add_control('alpha', fix_initial=False, fix_final=False, continuity=True,
                                      rate_continuity=True, rate2_continuity=False, lower=alpha_bounds[0],
                                      upper=alpha_bounds[1], ref=alpha_bounds[1])

            self.phase[i].add_design_parameter('w', opt=False, val=self.sc.w / self.body.vc)

            if i == 1:
                self.phase[i].add_design_parameter('thrust', opt=False, val=0.0)
            else:
                self.phase[i].add_design_parameter('thrust', opt=False, val=self.sc.twr)

        # constraint on injection
        h = (self.guess.ht.arrOrb.ra/self.body.R)*(self.guess.ht.arrOrb.va/self.body.vc)

        self.phase[2].add_boundary_constraint('h', loc='final', equals=h, shape=(1,))
        self.phase[2].add_boundary_constraint('a', loc='final', equals=self.guess.ht.arrOrb.a/self.body.R, shape=(1,))

        # objective
        self.phase[2].add_objective('m', loc='final', scaler=-1.0)

        # linkage
        self.trajectory.link_phases(ph_name, vars=['time', 'r', 'theta', 'u', 'v', 'm'])

        # additional outputs
        for i in range(2):
            for s in ['a', 'h']:
                self.phase[i].add_timeseries_output(s)

        # setup
        self.setup()

        # time grid and initial guess
        ti = np.reshape(np.array([0.0, self.tof_adim[0], self.tof_adim[0] + self.tof_adim[1]]), (3, 1))
        t_all = []
        state_nodes_abs = []
        control_nodes_abs = []
        nb_nodes = [0]

        for i in range(3):
            sn, cn, ts, tc, ta = self.set_time_phase(ti[i, 0], self.tof_adim[i, 0], self.phase[i], self.phase_name[i])

            t_all.append(ta)
            state_nodes_abs.append(sn + nb_nodes[-1])
            control_nodes_abs.append(cn + nb_nodes[-1])
            nb_nodes.append(len(ta) + nb_nodes[-1])

        self.guess.compute_trajectory(t_eval=np.vstack(t_all)*self.body.tc)

        for i in range(3):
            self.set_initial_guess_phase(state_nodes_abs[i], control_nodes_abs[i], self.phase_name[i])

        self.p.run_model()

        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)

    def set_initial_guess_phase(self, state_nodes, control_nodes, phase_name):

        self.p[phase_name + '.states:r'] = np.take(self.guess.states[:, 0]/self.body.R, state_nodes)
        self.p[phase_name + '.states:theta'] = np.take(self.guess.states[:, 1], state_nodes)
        self.p[phase_name + '.states:u'] = np.take(self.guess.states[:, 2]/self.body.vc, state_nodes)
        self.p[phase_name + '.states:v'] = np.take(self.guess.states[:, 3]/self.body.vc, state_nodes)
        self.p[phase_name + '.states:m'] = np.take(self.guess.states[:, 4], state_nodes)
        self.p[phase_name + '.controls:alpha'] = np.take(self.guess.controls[:, 1], control_nodes)
