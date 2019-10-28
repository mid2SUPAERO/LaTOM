"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp import MultiPhaseNLP
from rpfm.nlp.nlp_2d import TwoDimVarNLP, TwoDimNLP
from rpfm.guess.guess_heo_2d import TwoDimLLO2HEOGuess, TwoDim3PhasesHEO2LLOGuess
from rpfm.odes.odes_2d import ODE2dLLO2Apo, ODE2dConstThrust, ODE2dCoast, ODE2dLLO2HEO


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


class TwoDim3PhasesLLO2HEONLP(MultiPhaseNLP):

    def __init__(self, body, sc, alt, rp, t, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False):

        self.guess = TwoDim3PhasesHEO2LLOGuess(body.GM, body.R, alt, rp, t, sc)
        self.tof_adim = np.reshape(np.array([self.guess.pow1.tf, self.guess.ht.tof,
                                             (self.guess.pow2.tf - self.guess.pow2.t0)]), (3, 1))/body.tc

        ode_kwargs = ({'GM': 1.0, 'T': sc.twr, 'w': sc.w / body.vc}, {'GM': 1.0},
                      {'GM': 1.0, 'T': sc.twr, 'w': sc.w / body.vc})

        ode_class = (ODE2dConstThrust, ODE2dCoast, ODE2dLLO2HEO)

        MultiPhaseNLP.__init__(self, body, sc, method, nb_seg, order, solver, ode_class, ode_kwargs, ph_name,
                               snopt_opts=snopt_opts, rec_file=rec_file)

        # set options

        # time
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

        # first phase
        self.phase[0].set_state_options('r', fix_initial=True, fix_final=False, lower=1.0, ref0=1.0,
                                        ref=self.guess.ht.rp/self.body.R)

        self.phase[0].set_state_options('theta', fix_initial=True, fix_final=False, lower=0.0,
                                        ref=self.guess.pow1.thetaf)
        self.phase[0].set_state_options('u', fix_initial=True, fix_final=False,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)
        self.phase[0].set_state_options('v', fix_initial=True, fix_final=False, lower=0.0,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)
        self.phase[0].set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                        ref0=self.sc.m_dry, ref=self.sc.m0)

        self.phase[0].add_control('alpha', fix_initial=False, fix_final=False, continuity=True, rate_continuity=True,
                                  rate2_continuity=False, lower=alpha_bounds[0], upper=alpha_bounds[1],
                                  ref=alpha_bounds[1])

        self.phase[0].add_design_parameter('w', opt=False, val=self.sc.w/self.body.vc)
        self.phase[0].add_design_parameter('thrust', opt=False, val=self.sc.twr)

        # second phase
        self.phase[1].set_state_options('r', fix_initial=False, fix_final=False, lower=1.0, ref0=1.0,
                                        ref=self.guess.ht.ra/self.body.R)
        self.phase[1].set_state_options('theta', fix_initial=False, fix_final=False, lower=0.0, ref=np.pi)
        self.phase[1].set_state_options('u', fix_initial=False, fix_final=False,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)
        self.phase[1].set_state_options('v', fix_initial=False, fix_final=False,
                                        ref=self.guess.ht.depOrb.vp/self.body.vc)

        # third phase
        self.phase[2].set_state_options('r', fix_initial=False, fix_final=False, lower=1.0, ref0=1.0,
                                        ref=self.guess.ht.ra/self.body.R)
        self.phase[2].set_state_options('theta', fix_initial=False, fix_final=False, lower=0.0,
                                        ref=np.pi)
        self.phase[2].set_state_options('u', fix_initial=False, fix_final=False,
                                        ref=self.guess.ht.arrOrb.va/self.body.vc)
        self.phase[2].set_state_options('v', fix_initial=False, fix_final=False,
                                        ref=self.guess.ht.arrOrb.va/self.body.vc)
        self.phase[2].set_state_options('m', fix_initial=False, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                        ref0=self.sc.m_dry, ref=self.sc.m0)

        self.phase[2].add_control('alpha', fix_initial=False, fix_final=False, continuity=True, rate_continuity=True,
                                  rate2_continuity=False, lower=alpha_bounds[0], upper=alpha_bounds[1],
                                  ref=alpha_bounds[1])

        self.phase[2].add_design_parameter('w', opt=False, val=self.sc.w / self.body.vc)
        self.phase[2].add_design_parameter('thrust', opt=False, val=self.sc.twr)

        # constraint on injection
        h = (self.guess.ht.arrOrb.ra/self.body.R)*(self.guess.ht.arrOrb.va/self.body.vc)

        self.phase[2].add_boundary_constraint('h', loc='final', equals=h, shape=(1,))
        self.phase[2].add_boundary_constraint('a', loc='final', equals=self.guess.ht.arrOrb.a/self.body.R, shape=(1,))

        # objective
        self.phase[2].add_objective('m', loc='final', scaler=-1.0)

        # linkage
        self.trajectory.link_phases(ph_name, vars=['time', 'r', 'theta', 'u', 'v'])
        self.trajectory.link_phases([ph_name[0], ph_name[2]], vars=['m', 'alpha', 'thrust', 'w'])

        # setup
        self.setup()

        # time grid and initial guess
        t_state = []
        t_control = []
        idx_state_control = []
        nb_nodes = [0]

        ti = np.reshape(np.array([0.0, self.tof_adim[0], self.tof_adim[0] + self.tof_adim[1]]), (3, 1))

        for i in range(3):
            ts, tc, idx = self.set_time_phase(ti[i, 0], self.tof_adim[i, 0], self.phase[i], self.phase_name[i])
            t_state.append(ts)
            t_control.append(tc)
            idx_state_control.append(idx)
            nb_nodes.append(len(tc) + nb_nodes[-1])

        t_row = np.hstack(t_control)
        t = np.reshape(t_row, (len(t_row), 1))

        self.guess.compute_trajectory(t_eval=t*self.body.tc)

        for i in range(3):

            idx = np.reshape(idx_state_control[i], (len(idx_state_control[i]), 1)) + nb_nodes[i]
            alpha_row = self.guess.controls[nb_nodes[i]:nb_nodes[i + 1], 1]
            alpha = np.reshape(alpha_row, (len(alpha_row), 1))

            self.set_initial_guess_phase(idx, alpha, self.phase_name[i])

        self.p.run_model()

        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)

    def set_time_phase(self, ti, tof, phase, phase_name):

        # set the initial time and the time of flight initial guesses
        self.p[phase_name + '.t_initial'] = ti
        self.p[phase_name + '.t_duration'] = tof

        self.p.run_model()  # run the model to compute the time grid
        t_all = self.p[phase_name + '.time']  # time vector for all nodes

        # states and controls nodes indices
        state_nodes = phase.options['transcription'].grid_data.subset_node_indices['state_input']
        control_nodes = phase.options['transcription'].grid_data.subset_node_indices['control_input']

        # time vectors for states and controls nodes
        t_control = np.take(t_all, control_nodes)
        t_state = np.take(t_all, state_nodes)

        # indices of the states time vector elements in the controls time vector
        idx_state_control = np.nonzero(np.isin(t_control, t_state))[0]

        return t_state, t_control, idx_state_control

    def set_initial_guess_phase(self, idx_state_control, alpha, phase_name):

        self.p[phase_name + '.states:r'] = np.take(self.guess.states[:, 0]/self.body.R, idx_state_control)
        self.p[phase_name + '.states:theta'] = np.take(self.guess.states[:, 1], idx_state_control)
        self.p[phase_name + '.states:u'] = np.take(self.guess.states[:, 2]/self.body.vc, idx_state_control)
        self.p[phase_name + '.states:v'] = np.take(self.guess.states[:, 3]/self.body.vc, idx_state_control)

        if phase_name != self.phase_name[1]:
            self.p[phase_name + '.states:m'] = np.take(self.guess.states[:, 4], idx_state_control)
            self.p[phase_name + '.controls:alpha'] = alpha
