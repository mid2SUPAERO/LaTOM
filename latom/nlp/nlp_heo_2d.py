"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.nlp.nlp import MultiPhaseNLP
from latom.nlp.nlp_2d import TwoDimVarNLP, TwoDimNLP
from latom.guess.guess_heo_2d import TwoDimLLO2HEOGuess, TwoDim3PhasesLLO2HEOGuess
from latom.odes.odes_2d_group import ODE2dLLO2HEO, ODE2dLLO2Apo


class TwoDimLLO2HEONLP(TwoDimVarNLP):
    """TwoDimLLO2HEONLP transcribes a continuous-time optimal control problem for a two-dimensional transfer trajectory
    from a Low Lunar Orbit (LLO) to an Highly Elliptical Orbit (HEO) into a Non Linear Programming Problem (NLP) using
    the OpenMDAO and dymos libraries.

    The transfer is modeled as a single phase ascent trajectory from the departure LLO to the apoapsis of the arrival
    HEO. The thrust magnitude is allowed to vary over time.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        LLO altitude [m]
    rp : float
        HEO periapsis radius [m]
    t : float
        HEO period [s]
    alpha_bounds : tuple
        Lower and upper bounds on thrust vector direction [rad]
    t_bounds : tuple
        Time of flight bounds expressed as fraction of `tof` [-]
    method : str
        Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
        allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
    nb_seg : int
        Number of segments in which each phase is discretized
    order : int
        Transcription order within each phase, must be odd
    solver : str
        NLP solver, must be supported by OpenMDAO
    ph_name : str
        Name of the phase within OpenMDAO
    snopt_opts : dict or ``None``, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is ``None``
    rec_file : str or ``None``, optional
        Name of the file in which the computed solution is recorded or ``None``. Default is ``None``
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``
    u_bound : str or ``None``, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or ``None``. Default is ``lower``
    fix_final : bool, optional
        ``True`` if the final time is fixed, ``False`` otherwise. Default is ``True``

    """

    def __init__(self, body, sc, alt, rp, t, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False, u_bound='lower', fix_final=True):
        """Initializes TwoDimLLO2HEONLP class. """

        guess = TwoDimLLO2HEOGuess(body.GM, body.R, alt, rp, t, sc)

        TwoDimVarNLP.__init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                              guess, snopt_opts=snopt_opts, rec_file=rec_file, check_partials=check_partials,
                              u_bound=u_bound, fix_final=fix_final)

    def set_states_options(self, theta, u_bound=None):
        """Set the states variables options.

        Parameters
        ----------
        theta : float
            Unit reference value for spawn angle [rad]
        u_bound : str or ``None``, optional
                Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or ``None``. Default is ``lower``

        Returns
        -------

        """

        self.phase.set_state_options('r', fix_initial=True, fix_final=True, lower=1.0,
                                     ref0=self.guess.ht.depOrb.a/self.body.R,
                                     ref=self.guess.ht.arrOrb.ra/self.body.R)

        self.phase.set_state_options('theta', fix_initial=False, fix_final=True, lower=-np.pi/2, ref=theta)
        self.phase.set_state_options('u', fix_initial=True, fix_final=True, ref=self.v_circ/self.body.vc)
        self.phase.set_state_options('v', fix_initial=True, fix_final=True, lower=0.0, ref=self.v_circ/self.body.vc)

        self.phase.set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                     ref0=self.sc.m_dry, ref=self.sc.m0)


class TwoDimLLO2ApoNLP(TwoDimNLP):
    """TwoDimLLO2HEONLP transcribes a continuous-time optimal control problem for a two-dimensional transfer trajectory
    from a Low Lunar Orbit (LLO) to an Highly Elliptical Orbit (HEO) into a Non Linear Programming Problem (NLP) using
    the OpenMDAO and dymos libraries.

    The transfer is modeled as a single phase ascent trajectory from the departure LLO to the apoapsis of the arrival
    HEO. The thrust magnitude is allowed to vary over time.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        LLO altitude [m]
    rp : float
        HEO periapsis radius [m]
    t : float
        HEO period [s]
    alpha_bounds : tuple
        Lower and upper bounds on thrust vector direction [rad]
    t_bounds : tuple
        Time of flight bounds expressed as fraction of `tof` [-]
    method : str
        Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
        allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
    nb_seg : int
        Number of segments in which each phase is discretized
    order : int
        Transcription order within each phase, must be odd
    solver : str
        NLP solver, must be supported by OpenMDAO
    ph_name : str
        Name of the phase within OpenMDAO
    snopt_opts : dict or ``None``, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is ``None``
    rec_file : str or ``None``, optional
        Name of the file in which the computed solution is recorded or ``None``. Default is ``None``
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``
    params : dict or ``None``, optional
        Optional parameters to be passed when a continuation method is employed and the NLP guess is not defined or
        ``None``. Default is ``None``

    Attributes
    ----------
    guess : TwoDimLLO2HEOGuess or ``None``
        Initial guess or ``None`` if continuation is used

    """

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
        # self.add_timeseries_output()  # add semi-major axis, specific energy and angular momentum to the outputs

        if params is None:
            self.set_options(self.guess.ht.depOrb.rp, self.guess.ht.transfer.vp, self.guess.pow.thetaf,
                             self.guess.pow.tf, t_bounds=t_bounds)
            self.set_initial_guess(check_partials=check_partials)
        else:
            self.set_options(params['rp_llo'], params['vp_hoh'], params['thetaf_pow'], params['tof'],
                             t_bounds=t_bounds)
            self.set_continuation_guess(params['tof'], params['states'], params['controls'],
                                        check_partials=check_partials)

    def add_timeseries_output(self, names=('a', 'eps', 'h')):
        """Adds the semi-major axis, specific energy and specific angular momentum magnitude to the time series outputs
        of the phase.

        Parameters
        ----------
        names : iterable
            List of strings corresponding to the names of the variables to be added to the outputs

        """

        for n in names:
            self.phase.add_timeseries_output(n, shape=(1,))

    def set_options(self, rp, vp, thetaf, tof, t_bounds=None):
        """Set the states, controls and time options, add the design parameters and boundary constraints, define the
        objective of the optimization.

        Parameters
        ----------
        rp : float
            Unit reference value for lengths [m]
        vp : float
            Unit reference value for velocities [m/s]
        thetaf : float
            Unit reference value for spawn angle [rad]
        tof : float
            Guessed time of flight [s]
        t_bounds : iterable or ``None``, optional
            Time of flight lower and upper bounds expressed as fraction of `tof` [-]

        """

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
        """Set the initial guess for a single solution or the first solution of a continuation procedure.

        Parameters
        ----------
        check_partials : bool, optional
            Check the partial derivatives computed analytically against complex step method. Default is ``False``
        fix_final : bool, optional
            ``True`` if the final time is fixed, ``False`` otherwise. Default is ``False``
        throttle : bool, optional
            ``True`` of variable thrust, ``False`` otherwise

        """

        TwoDimNLP.set_initial_guess(self, check_partials=check_partials, fix_final=fix_final, throttle=throttle)

    def set_continuation_guess(self, tof, states, controls, check_partials=False):
        """Set the initial guess for the solution ``k+1`` as the optimal transfer found for the solution ``k`` during
        a continuation procedure.

        Parameters
        ----------
        tof : float
            Time of flight for the previous optimal solution [s]
        states : ndarray
            States on the states discretization nodes for the previous optimal solution
        controls : ndarray
            Controls on the controls discretization nodes for the previous optimal solution
        check_partials : bool, optional
            Check the partial derivatives computed analytically against complex step method. Default is ``False``

        """

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


class TwoDim3PhasesLLO2HEONLP(MultiPhaseNLP):
    """TwoDim3PhasesLLO2HEONLP transcribes an optimal control problem for a three-phases ascent trajectory from a
    circular Low Lunar Orbit (LLO) to an Highly Elliptical Orbit (HEO) into a Non Linear Programming Problem (NLP) using
    the OpenMDAO and dymos libraries.

    The transfer is modeled with a first powered phase at maximum thrust to leave the initial LLO, and intermediate
    coasting phase and a second powered phase to inject in the target HEO.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Periselene altitude at which the powered descent is initiated [m]
    rp : float
        HEO periapsis radius [m]
    t : float
        HEO period [s]
    alpha_bounds : iterable
        Lower and upper bounds on thrust vector direction [rad]
    t_bounds : tuple
        Time of flight bounds expressed as fraction of `tof` [-]
    method : str
        Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
        allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
    nb_seg : int or tuple
        Number of segments in which each phase is discretized
    order : int or tuple
        Transcription order within each phase, must be odd
    solver : str
        NLP solver, must be supported by OpenMDAO
    ph_name : tuple
        Name of the phase within OpenMDAO
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``

    Attributes
    ----------
    guess : TwoDim3PhasesLLO2HEOGuess
        Initial guess for the iterative NLP solution
    tof_adim : ndarray
        Time of flight in non-dimensional units [-]

    """

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
        """Set the initial guess for a given `Phase`.

        Parameters
        ----------
        state_nodes : ndarray
            Indexes corresponding to the states discretization nodes within the specified transcription
        control_nodes : ndarray
            Indexes corresponding to the controls discretization nodes within the specified transcription
        phase_name : str
            Name of the current `Phase` object

        """

        self.p[phase_name + '.states:r'] = np.take(self.guess.states[:, 0]/self.body.R, state_nodes)
        self.p[phase_name + '.states:theta'] = np.take(self.guess.states[:, 1], state_nodes)
        self.p[phase_name + '.states:u'] = np.take(self.guess.states[:, 2]/self.body.vc, state_nodes)
        self.p[phase_name + '.states:v'] = np.take(self.guess.states[:, 3]/self.body.vc, state_nodes)
        self.p[phase_name + '.states:m'] = np.take(self.guess.states[:, 4], state_nodes)
        self.p[phase_name + '.controls:alpha'] = np.take(self.guess.controls[:, 1], control_nodes)
