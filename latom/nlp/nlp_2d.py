"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.nlp.nlp import SinglePhaseNLP, MultiPhaseNLP
from latom.odes.odes_2d import ODE2dVarThrust, ODE2dConstThrust, ODE2dVertical
from latom.odes.odes_2d_group import ODE2dVToff
from latom.guess.guess_2d import TwoDimAscGuess, TwoDimDescGuess


class TwoDimNLP(SinglePhaseNLP):
    """TwoDimNLP class transcribes a two-dimensional, continuous-time optimal control problem in trajectory
    optimization into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-dimensional trajectory is described in polar coordinates centered at the center of the attracting body.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Orbit altitude [m]
    alpha_bounds : iterable
        Lower and upper bounds on thrust vector direction [rad]
    method : str
        Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
        allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
    nb_seg : int
        Number of segments in which each phase is discretized
    order : int
        Transcription order within each phase, must be odd
    solver : str
        NLP solver, must be supported by OpenMDAO
    ode_class : ExplicitComponent
        Instance of OpenMDAO `ExplicitComponent` describing the Ordinary Differential Equations (ODEs) that drive the
        system dynamics
    ode_kwargs : dict
        Keywords arguments to be passed to `ode_class`
    ph_name : str
        Name of the phase within OpenMDAO
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None

    Attributes
    ----------
    alt : float
        Orbit altitude [m]
    alpha_bounds : ndarray
        Lower and upper bounds on thrust vector direction [rad]
    r_circ : float
        Orbit radius [m]
    v_circ : float
        Orbital velocity [m/s]
    guess : TwoDimGuess
        Initial guess to be provided before solving the NLP

    """

    def __init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ode_class,
                 ode_kwargs, ph_name, snopt_opts=None, rec_file=None):
        """Initializes TwoDimNLP class. """

        # set central body, spacecraft and transcription
        SinglePhaseNLP.__init__(self, body, sc, method, nb_seg, order, solver, ode_class, ode_kwargs, ph_name,
                                snopt_opts=snopt_opts, rec_file=rec_file)

        self.alt = alt
        self.alpha_bounds = np.asarray(alpha_bounds)
        self.r_circ = body.R + self.alt
        self.v_circ = (body.GM/self.r_circ)**0.5

        self.guess = None

    def set_states_options(self, theta, u_bound=None):
        """Set options on the state variables of the NLP.

        Parameters
        ----------
        theta : float
            Reference value for spawn angle [rad]
        u_bound : str or None, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is None

        """

        # radius
        self.phase.set_state_options('r', fix_initial=True, fix_final=True, lower=1.0, ref0=1.0,
                                     ref=self.r_circ/self.body.R)

        # angle (all cases except descent with variable thrust and constraint on minimum safe altitude)
        if theta > 0.0:
            self.phase.set_state_options('theta', fix_initial=True, fix_final=False, lower=0.0, ref=theta)

        # angle (only descent with variable thrust and constraint on minimum safe altitude)
        else:
            self.phase.set_state_options('theta', fix_initial=False, fix_final=True, upper=0.0, adder=-theta,
                                         scaler=-1.0/theta)

        # positive radial velocity (ascent)
        if u_bound == 'lower':
            self.phase.set_state_options('u', fix_initial=True, fix_final=True, lower=0.0, ref=self.v_circ/self.body.vc)

        # negative radial velocity (descent)
        elif u_bound == 'upper':
            self.phase.set_state_options('u', fix_initial=True, fix_final=True, upper=0.0,
                                         adder=self.v_circ/self.body.vc, scaler=self.body.vc/self.v_circ)

        # no path constraints on radial velocity
        elif u_bound is None:
            self.phase.set_state_options('u', fix_initial=True, fix_final=True, ref=self.v_circ/self.body.vc)
        else:
            raise ValueError('u_bound must be either lower, upper or None')

        # tangential velocity
        self.phase.set_state_options('v', fix_initial=True, fix_final=True, lower=0.0, ref=self.v_circ/self.body.vc)

        # spacecraft mass
        self.phase.set_state_options('m', fix_initial=True, fix_final=False, lower=self.sc.m_dry, upper=self.sc.m0,
                                     ref0=self.sc.m_dry, ref=self.sc.m0)

    def set_controls_options(self, throttle=True):
        """Set options on control variables.

        Parameters
        ----------
        throttle : bool, optional
            ``True`` for variable thrust magnitude, ``False`` otherwise. Default is `'True``

        """

        if throttle:
            twr_min = self.sc.T_min / self.sc.m0 / self.body.g
            self.phase.add_control('thrust', fix_initial=False, fix_final=False, continuity=False,
                                   rate_continuity=False, rate2_continuity=False, lower=twr_min, upper=self.sc.twr,
                                   ref0=twr_min, ref=self.sc.twr)
        else:
            self.phase.add_design_parameter('thrust', opt=False, val=self.sc.twr)

        self.phase.add_control('alpha', fix_initial=False, fix_final=False, continuity=True, rate_continuity=True,
                               rate2_continuity=False, lower=self.alpha_bounds[0], upper=self.alpha_bounds[1],
                               ref=self.alpha_bounds[1])

        self.phase.add_design_parameter('w', opt=False, val=self.sc.w/self.body.vc)

    def set_initial_guess(self, check_partials=False, fix_final=False, throttle=True):
        """Set the initial guess for the iterative solution of the NLP.

        Parameters
        ----------
        check_partials : bool, optional
            Check the partial derivatives computed analytically against complex step method. Default is ``False``
        fix_final : bool, optional
            ``True`` if the final time is fixed, ``False`` otherwise. Default is ``False``
        throttle : bool, optional
            ``True`` for variable thrust magnitude, ``False`` otherwise. Default is `'True``

        """

        self.set_initial_guess_interpolation(bcs=np.ones((7, 2)), check_partials=False, throttle=throttle)
        self.guess.compute_trajectory(fix_final=fix_final, t_eval=self.t_all*self.body.tc, throttle=throttle)

        self.p[self.phase_name + '.states:r'] = np.take(self.guess.states[:, 0]/self.body.R, self.state_nodes)
        self.p[self.phase_name + '.states:theta'] = np.take(self.guess.states[:, 1], self.state_nodes)
        self.p[self.phase_name + '.states:u'] = np.take(self.guess.states[:, 2]/self.body.vc, self.state_nodes)
        self.p[self.phase_name + '.states:v'] = np.take(self.guess.states[:, 3]/self.body.vc, self.state_nodes)
        self.p[self.phase_name + '.states:m'] = np.take(self.guess.states[:, 4], self.state_nodes)
        self.p[self.phase_name + '.controls:alpha'] = np.take(self.guess.controls[:, 1], self.control_nodes)

        if throttle:  # variable thrust
            self.p[self.phase_name + '.controls:thrust'] = np.take(self.guess.controls[:, 0]/self.sc.m0/self.body.g,
                                                                   self.control_nodes)

        self.p.run_model()

        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)

    def set_initial_guess_interpolation(self, bcs=np.ones((7, 2)), check_partials=False, throttle=True):
        """Set the initial guess for the solution of the NLP interpolating the Boundary Conditions (BCs) imposed on the
        state and control variables.

        Parameters
        ----------
        bcs : ndarray, optional
            Boundary Conditions on state and control variables. Default is all ones
        check_partials : bool, optional
            Check the partial derivatives computed analytically against complex step method. Default is ``False``
        throttle : bool, optional
            ``True`` for variable thrust magnitude, ``False`` otherwise. Default is `'True``

        """

        self.set_time_guess(self.tof)

        self.p[self.phase_name + '.states:r'] = self.phase.interpolate(ys=bcs[0], nodes='state_input')
        self.p[self.phase_name + '.states:theta'] = self.phase.interpolate(ys=bcs[1], nodes='state_input')
        self.p[self.phase_name + '.states:u'] = self.phase.interpolate(ys=bcs[2], nodes='state_input')
        self.p[self.phase_name + '.states:v'] = self.phase.interpolate(ys=bcs[3], nodes='state_input')
        self.p[self.phase_name + '.states:m'] = self.phase.interpolate(ys=bcs[4], nodes='state_input')

        self.p[self.phase_name + '.controls:alpha'] = self.phase.interpolate(ys=bcs[5], nodes='control_input')

        if throttle:
            self.p[self.phase_name + '.controls:alpha'] = self.phase.interpolate(ys=bcs[6], nodes='control_input')

        self.p.run_model()

        if check_partials:
            self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)


class TwoDimConstNLP(TwoDimNLP):
    """TwoDimConstNLP class transcribes a two-dimensional, continuous-time optimal control problem in trajectory
    optimization into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-dimensional trajectory is described in polar coordinates centered at the center of the attracting body.
    The thrust delivered by the spacecraft engines is supposed to have a constant magnitude throughout the whole phase.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Orbit altitude [m]
    theta : float
        Guessed spawn angle [rad]
    alpha_bounds : tuple
        Lower and upper bounds on thrust vector direction [rad]
    tof : float
        Guessed time of flight [s]
    t_bounds : tuple
        Time of flight bounds expressed as fraction of TOF [-]
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
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None
    u_bound : str or None, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is None

    """

    def __init__(self, body, sc, alt, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, u_bound=None):
        """Initializes TwoDimConstNLP class. """

        ode_kwargs = {'GM': 1.0, 'T': sc.twr, 'w': sc.w / body.vc}

        TwoDimNLP.__init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ODE2dConstThrust,
                           ode_kwargs, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.set_options(theta, tof, t_bounds, u_bound=u_bound)
        self.setup()

    def set_options(self, theta, tof, t_bounds, u_bound=None):
        """Set options on state and control variables, time and objective function.

        Parameters
        ----------
        theta : float
            Guessed spawn angle [rad]
        tof : float
            Guessed time of flight [s]
        t_bounds : tuple
            Time of flight bounds expressed as fraction of TOF [-]
        u_bound : str or None, optional
                Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is None

        """

        self.set_states_options(theta, u_bound=u_bound)
        self.set_controls_options(throttle=False)
        self.set_time_options(tof, t_bounds)
        self.set_objective()


class TwoDimAscConstNLP(TwoDimConstNLP):
    """TwoDimAscConstNLP class transcribes a two-dimensional, continuous-time optimal control problem in trajectory
    optimization into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-dimensional ascent trajectory is described in polar coordinates centered at the center of the
    attracting body. The thrust delivered by the spacecraft engines is supposed to have a constant magnitude throughout
    the whole phase.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Orbit altitude [m]
    theta : float
        Guessed spawn angle [rad]
    alpha_bounds : tuple
        Lower and upper bounds on thrust vector direction [rad]
    tof : float
        Guessed time of flight [s]
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
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``
    u_bound : str or None, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is ``lower``

    """

    def __init__(self, body, sc, alt, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                 ph_name, snopt_opts=None, rec_file=None, check_partials=False, u_bound='lower'):
        """Initializes TwoDimAscConstNLP class. """

        TwoDimConstNLP.__init__(self, body, sc, alt, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                                ph_name, snopt_opts=snopt_opts, rec_file=rec_file, u_bound=u_bound)

        bcs = np.array([[1.0, self.r_circ/self.body.R], [0.0, theta], [0.0, 0.0], [0.0, self.v_circ/self.body.vc],
                        [self.sc.m0, self.sc.m_dry], [0.0, 0.0]])

        self.set_initial_guess_interpolation(bcs, check_partials=check_partials, throttle=False)


class TwoDimDescConstNLP(TwoDimConstNLP):
    """TwoDimDescConstNLP class transcribes a two-dimensional, continuous-time optimal control problem in trajectory
    optimization into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-dimensional descent trajectory is described in polar coordinates centered at the center of the
    attracting body. The thrust delivered by the spacecraft engines is supposed to have a constant magnitude throughout
    the whole phase.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Orbit altitude [m]
    theta : float
        Guessed spawn angle [rad]
    alpha_bounds : tuple
        Lower and upper bounds on thrust vector direction [rad]
    tof : float
        Guessed time of flight [s]
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
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``
    u_bound : str or None, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is ``upper``

    Attributes
    ----------
    vp : float
        Velocity at the periapsis of the Hohmann transfer where the final powered descent is initiated [m/s]

    """

    def __init__(self, body, sc, alt, vp, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                 ph_name, snopt_opts=None, rec_file=None, check_partials=False, u_bound='upper'):
        """Initializes TwoDimDescConstNLP class. """

        TwoDimConstNLP.__init__(self, body, sc, alt, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                                ph_name, snopt_opts=snopt_opts, rec_file=rec_file, u_bound=u_bound)

        self.vp = vp

        bcs = np.array([[self.r_circ/self.body.R, 1.0], [0.0, theta], [0.0, 0.0], [self.vp/self.body.vc, 0.0],
                        [self.sc.m0, self.sc.m_dry], [1.5*np.pi, np.pi/2]])

        self.set_initial_guess_interpolation(bcs, check_partials=check_partials, throttle=False)


class TwoDimVarNLP(TwoDimNLP):
    """TwoDimVarNLP class transcribes a two-dimensional, continuous-time optimal control problem in trajectory
    optimization into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-dimensional trajectory is described in polar coordinates centered at the center of the attracting body.
    The thrust delivered by the spacecraft engines varies in magnitude during the phase.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Orbit altitude [m]
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
    guess : TwoDimGuess
        Initial guess for the NLP solution
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``
    u_bound : str or None, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is None
    fix_final : bool, optional
        ``True`` if the final time is fixed, ``False`` otherwise. Default is ``False``

    Attributes
    ----------
    guess : TwoDimGuess
        Initial guess for the NLP solution

    """

    def __init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name, guess,
                 snopt_opts=None, rec_file=None, check_partials=False, u_bound=None, fix_final=False):
        """Initializes TwoDimVarNLP class. """

        ode_kwargs = {'GM': 1.0, 'w': sc.w / body.vc}

        TwoDimNLP.__init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ODE2dVarThrust, ode_kwargs,
                           ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.guess = guess

        self.set_options(np.pi, t_bounds, u_bound=u_bound)  # was pi/2
        self.setup()
        self.set_initial_guess(check_partials=check_partials, fix_final=fix_final)

    def set_options(self, theta, t_bounds, u_bound=None):
        """Set options on state and control variables, time and objective function.

        Parameters
        ----------
        theta : float
            Guessed spawn angle [rad]
        t_bounds : tuple
            Time of flight bounds expressed as fraction of `tof` [-]
        u_bound : str or None, optional
                Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is None

        """

        self.set_states_options(theta, u_bound=u_bound)
        self.set_controls_options(throttle=True)
        self.set_time_options(self.guess.tf, t_bounds)
        self.set_objective()


class TwoDimAscVarNLP(TwoDimVarNLP):
    """TwoDimAscVarNLP class transcribes a two-dimensional, continuous-time optimal control problem in trajectory
    optimization into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-dimensional ascent trajectory is described in polar coordinates centered at the center of the attracting
    body. The thrust delivered by the spacecraft engines varies in magnitude during the phase.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Orbit altitude [-]
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
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``
    u_bound : str or None, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is ``lower``
    fix_final : bool, optional
        ``True`` if the final time is fixed, ``False`` otherwise. Default is ``False``

    """

    def __init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name, snopt_opts=None,
                 rec_file=None, check_partials=False, u_bound='lower', fix_final=False):
        """Initializes TwoDimAscVarNLP class. """

        TwoDimVarNLP.__init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                              TwoDimAscGuess(body.GM, body.R, alt, sc), snopt_opts=snopt_opts, rec_file=rec_file,
                              check_partials=check_partials, u_bound=u_bound, fix_final=fix_final)


class TwoDimDescVarNLP(TwoDimVarNLP):
    """TwoDimDescVarNLP class transcribes a two-dimensional, continuous-time optimal control problem in trajectory
    optimization into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-dimensional descent trajectory is described in polar coordinates centered at the center of the attracting
    body. The thrust delivered by the spacecraft engines varies in magnitude during the phase.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Orbit altitude [m]
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
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``
    u_bound : str or None, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is ``upper``
    fix_final : bool, optional
        ``True`` if the final time is fixed, ``False`` otherwise. Default is ``False``

    """

    def __init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name, snopt_opts=None,
                 rec_file=None, check_partials=False, u_bound='upper', fix_final=False):
        """Initializes TwoDimDescVarNLP class. """

        TwoDimVarNLP.__init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                              TwoDimDescGuess(body.GM, body.R, alt, sc), snopt_opts=snopt_opts, rec_file=rec_file,
                              check_partials=check_partials, u_bound=u_bound, fix_final=fix_final)


class TwoDimVToffNLP(TwoDimVarNLP):
    """TwoDimVToffNLP class transcribes a two-dimensional, continuous-time optimal control problem in trajectory
    optimization into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-dimensional ascent/descent trajectory is described in polar coordinates centered at the center of the
    attracting body. The thrust delivered by the spacecraft engines varies in magnitude during the phase.
    An appropriate path constraint is imposed on the spacecraft state to guarantee a vertical take-off or landing.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Orbit altitude [m]
    alt_safe : float
        Minimum altitude above the Moon surface to be maintained by the spacecraft far from the launch site [m]
    slope : float
        Slope of the path constraint on the spacecraft radius and angle close to the launch site.
        Higher the value, steeper the ascent/descent [-]
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
    guess : TwoDimGuess
        Initial guess for the NLP solution
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``
    u_bound : str or None, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is None
    fix_final : bool, optional
        ``True`` if the final time is fixed, ``False`` otherwise. Default is ``False``

    Attributes
    ----------
    alt_safe : float
        Minimum altitude above the Moon surface to be maintained by the spacecraft far from the launch site [m]
    slope : float
        Slope of the path constraint on the spacecraft radius and angle close to the launch site.
        Higher the value, steeper the ascent [-]
    guess : TwoDimGuess
        Initial guess for the NLP solution

    """

    def __init__(self, body, sc, alt, alt_safe, slope, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 guess, snopt_opts=None, rec_file=None, check_partials=False, u_bound=None, fix_final=False):
        """Initializes TwoDimVToffNLP class. """

        ode_kwargs = {'GM': 1.0, 'w': sc.w/body.vc, 'R': 1.0, 'alt_safe': alt_safe/body.R, 'slope': slope}

        TwoDimNLP.__init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ODE2dVToff, ode_kwargs,
                           ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.alt_safe = alt_safe
        self.slope = slope

        self.guess = guess

        self.set_options(np.sign(slope)*np.pi, t_bounds, u_bound=u_bound)
        self.setup()
        self.set_initial_guess(check_partials=check_partials, fix_final=fix_final)

    def set_options(self, theta, t_bounds, u_bound=None):
        """Set the states and controls options and the path constraint.

        Parameters
        ----------
        theta : float
            Guessed spawn angle [rad]
        t_bounds : tuple
            Time of flight bounds expressed as fraction of `tof` [-]
        u_bound : str or None, optional
                Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is None

        """

        self.phase.add_path_constraint('dist_safe', lower=0.0, ref=self.alt_safe/self.body.R)
        self.phase.add_timeseries_output('r_safe')

        TwoDimVarNLP.set_options(self, theta, t_bounds, u_bound=u_bound)


class TwoDimAscVToffNLP(TwoDimVToffNLP):
    """TwoDimAscVToffNLP class transcribes a two-dimensional, continuous-time optimal control problem in trajectory
    optimization into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-dimensional ascent trajectory is described in polar coordinates centered at the center of the attracting
    body. The thrust delivered by the spacecraft engines varies in magnitude during the phase.
    An appropriate path constraint is imposed on the spacecraft state to guarantee a vertical take-off.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Orbit altitude [m]
    alt_safe : float
        Minimum altitude above the Moon surface to be maintained by the spacecraft far from the launch site [m]
    slope : float
        Slope of the path constraint on the spacecraft radius and angle close to the launch site.
        Higher the value, steeper the ascent [-]
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
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``
    u_bound : str or None, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is ``lower``
    fix_final : bool, optional
        ``True`` if the final time is fixed, ``False`` otherwise. Default is ``False``

    """

    def __init__(self, body, sc, alt, alt_safe, slope, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False, u_bound='lower', fix_final=False):
        """Initializes TwoDimAscVToffNLP class. """

        TwoDimVToffNLP.__init__(self, body, sc, alt, alt_safe, slope, alpha_bounds, t_bounds, method, nb_seg, order,
                                solver, ph_name, TwoDimAscGuess(body.GM, body.R, alt, sc), snopt_opts=snopt_opts,
                                rec_file=rec_file, check_partials=check_partials, u_bound=u_bound, fix_final=fix_final)


class TwoDimDescVLandNLP(TwoDimVToffNLP):
    """TwoDimAscVToffNLP class transcribes a two-dimensional, continuous-time optimal control problem in trajectory
    optimization into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-dimensional descent trajectory is described in polar coordinates centered at the center of the attracting
    body. The thrust delivered by the spacecraft engines varies in magnitude during the phase.
    An appropriate path constraint is imposed on the spacecraft state to guarantee a vertical landing.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Orbit altitude [m]
    alt_safe : float
        Minimum altitude above the Moon surface to be maintained by the spacecraft far from the launch site [m]
    slope : float
        Slope of the path constraint on the spacecraft radius and angle close to the launch site.
        Higher the value, steeper the descent [-]
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
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None
    check_partials : bool, optional
        Check the partial derivatives computed analytically against complex step method. Default is ``False``
    u_bound : str or None, optional
            Bounds on spacecraft radial velocity between ``lower`` and ``upper`` or None. Default is ``upper``
    fix_final : bool, optional
        ``True`` if the final time is fixed, ``False`` otherwise. Default is ``False``

    """

    def __init__(self, body, sc, alt, alt_safe, slope, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None, check_partials=False, u_bound='upper', fix_final=True):
        """Initializes TwoDimDescVLandNLP class. """

        TwoDimVToffNLP.__init__(self, body, sc, alt, alt_safe, slope, alpha_bounds, t_bounds, method, nb_seg, order,
                                solver, ph_name, TwoDimDescGuess(body.GM, body.R, alt, sc), snopt_opts=snopt_opts,
                                rec_file=rec_file, check_partials=check_partials, u_bound=u_bound, fix_final=fix_final)


class TwoDimDescTwoPhasesNLP(MultiPhaseNLP):
    """TwoDimDescTwoPhasesNLP transcribes a continuous-time optimal control problem for a two-dimensional descent
    trajectory into a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    The two-phases transfer is constituted by an initial deorbit burn to lower the periapsis of the departure orbit,
    an Hohmann transfer, a first powered phase from its periapsis to a predetermined altitude or time to go and a final
    vertical descent at full thrust.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    alt : float
        Periselene altitude at which the powered descent is initiated [m]
    alt_switch : float
        Altitude at which the vertical descent is triggered [m]
    vp :float
        Periselene velocity at which the powered descent is initiated [m/s]
    theta : float
        Guessed spawn angle [rad]
    alpha_bounds : iterable
        Lower and upper bounds on thrust vector direction [rad]
    tof : iterable
        Guessed time of flight for the two phases [s]
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
    fix : str, optional
        ``alt`` to trigger the vertical phase at fixed altitude equal to `alt_switch`, ``time`` to trigger the vertical
        phase at fixed time to go equal to the second component of `tof`. Default is ``alt``

    Attributes
    ----------
    alt : float
        Periselene altitude at which the powered descent is initiated [m]
    alt_switch : float
        Altitude at which the vertical descent is triggered [m]
    rp : float
        Periselene radius at which the powered descent is initiated [m]
    r_switch : float
        Radius at which the vertical descent is triggered [m]
    vp :float
        Periselene velocity at which the powered descent is initiated [m/s]
    alpha_bounds : iterable
        Lower and upper bounds on thrust vector direction [rad]
    tof : iterable
        Guessed time of flight for the two phases [s]
    fix : str, optional
        ``alt`` to trigger the vertical phase at fixed altitude equal to `alt_switch`, ``time`` to trigger the vertical
        phase at fixed time to go equal to the second component of `tof`. Default is ``alt``

    """

    def __init__(self, body, sc, alt, alt_switch, vp, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                 ph_name, snopt_opts=None, rec_file=None, check_partials=False, fix='alt'):
        """Initializes TwoDimDescTwoPhasesNLP class. """

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
        """Set the time, states and control options for both phases and add the NLP objective.

        Parameters
        ----------
        theta : float
            Guessed spawn angle [rad]
        t_bounds : iterable
            Time of flight bounds expressed as fraction of `tof` [-]

        """

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
        """Set the initial guess for the NLP solution as simple linear interpolation of the Boundary Conditions.

        Parameters
        ----------
        theta : float
            Guessed spawn angle [rad]
        check_partials : bool, optional
            Check the partial derivatives computed analytically against complex step method. Default is ``False``

        """

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
