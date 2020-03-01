"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from openmdao.api import Problem, Group, SqliteRecorder, DirectSolver, pyOptSparseDriver
from dymos import Phase, Trajectory, GaussLobatto, Radau, RungeKutta

from rpfm.utils.const import rec_excludes


class NLP:
    """NLP class transcribes a continuous-time optimal control problem in trajectory optimization into
    a Non Linear Programming Problem (NLP) using the OpenMDAO and dymos libraries.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    method : str
        Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
        allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
    nb_seg : int or tuple
        Number of segments in which each phase is discretized
    order : int or tuple
        Transcription order within each phase, must be odd
    solver : str
        NLP solver, must be supported by OpenMDAO
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide [1]_ for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None

    Attributes
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    method : str
        Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
        allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
    nb_seg : int or tuple
        Number of segments in which each phase is discretized
    order : int or tuple
        Transcription order within each phase, must be odd
    solver : str
        NLP solver, must be supported by OpenMDAO
    snopt_opts : dict or None
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide [1]_ for more details.
    rec_file : str or None
        Name of the file in which the computed solution is recorded or None
    p : Problem
        OpenMDAO `Problem` class instance representing the NLP
    trajectory : Trajectory
        Dymos `Trajectory` class instance representing the spacecraft trajectory
    p_exp : Problem
        OpenMDAO `Problem` class instance representing the explicitly simulated trajectory

    References
    ----------
    .. [1] Gill, Philip E., et al. User’s Guide for SNOPT Version 7.7: Software for Large-Scale Nonlinear Programming,
        Feb. 2019, p. 126.

    """

    def __init__(self, body, sc, method, nb_seg, order, solver, snopt_opts=None, rec_file=None):
        """Initializes NLP class. """

        # input parameters
        self.body = body
        self.sc = sc
        self.method = method
        self.nb_seg = nb_seg
        self.order = order
        self.solver = solver

        if self.solver == 'SNOPT':
            self.snopt_opts = snopt_opts
        else:
            self.snopt_opts = None

        self.rec_file = rec_file

        # Problem object
        self.p = Problem(model=Group())

        # Problem Driver
        self.p.driver = pyOptSparseDriver()
        self.p.driver.options['optimizer'] = self.solver
        self.p.driver.options['print_results'] = False
        self.p.driver.options['dynamic_derivs_sparsity'] = True

        if self.snopt_opts is not None:
            for k in self.snopt_opts.keys():
                self.p.driver.opt_settings[k] = self.snopt_opts[k]

        self.p.driver.declare_coloring(show_summary=True, show_sparsity=False)

        # Problem Recorder
        if rec_file is not None:

            recorder = SqliteRecorder(rec_file)
            opts = ['record_objectives', 'record_constraints', 'record_desvars']

            self.p.add_recorder(recorder)

            for opt in opts:
                self.p.recording_options[opt] = False

            self.p.recording_options['excludes'] = rec_excludes

        self.rec_file = rec_file

        # Trajectory object
        self.trajectory = self.p.model.add_subsystem('traj', Trajectory())

        # Problem object for explicit simulation
        self.p_exp = None

    def setup(self):
        """Set up the Jacobian type, linear solver and derivatives type. """

        self.p.model.options['assembled_jac_type'] = 'csc'
        self.p.model.linear_solver = DirectSolver()
        self.p.setup(check=True, force_alloc_complex=True, derivatives=True)

    def exp_sim(self, rec_file=None):
        """Explicitly simulate the implicitly obtained optimal solution using Scipy `solve_ivp` method. """

        if rec_file is not None:
            self.p_exp = self.trajectory.simulate(atol=1e-12, rtol=1e-12, record_file=rec_file)
        else:
            self.p_exp = self.trajectory.simulate(atol=1e-12, rtol=1e-12)

        self.cleanup()

    def cleanup(self):
        """Clean up resources. """

        self.trajectory.cleanup()
        self.p.driver.cleanup()
        self.p.cleanup()

    def __str__(self):
        """Prints info on the NLP.

        Returns
        -------
        s : str
            Info on the NLP

        """

        lines = ['\n{:^40s}'.format('NLP characteristics:'),
                 '\n{:<25s}{:<15s}'.format('Solver:', self.solver),
                 '{:<25s}{:<15s}'.format('Transcription method:', str(self.method)),
                 '{:<25s}{:<15s}'.format('Number of segments:', str(self.nb_seg)),
                 '{:<25s}{:<15s}'.format('Transcription order:', str(self.order))]

        s = '\n'.join(lines)

        return s


class SinglePhaseNLP(NLP):
    """SinglePhaseNLP transcribes a continuous-time optimal control problem in trajectory optimization constituted by
    a single phase into a Non Linear Programming Problem (NLP) using the libraries OpenMDAO and dymos.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
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
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide [1]_ for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None

    Attributes
    ----------
    phase : Phase
        Dymos `Phase` object representing the unique phase of the `Trajectory` class instance
    phase_name : str
        Complete name of the `Phase` instance within OpenMDAO
    state_nodes : ndarray
        Indexes corresponding to the state discretization nodes of the discretized phase
    control_nodes : ndarray
        Indexes corresponding to the control discretization nodes of the discretized phase
    t_all : ndarray
        Time instants corresponding to all discretization nodes [-]
    t_state : ndarray
        Time instants corresponding to the state discretization nodes [-]
    t_control : ndarray
        Time instants corresponding to the control discretization nodes [-]
    idx_state_control : ndarray
        Indexes of the state discretization nodes among the control discretization nodes
    tof : float
        Phase time of flight (TOF) [-]

    """

    def __init__(self, body, sc, method, nb_seg, order, solver, ode_class, ode_kwargs, ph_name, snopt_opts=None,
                 rec_file=None):

        NLP.__init__(self, body, sc, method, nb_seg, order, solver, snopt_opts=snopt_opts, rec_file=rec_file)

        # Transcription object
        if self.method == 'gauss-lobatto':
            self.transcription = GaussLobatto(num_segments=self.nb_seg, order=self.order, compressed=True)
        elif self.method == 'radau-ps':
            self.transcription = Radau(num_segments=self.nb_seg, order=self.order, compressed=True)
        elif self.method == 'runge-kutta':
            self.transcription = RungeKutta(num_segments=self.nb_seg, order=self.order, compressed=True)
        else:
            raise ValueError('method must be either gauss-lobatto or radau-ps')

        # Phase object
        self.phase = self.trajectory.add_phase(ph_name, Phase(ode_class=ode_class, ode_init_kwargs=ode_kwargs,
                                                              transcription=self.transcription))
        self.phase_name = ''.join(['traj.', ph_name])

        # discretization nodes
        self.state_nodes = None
        self.control_nodes = None
        self.t_all = None
        self.t_state = None
        self.t_control = None
        self.idx_state_control = None

        # time of flight
        self.tof = None

    def set_objective(self):
        """Set the NLP objective as the minimization of the opposite of the final spacecraft mass. """

        self.phase.add_objective('m', loc='final', scaler=-1.0)

    def set_time_options(self, tof, t_bounds):
        """Set the time options on the phase.

        Parameters
        ----------
        tof : float
            Phase time of flight (TOF) [-]
        t_bounds : tuple
            Time of flight lower and upper bounds expressed as a fraction of `tof`

        """

        self.tof = tof

        if t_bounds is not None:
            t_bounds = tof * np.asarray(t_bounds)
            self.phase.set_time_options(fix_initial=True, duration_ref=tof/self.body.tc,
                                        duration_bounds=t_bounds/self.body.tc)
        else:
            self.phase.set_time_options(fix_initial=True, duration_ref=tof/self.body.tc)

    def set_time_guess(self, tof):
        """Compute the time grid on the phase to retrieve the time instants corresponding to states, controls and all
        discretization nodes.

        Parameters
        ----------
        tof : float
            Phase time of flight (TOF) [-]

        """

        # set initial and transfer time
        self.p[self.phase_name + '.t_initial'] = 0.0
        self.p[self.phase_name + '.t_duration'] = tof/self.body.tc

        self.p.run_model()  # compute time grid

        # states and controls nodes
        state_nodes = self.phase.options['transcription'].grid_data.subset_node_indices['state_input']
        control_nodes = self.phase.options['transcription'].grid_data.subset_node_indices['control_input']

        self.state_nodes = np.reshape(state_nodes, (len(state_nodes), 1))
        self.control_nodes = np.reshape(control_nodes, (len(control_nodes), 1))

        # time on the discretization nodes
        t_all = self.p[self.phase_name + '.time']

        self.t_all = np.reshape(t_all, (len(t_all), 1))
        self.t_state = np.take(self.t_all, self.state_nodes)
        self.t_control = np.take(self.t_all, self.control_nodes)


class MultiPhaseNLP(NLP):
    """MultiPhaseNLP transcribes a continuous-time optimal control problem in trajectory optimization constituted by
    multiple phases into a Non Linear Programming Problem (NLP) using the libraries OpenMDAO and dymos.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class representing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class representing the spacecraft
    method : str
        Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
        allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
    nb_seg : int or tuple
        Number of segments in which each phase is discretized
    order : int or tuple
        Transcription order within each phase, must be odd
    solver : str
        NLP solver, must be supported by OpenMDAO
    ode_class : tuple
        Instance of OpenMDAO `ExplicitComponent` describing the Ordinary Differential Equations (ODEs) that drive the
        system dynamics
    ode_kwargs : tuple
        Keywords arguments to be passed to `ode_class`
    ph_name : tuple
        Name of the phase within OpenMDAO
    snopt_opts : dict or None, optional
        SNOPT optional settings expressed as key-value pairs. Refer to the SNOPT User Guide [1]_ for more details.
        Default is None
    rec_file : str or None, optional
        Name of the file in which the computed solution is recorded or None. Default is None

    Attributes
    ----------
    transcription : list
        List of dymos transcription class instances representing the transcription method, order and segments within
        each phase
    phase : list
        List of dymos `Phase` object representing the different phases of the `Trajectory` class instance
    phase_name : list
        List of complete names of the `Phase` instance within OpenMDAO

    """

    def __init__(self, body, sc, method, nb_seg, order, solver, ode_class, ode_kwargs, ph_name, snopt_opts=None,
                 rec_file=None):

        if isinstance(order, int):
            order = tuple(order for _ in range(len(nb_seg)))

        if isinstance(method, str):
            method = tuple(method for _ in range(len(nb_seg)))

        NLP.__init__(self, body, sc, method, nb_seg, order, solver, snopt_opts=snopt_opts, rec_file=rec_file)

        # Transcription objects list
        self.transcription = []

        for i in range(len(self.nb_seg)):
            if self.method[i] == 'gauss-lobatto':
                t = GaussLobatto(num_segments=self.nb_seg[i], order=self.order[i], compressed=True)
            elif self.method[i] == 'radau-ps':
                t = Radau(num_segments=self.nb_seg[i], order=self.order[i], compressed=True)
            elif self.method[i] == 'runge-kutta':
                t = RungeKutta(num_segments=self.nb_seg[i], order=self.order[i], compressed=True)
            else:
                raise ValueError('method must be either gauss-lobatto, radau-ps or runge-kutta')
            self.transcription.append(t)

        # Phase objects list
        self.phase = []
        self.phase_name = []

        for i in range(len(self.nb_seg)):
            ph = self.trajectory.add_phase(ph_name[i], Phase(ode_class=ode_class[i], ode_init_kwargs=ode_kwargs[i],
                                                             transcription=self.transcription[i]))
            self.phase.append(ph)
            self.phase_name.append(''.join(['traj.', ph_name[i]]))

    def set_time_phase(self, ti, tof, phase, phase_name):
        """Compute the time grid within one phase to retrieve the time instants corresponding to the state, control and
        all discretization nodes.

        Parameters
        ----------
        ti : float
            Initial time [-]
        tof : float
            Time of flight (TOF) [-]
        phase : Phase
            Current phase
        phase_name : str
            Current phase name

        Returns
        -------
        state_nodes : ndarray
            Indexes corresponding to the state discretization nodes
        control_nodes : ndarray
            Indexes corresponding to the control discretization nodes
        t_state : ndarray
            Time instants on the state discretization nodes [-]
        t_control : ndarray
            Time instants on the control discretization nodes [-]
        t_all : ndarray
            Time instants on all discretization nodes [-]

        """

        # set the initial time and the time of flight initial guesses
        self.p[phase_name + '.t_initial'] = ti
        self.p[phase_name + '.t_duration'] = tof

        self.p.run_model()  # run the model to compute the time grid

        # states and controls nodes indices
        sn = phase.options['transcription'].grid_data.subset_node_indices['state_input']
        cn = phase.options['transcription'].grid_data.subset_node_indices['control_input']

        state_nodes = np.reshape(sn, (len(sn), 1))
        control_nodes = np.reshape(cn, (len(cn), 1))

        # time vectors on all, states and controls nodes
        t_all = self.p[phase_name + '.time']

        t_all = np.reshape(t_all, (len(t_all), 1))
        t_control = np.take(t_all, control_nodes)
        t_state = np.take(t_all, state_nodes)

        return state_nodes, control_nodes, t_state, t_control, t_all


if __name__ == '__main__':

    from rpfm.utils.primary import Moon
    from rpfm.utils.spacecraft import Spacecraft

    moon = Moon()
    nlp = NLP(moon, Spacecraft(450., 2.1, g=moon.g), 'gauss-lobatto', 100, 3, 'IPOPT')

    print(nlp)
