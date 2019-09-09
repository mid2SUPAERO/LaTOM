"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from openmdao.api import Problem, Group, SqliteRecorder, DirectSolver, pyOptSparseDriver
from dymos import Phase, Trajectory, GaussLobatto, Radau

from rpfm.utils.const import rec_excludes
from rpfm.utils.primary import Moon
from rpfm.utils.const import g0


class NLP:

    def __init__(self, sc, method, nb_seg, order, solver, snopt_opts=None, rec_file=None):
        """Initializes NLP class. """

        self.moon = Moon()  # Moon object

        # input parameters
        self.sc = sc
        self.method = method
        self.nb_seg = nb_seg
        self.order = order
        self.solver = solver

        if self.solver == 'SNOPT':
            self.snopt_opts = snopt_opts
        else:
            self.snopt_opts = None

        # non dimensional specific impulse for the Spacecraft object
        self.Isp_adim = self.sc.Isp/self.moon.tc
        self.ode_kwargs = {'Isp': self.Isp_adim, 'g0': g0}

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

        self.p.model.options['assembled_jac_type'] = 'csc'
        self.p.model.linear_solver = DirectSolver()
        self.p.setup(check=True, force_alloc_complex=True, derivatives=True)

    def exp_sim(self, rec_file=None):

        if rec_file is not None:
            self.p_exp = self.trajectory.simulate(atol=1e-30, rtol=1e-30, record_file=rec_file)
        else:
            self.p_exp = self.trajectory.simulate(atol=1e-30, rtol=1e-30)

        self.cleanup()

    def cleanup(self):

        self.trajectory.cleanup()
        self.p.driver.cleanup()
        self.p.cleanup()


class SinglePhaseNLP(NLP):

    def __init__(self, sc, method, nb_seg, order, solver, ode_class, ph_name, snopt_opts=None,
                 rec_file=None):

        NLP.__init__(self, sc, method, nb_seg, order, solver, snopt_opts=snopt_opts, rec_file=rec_file)

        # Transcription object
        if self.method == 'gauss-lobatto':
            self.transcription = GaussLobatto(num_segments=self.nb_seg, order=self.order, compressed=True)
        elif self.method == 'radau-ps':
            self.transcription = Radau(num_segments=self.nb_seg, order=self.order, compressed=True)
        else:
            raise ValueError('method must be either gauss-lobatto or radau-ps')

        # Phase object
        self.phase = self.trajectory.add_phase(ph_name, Phase(ode_class=ode_class, ode_init_kwargs=self.ode_kwargs,
                                                              transcription=self.transcription))
        self.phase_name = ''.join(['traj.', ph_name])

        # discretization nodes
        self.state_nodes = None
        self.control_nodes = None
        self.t_state = None
        self.t_control = None
        self.idx_state_control = None

    def set_objective(self):

        m_ref0 = (self.sc.m0 + self.sc.m_dry)*0.5
        self.phase.add_objective('m', loc='final', ref0=m_ref0, ref=self.sc.m_dry)

    def set_time_options(self, t_bounds):

        self.phase.set_time_options(fix_initial=True, duration_ref0=np.mean(t_bounds),
                                    duration_ref=t_bounds[1], duration_bounds=t_bounds)

    def set_time_guess(self, tof):

        # set initial and transfer time
        self.p[self.phase_name + '.t_initial'] = 0.0
        self.p[self.phase_name + '.t_duration'] = tof

        self.p.run_model()  # compute time grid

        # states and controls nodes
        self.state_nodes = self.phase.options['transcription'].grid_data.subset_node_indices['state_input']
        self.control_nodes = self.phase.options['transcription'].grid_data.subset_node_indices['control_input']

        # time on the discretization nodes
        t_all = self.p[self.phase_name + '.time']
        self.t_control = np.take(t_all, self.control_nodes)
        self.t_state = np.take(t_all, self.state_nodes)

        # indices of the states time vector elements in the controls time vector
        self.idx_state_control = np.nonzero(np.isin(self.t_control, self.t_state))[0]


class MultiPhaseNLP(NLP):

    def __init__(self, sc, method, nb_seg, order, solver, ode_class, ph_name, snopt_opts=None, rec_file=None):

        if isinstance(order, int):
            order = tuple(order for _ in range(len(nb_seg)))

        NLP.__init__(self, sc, method, nb_seg, order, solver, snopt_opts=snopt_opts, rec_file=rec_file)

        # Transcription objects list
        self.transcription = []

        for i in range(len(self.nb_seg)):
            if self.method == 'gauss-lobatto':
                t = GaussLobatto(num_segments=self.nb_seg[i], order=self.order[i], compressed=True)
            elif self.method == 'radau-ps':
                t = Radau(num_segments=self.nb_seg[i], order=self.order[i], compressed=True)
            else:
                raise ValueError('method must be either gauss-lobatto or radau-ps')
            self.transcription.append(t)

        # Phase objects list
        self.phase = []
        self.phase_name = []

        for i in range(len(self.nb_seg)):
            ph = self.trajectory.add_phase(ph_name[i], Phase(ode_class=ode_class[i], ode_init_kwargs=self.ode_kwargs,
                                                             transcription=self.transcription[i]))
            self.phase.append(ph)
            self.phase_name.append(''.join(['traj.', ph_name[i]]))
