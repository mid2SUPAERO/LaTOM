"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from openmdao.api import Problem, Group, SqliteRecorder, DirectSolver, pyOptSparseDriver
from dymos import Phase, Trajectory, GaussLobatto, Radau

from rpfm.utils.const import rec_excludes
from rpfm.utils.primary import Moon


class NLP:

    def __init__(self, method, nb_seg, order, solver, snopt_opts=None, rec_file=None):
        """Initializes NLP class. """

        self.p = Problem(model=Group())
        self.method = method

        if isinstance(order, int) and isinstance(nb_seg, tuple):
            order = tuple(order for _ in range(len(nb_seg)))

        self.nb_seg = nb_seg
        self.order = order
        self.solver = solver

        if self.solver == 'SNOPT':
            self.snopt_opts = snopt_opts
        else:
            self.snopt_opts = None

        if rec_file is not None:
            self.set_recorder(rec_file)
        self.rec_file = rec_file

        # initialization
        self.transcription = []
        self.trajectory = None
        self.phase = []
        self.phase_name = []
        self.p_exp = None

        self.set_driver()
        self.set_transcription()

    def set_recorder(self, rec_file):

        recorder = SqliteRecorder(rec_file)

        self.p.add_recorder(recorder)
        opts = ['record_objectives', 'record_constraints', 'record_desvars']

        for opt in opts:
            self.p.recording_options[opt] = False

        self.p.recording_options['excludes'] = rec_excludes

    def set_driver(self):

        self.p.driver = pyOptSparseDriver()
        self.p.driver.options['optimizer'] = self.solver
        self.p.driver.options['print_results'] = False
        self.p.driver.options['dynamic_derivs_sparsity'] = True

        if self.snopt_opts is not None:
            for k in self.snopt_opts.keys():
                self.p.driver.opt_settings[k] = self.snopt_opts[k]

        self.p.driver.declare_coloring(show_summary=True, show_sparsity=False)

    def set_transcription(self):

        if self.method == 'gauss-lobatto':

            if isinstance(self.nb_seg, int):
                self.transcription = GaussLobatto(num_segments=self.nb_seg, order=self.order, compressed=True)
            else:
                for i in range(len(self.nb_seg)):
                    t = GaussLobatto(num_segments=self.nb_seg[i], order=self.order[i], compressed=True)
                    self.transcription.append(t)

        elif self.method == 'radau-ps':

            if isinstance(self.nb_seg, int):
                self.transcription = Radau(num_segments=self.nb_seg, order=self.order, compressed=True)
            else:
                for i in range(len(self.nb_seg)):
                    t = Radau(num_segments=self.nb_seg[i], order=self.order[i], compressed=True)
                    self.transcription.append(t)

        else:
            raise ValueError("Transcription method must be either 'gauss-lobatto' or 'radau-ps'")

    def set_trajectory(self, ode_class, ode_kwargs, phase_name):

        self.trajectory = self.p.model.add_subsystem('traj', Trajectory())

        if isinstance(self.nb_seg, int):
            self.phase = self.trajectory.add_phase(phase_name, Phase(ode_class=ode_class, ode_init_kwargs=ode_kwargs,
                                                                     transcription=self.transcription))
            self.phase_name = ''.join(['traj.', phase_name])

        else:
            for i in range(len(self.nb_seg)):
                ph = self.trajectory.add_phase(phase_name[i], Phase(ode_class=ode_class[i],
                                                                    ode_init_kwargs=ode_kwargs[i],
                                                                    transcription=self.transcription[i]))
                ph_name = ''.join(['traj.', phase_name[i]])
                self.phase.append(ph)
                self.phase_name.append(ph_name)

    def setup(self):

        self.p.model.options['assembled_jac_type'] = 'csc'
        self.p.model.linear_solver = DirectSolver(assemble_jac=True)
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

    def __str__(self):

        lines = ['\n{:^40s}'.format('NLP characteristics:'),
                 '\n{:<25s}{:<15s}'.format('Solver:', self.solver),
                 '{:<25s}{:<15s}'.format('Transcription method:', self.method),
                 '{:<25s}{:<15s}'.format('Number of segments:', str(self.nb_seg)),
                 '{:<25s}{:<15s}'.format('Transcription order:', str(self.order))]

        s = '\n'.join(lines)

        return s


if __name__ == '__main__':

    nlp = NLP('gauss-lobatto', 100, 3, 'IPOPT')
    print(nlp)
