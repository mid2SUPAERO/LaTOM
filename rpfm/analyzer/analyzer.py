"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from time import time


class Analyzer:

    def __init__(self, body, sc):

        self.body = body
        self.sc = sc
        self.phase_name = None

        self.nlp = None
        self.tof = self.time = self.states = self.controls = None
        self.tof_exp = self.time_exp = self.states_exp = self.controls_exp = None
        self.err = None
        self.rm_res = self.gm_res = None

    def run_driver(self):

        if self.nlp.rec_file is not None:
            self.nlp.p.record_iteration('initial')

        t0 = time()

        failed = self.nlp.p.run_driver()

        tf = time()
        dt = tf - t0

        print('\nTime to solve the NLP problem:', dt, 's\n')

        if self.nlp.rec_file is not None:
            self.nlp.p.record_iteration('final')

        self.nlp.cleanup()

        return failed

    def get_time_series(self, p, scaled=False):

        return None, None, None, None

    def get_solutions(self, explicit=True, scaled=False):

        tof, t, states, controls = self.get_time_series(self.nlp.p, scaled=scaled)

        self.tof = tof
        self.time = t
        self.states = states
        self.controls = controls

        if explicit:

            tof, t, states, controls = self.get_time_series(self.nlp.p_exp, scaled=scaled)

            self.tof_exp = tof
            self.time_exp = t
            self.states_exp = states
            self.controls_exp = controls

            if isinstance(self.phase_name, str):
                self.err = self.states[-1, :] - self.states_exp[-1, :]
            else:
                self.err = np.empty((0, np.size(self.states[0][0, :])))
                for i in range(len(self.phase_name)):
                    err = np.reshape(self.states[i][-1, :] - self.states_exp[i][-1, :],
                                     (1, np.size(self.states[i][-1, :])))
                    self.err = np.append(self.err, err, axis=0)

        if scaled:
            self.rm_res = 1.0
            self.gm_res = 1.0
        else:
            self.rm_res = self.body.R
            self.gm_res = self.body.GM
