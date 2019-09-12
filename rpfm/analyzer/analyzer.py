"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from time import time


class Analyzer:

    def __init__(self, body, sc):

        self.body = body
        self.sc = sc

        self.nlp = None

        self.tof = None
        self.time = None
        self.states = None
        self.controls = None

        self.tof_exp = None
        self.time_exp = None
        self.states_exp = None
        self.controls_exp = None

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
