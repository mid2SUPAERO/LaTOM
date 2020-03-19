"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.reader.reader import Reader
from latom.plots.solutions import TwoDimMultiPhaseSolPlot
from latom.utils.keplerian_orbit import TwoDimOrb
from latom.utils.const import states_2d


class TwoDim3PhasesLLO2HEOReader(Reader):

    def __init__(self, body, db, case_id='final', db_exp=None, scaled=False):

        Reader.__init__(self, db, case_id=case_id, db_exp=db_exp)

        self.body = body

        if scaled:
            self.gm_res = 1.0
            self.rm_res = 1.0
            self.states_scalers = None
        else:
            self.gm_res = body.GM
            self.rm_res = body.R
            self.states_scalers = np.array([body.R, 1.0, body.vc, body.vc, 1.0])

        self.phase_name = []

        for s in ['dep', 'coast', 'arr']:
            ph_name = '.'.join(['traj', s, 'timeseries'])
            self.phase_name.append(ph_name)

        self.tof, self.time, self.states, self.controls = self.get_time_series(self.case)

        if db_exp is not None:
            self.tof_exp, self.time_exp, self.states_exp, self.controls_exp = self.get_time_series(self.case_exp)
        else:
            self.tof_exp = self.time_exp = self.states_exp = self.controls_exp = None

        self.coe_inj = TwoDimOrb.polar2coe(self.gm_res, self.states[-1][-1, 0], self.states[-1][-1, 2],
                                           self.states[-1][-1, 3])

    def get_time_series(self, case):

        time = []
        tof = []
        states = []
        controls = []

        for i in range(3):

            t = case.outputs.get(self.phase_name[i] + '.time')

            s = np.empty((np.size(t), 0))

            for k in states_2d:
                sk = case.outputs.get(self.phase_name[i] + '.states:' + k)
                s = np.append(s, sk, axis=1)

            alpha = case.outputs.get(self.phase_name[i] + '.controls:alpha')
            thrust = case.outputs.get(self.phase_name[i] + '.design_parameters:thrust')

            if self.states_scalers is not None:
                t = t*self.body.tc
                s = s*self.states_scalers
                thrust = thrust*self.body.g*s[0, -1]

            c = np.hstack((thrust, alpha))

            time.append(t)
            tof.append(t[-1] - t[0])
            states.append(s)
            controls.append(c)

        return tof, time, states, controls

    def plot(self):

        dtheta = self.coe_inj[-1] - self.states[-1][-1, 1]

        sol_plot = TwoDimMultiPhaseSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp,
                                           self.states_exp, a=self.coe_inj[0], e=self.coe_inj[1], dtheta=dtheta)
        sol_plot.plot()
