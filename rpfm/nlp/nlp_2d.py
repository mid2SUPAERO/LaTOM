"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp import SinglePhaseNLP

from rpfm.utils.const import g0
from rpfm.odes.odes_2d import ODE2dVarThrust, ODE2dConstThrust


class TwoDimNLP(SinglePhaseNLP):

    def set_states_options(self, r, theta, sc):

        r_adim = r/self.moon.R
        r_ref0 = (1.0 + r_adim)*0.5

        v_circ = (self.moon.GM/r)**0.5
        v_circ_adim = v_circ/self.moon.vc

        self.phase.add_state('r', rate_source='rdot', targets='r', fix_initial=True, fix_final=False, lower=1.0,
                             ref0=r_ref0, ref=r_adim)
        self.phase.add_state('theta', rate_source='thetadot', fix_initial=True, fix_final=False, lower=0.0,
                             ref0=theta*0.5, ref=theta)
        self.phase.add_state('u', rate_source='udot', targets='u', fix_initial=True, fix_final=True)
        self.phase.add_state('v', rate_source='vdot', targets='v', fix_initial=True, fix_final=True, lower=0.0,
                             ref0=v_circ_adim/2, ref=v_circ_adim)
        self.phase.add_state('m', rate_source='mdot', targets='m', fix_initial=True, fix_final=False, lower=sc.m_dry,
                             upper=sc.m0, ref0=(sc.m0 + sc.m_dry)*0.5, ref=sc.m0)

    def set_alpha_options(self, alpha_bounds):

        self.phase.add_control('alpha', fix_initial=False, fix_final=False, targets='alpha', lower=alpha_bounds[0],
                               upper=alpha_bounds[1], ref0=np.mean(alpha_bounds), ref=alpha_bounds[1], continuity=True,
                               rate_continuity=False, rate2_continuity=False)


class TwoDimAscConstNLP(TwoDimNLP):

    def __int__(self, method, nb_seg, order, solver, sc, r, theta, tof, t_bounds, snopt_opts=None, rec_file=None):

        SinglePhaseNLP.__init__(self, method, nb_seg, order, solver, ODE2dConstThrust, {'Isp': sc.Isp, 'g0': g0},
                                'powered', snopt_opts=snopt_opts, rec_file=rec_file)

        tof_adim = tof*self.moon.tc
        self.set_time_options(tof_adim*t_bounds)

        self.set_states_options(r, theta, sc)

    def set_controls_options(self):

        TwoDimNLP.set_alpha_options(self, (0.0, np.pi))

    def set_initial_guess(self, r, theta, sc):

        pass


class TwoDimAscVarNLP(TwoDimNLP):

    def set_controls_options(self, sc):

        TwoDimNLP.set_alpha_options(self, (0.0, np.pi))

        thrust_min_adim = sc.T_min/self.moon.g
        thrust_max_adim = sc.T_max/self.moon.g

        self.phase.add_control('thrust', fix_initial=False, fix_final=False, targets='thrust', lower=thrust_min_adim,
                               upper=thrust_max_adim, ref0=(thrust_min_adim + thrust_max_adim)*0.5, ref=thrust_max_adim,
                               continuity=False, rate_continuity=False, rate2_continuity=False)

    def set_initial_guess(self, states, controls):

        pass


class TwoDimAscVToffNLP(TwoDimAscVarNLP):

    pass
