"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp import SinglePhaseNLP
from rpfm.odes.odes_2d import ODE2dVarThrust, ODE2dConstThrust
from rpfm.guess.guess_2d import TwoDimAscGuess
from rpfm.utils.const import g0, states_2d


class TwoDimNLP(SinglePhaseNLP):

    def __init__(self, sc, alt, theta, delta_theta, alpha_bounds, method, nb_seg, order, solver, ode_class, ode_kwargs,
                 ph_name, snopt_opts=None, rec_file=None):

        SinglePhaseNLP.__init__(self, sc, method, nb_seg, order, solver, ode_class, ode_kwargs, ph_name,
                                snopt_opts=snopt_opts, rec_file=rec_file)

        # self.alt_adim = alt/self.moon.R
        self.alt_adim = alt
        self.alpha_bounds = np.asarray(alpha_bounds)

        # r_ref0 = (2 + self.alt_adim)*0.5
        self.r_ref = self.moon.R + self.alt_adim
        self.v_ref = (self.moon.GM/self.r_ref)**0.5

        self.phase.add_state('r', rate_source='rdot', targets='r', fix_initial=True, fix_final=True, lower=self.moon.R,
                             scaler=1e-6, defect_scaler=10)
        self.phase.add_state('theta', rate_source='thetadot', fix_initial=True, fix_final=False, lower=0.0,
                             scaler=1, defect_scaler=1)
        self.phase.add_state('u', rate_source='udot', targets='u', fix_initial=True, fix_final=True, scaler=1e-3,
                             defect_scaler=10)
        self.phase.add_state('v', rate_source='vdot', targets='v', fix_initial=True, fix_final=True, lower=0.0,
                             scaler=1e-3, defect_scaler=10)
        self.phase.add_state('m', rate_source='mdot', targets='m', fix_initial=True, fix_final=False, lower=sc.m_dry,
                             upper=sc.m0, scaler=1, defect_scaler=1)

        self.phase.add_control('alpha', fix_initial=False, fix_final=False, targets='alpha', lower=alpha_bounds[0],
                               upper=alpha_bounds[1], continuity=True, rate_continuity=True, rate2_continuity=False)


class TwoDimAscConstNLP(TwoDimNLP):

    def __init__(self, sc, alt, theta, delta_theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                 ph_name, snopt_opts=None, rec_file=None):

        ode_kwargs = {'thrust': sc.T_max, 'Isp': sc.Isp, 'g0': g0}

        TwoDimNLP.__init__(self, sc, alt, theta, delta_theta, alpha_bounds, method, nb_seg, order, solver,
                           ODE2dConstThrust, ode_kwargs, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.set_time_options(tof, t_bounds)
        self.set_objective()
        self.setup()

        # initial guess from linear interpolation
        self.set_time_guess(self.tof_adim)

        self.p[self.phase_name + '.states:r'] = self.phase.interpolate(ys=(self.moon.R, self.r_ref), nodes='state_input')
        self.p[self.phase_name + '.states:theta'] = self.phase.interpolate(ys=(0.0, theta), nodes='state_input')
        self.p[self.phase_name + '.states:u'] = self.phase.interpolate(ys=(0.0, 0.0), nodes='state_input')
        self.p[self.phase_name + '.states:v'] = self.phase.interpolate(ys=(0.0, self.v_ref), nodes='state_input')
        self.p[self.phase_name + '.states:m'] = self.phase.interpolate(ys=(self.sc.m0, self.sc.m0/2),
                                                                       nodes='state_input')

        self.p[self.phase_name + '.controls:alpha'] = self.phase.interpolate(ys=(alpha_bounds[0], alpha_bounds[0]),
                                                                             nodes='control_input')

        self.p.run_model()
        self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)


class TwoDimAscVarNLP(TwoDimNLP):

    def __init__(self, sc, alt, delta_theta, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None):

        ode_kwargs = {'Isp': sc.Isp, 'g0': g0}

        TwoDimNLP.__init__(self, sc, alt, np.pi, delta_theta, alpha_bounds, method, nb_seg, order, solver,
                           ODE2dVarThrust, ode_kwargs, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.phase.add_control('thrust', fix_initial=False, fix_final=False, targets='thrust', lower=self.sc.T_min,
                               upper=self.sc.T_max, ref0=(self.sc.T_min + self.sc.T_max)*0.5, ref=self.sc.T_max,
                               continuity=False, rate_continuity=False, rate2_continuity=False)

        self.guess = TwoDimAscGuess(self.moon.GM, self.moon.R, alt, sc)

        self.set_time_options(self.guess.tof, t_bounds)
        self.set_objective()
        self.setup()

        # initial guess from TwoDimAscGuess
        self.set_time_guess(self.tof_adim)
        self.guess.compute_trajectory(t=self.t_control*self.moon.tc)

        self.p[self.phase_name + '.states:r'] = np.take(self.guess.r/self.moon.R, self.idx_state_control)
        self.p[self.phase_name + '.states:theta'] = np.take(self.guess.theta, self.idx_state_control)
        self.p[self.phase_name + '.states:u'] = np.take(self.guess.u/self.moon.vc, self.idx_state_control)
        self.p[self.phase_name + '.states:v'] = np.take(self.guess.v/self.moon.vc, self.idx_state_control)
        self.p[self.phase_name + '.states:m'] = np.take(self.guess.m, self.idx_state_control)

        self.p[self.phase_name + '.controls:thrust'] = self.guess.T
        self.p[self.phase_name + '.controls:alpha'] = self.guess.alpha

        self.p.run_model()
        self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)


class TwoDimAscVToffNLP(TwoDimAscVarNLP):

    pass


if __name__ == '__main__':

    from rpfm.utils.spacecraft import Spacecraft

    s = Spacecraft(450, 2.1)

    # tv = TwoDimAscVarNLP(s, 100e3, 0.5, (0.0, np.pi/2), (0.5, 1.5), 'gauss-lobatto', 100, 3, 'IPOPT', 'powered')
    tc = TwoDimAscConstNLP(s, 86.87e3, 10*np.pi/180, 2.0, (0.0, np.pi/2), 500, (0.0, 2.0), 'gauss-lobatto', 30, 3,
                           'IPOPT', 'powered')

    # tc.p.run_driver()
