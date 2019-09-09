"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.nlp.nlp import SinglePhaseNLP
from rpfm.odes.odes_2d import ODE2dVarThrust, ODE2dConstThrust
from rpfm.guess.guess_2d import TwoDimAscGuess
from rpfm.utils.const import g0


class TwoDimNLP(SinglePhaseNLP):

    def __init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ode_class,
                 ode_kwargs, ph_name, snopt_opts=None, rec_file=None):

        SinglePhaseNLP.__init__(self, body, sc, method, nb_seg, order, solver, ode_class, ode_kwargs, ph_name,
                                snopt_opts=snopt_opts, rec_file=rec_file)

        self.alt = alt
        self.alpha_bounds = np.asarray(alpha_bounds)

        self.r_circ = self.body.R + self.alt
        self.v_circ = (self.body.GM/self.r_circ)**0.5

        self.phase.add_state('r', units='m', rate_source='rdot', targets='r', fix_initial=True, fix_final=True,
                             lower=body.R, scaler=np.power(10, -np.floor(np.log10(self.r_circ))), defect_scaler=10)
        self.phase.add_state('theta', units='rad', rate_source='thetadot', fix_initial=True, fix_final=False,
                             lower=0.0)
        self.phase.add_state('u', units='m/s', rate_source='udot', targets='u', fix_initial=True, fix_final=True,
                             scaler=np.power(10, -np.floor(np.log10(self.v_circ))), defect_scaler=10)
        self.phase.add_state('v', units='m/s', rate_source='vdot', targets='v', fix_initial=True, fix_final=True,
                             lower=0.0, scaler=np.power(10, -np.floor(np.log10(self.v_circ))), defect_scaler=10)
        self.phase.add_state('m', units='kg', rate_source='mdot', targets='m', fix_initial=True, fix_final=False,
                             lower=sc.m_dry, upper=sc.m0, scaler=np.power(10, -np.floor(np.log10(sc.m0))))

        self.phase.add_control('alpha', units='rad', targets='alpha', fix_initial=False, fix_final=False,
                               continuity=True, rate_continuity=True, rate2_continuity=False, lower=alpha_bounds[0],
                               upper=alpha_bounds[1])

        self.phase.add_design_parameter('w', units='m/s', opt=False, val=sc.w)


class TwoDimAscConstNLP(TwoDimNLP):

    def __init__(self, body, sc, alt, theta, alpha_bounds, tof, t_bounds, method, nb_seg, order, solver,
                 ph_name, snopt_opts=None, rec_file=None):

        ode_kwargs = {'GM': body.GM, 'T': sc.T_max, 'w': sc.w}

        TwoDimNLP.__init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ODE2dConstThrust,
                           ode_kwargs, ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.phase.add_design_parameter('thrust', units='N', opt=False, val=sc.T_max)

        self.set_time_options(tof, t_bounds)
        self.set_objective()
        self.setup()

        # initial guess from linear interpolation
        self.set_time_guess(self.tof)

        self.p[self.phase_name + '.states:r'] = self.phase.interpolate(ys=(body.R, self.r_circ), nodes='state_input')
        self.p[self.phase_name + '.states:theta'] = self.phase.interpolate(ys=(0.0, theta), nodes='state_input')
        self.p[self.phase_name + '.states:u'] = self.phase.interpolate(ys=(0.0, 0.0), nodes='state_input')
        self.p[self.phase_name + '.states:v'] = self.phase.interpolate(ys=(0.0, self.v_circ), nodes='state_input')
        self.p[self.phase_name + '.states:m'] = self.phase.interpolate(ys=(self.sc.m0, self.sc.m_dry),
                                                                       nodes='state_input')

        self.p[self.phase_name + '.controls:alpha'] = self.phase.interpolate(ys=(0.0, 0.0), nodes='control_input')

        self.p.run_model()
        self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)


class TwoDimAscVarNLP(TwoDimNLP):

    def __init__(self, body, sc, alt, alpha_bounds, t_bounds, method, nb_seg, order, solver, ph_name,
                 snopt_opts=None, rec_file=None):

        ode_kwargs = {'GM': body.GM, 'w': sc.w}

        TwoDimNLP.__init__(self, body, sc, alt, alpha_bounds, method, nb_seg, order, solver, ODE2dVarThrust, ode_kwargs,
                           ph_name, snopt_opts=snopt_opts, rec_file=rec_file)

        self.phase.add_control('thrust', units='N', targets='thrust', fix_initial=False, fix_final=False,
                               continuity=False, rate_continuity=False, rate2_continuity=False, lower=self.sc.T_min,
                               upper=self.sc.T_max, scaler=np.power(10, -np.floor(np.log10(self.sc.T_max))))

        self.guess = TwoDimAscGuess(self.body.GM, self.body.R, alt, sc)

        self.set_time_options(self.guess.tof, t_bounds)
        self.set_objective()
        self.setup()

        # initial guess from TwoDimAscGuess
        self.set_time_guess(self.tof)
        self.guess.compute_trajectory(t=self.t_control)

        self.p[self.phase_name + '.states:r'] = np.take(self.guess.r, self.idx_state_control)
        self.p[self.phase_name + '.states:theta'] = np.take(self.guess.theta, self.idx_state_control)
        self.p[self.phase_name + '.states:u'] = np.take(self.guess.u, self.idx_state_control)
        self.p[self.phase_name + '.states:v'] = np.take(self.guess.v, self.idx_state_control)
        self.p[self.phase_name + '.states:m'] = np.take(self.guess.m, self.idx_state_control)

        self.p[self.phase_name + '.controls:thrust'] = self.guess.T
        self.p[self.phase_name + '.controls:alpha'] = self.guess.alpha

        self.p.run_model()
        self.p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)


class TwoDimAscVToffNLP(TwoDimAscVarNLP):

    pass


if __name__ == '__main__':

    from rpfm.utils.spacecraft import Spacecraft
    from rpfm.utils.primary import Moon

    moon = Moon()
    s = Spacecraft(450.0, 2.1, g=moon.g)

    tv = TwoDimAscVarNLP(moon, s, 86.87e3, (-np.pi/2, np.pi/2), None, 'gauss-lobatto', 150, 3, 'SNOPT', 'powered')
    tc = TwoDimAscConstNLP(moon, s, 86.87e3, np.pi/2, (-np.pi/2, np.pi/2), 500, None, 'gauss-lobatto', 10, 3,
                           'IPOPT', 'powered')

    tv.p.run_driver()
    print(tv.p.get_val('traj.powered.timeseries.time')[-1, -1])
