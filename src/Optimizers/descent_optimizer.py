# -*- coding: utf-8 -*-
"""
The script defines the classes required to solve the optimal control problem
for the most fuel-efficient single-phase or two-phases powered descent trajectory
with an optional final constrained vertical path

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

from __future__ import print_function, division, absolute_import

#adjust path
import sys

if '..' not in sys.path:
    sys.path.append('..')

import numpy as np

from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
from dymos import Phase, Trajectory, GaussLobatto, Radau

from ODEs.ssto_ode import sstoODE, sstoODEvertical, sstoODEthrottle
from Optimizers.ssto_optimizer import sstoOptimizer, sstoOptimizer2p


class descentOptimizer(sstoOptimizer): #single pahse descent trajectory with constant thrust
    
    def set_state_options(self, phase, bcs, t0):
        
        """
        returns a Phase class instance with the state options set
        """
        
        phase.set_time_options(fix_initial=True, duration_scaler=self.s[0])
        phase.set_state_options('r', fix_initial=True, scaler=self.s[1], defect_scaler=self.ds[1], lower=bcs[0,1])
        phase.set_state_options('theta', fix_initial=True, scaler=self.s[2], defect_scaler=self.ds[2], lower=0.0)
        phase.set_state_options('u', fix_initial=True, scaler=self.s[3], defect_scaler=self.ds[3])
        phase.set_state_options('v', fix_initial=True, scaler=self.s[4], defect_scaler=self.ds[4], lower=0.0)
        phase.set_state_options('m', fix_initial=True, scaler=self.s[5], defect_scaler=self.ds[5],
                                lower=bcs[4,1], upper=bcs[4,0])
        
        return phase
        
    def set_initial_guess(self, p, phase, F, w, bcs, t0):
        
        """
        returns a Problem class instance with the initial guess set as a linear interpolation of the BCs
        """
        
        ph_name = phase.pathname
        
        p[ph_name + '.t_initial'] = 0.0
        p[ph_name + '.t_duration'] = t0

        p[ph_name + '.states:r'] = phase.interpolate(ys=bcs[0], nodes='state_input')
        p[ph_name + '.states:theta'] = phase.interpolate(ys=bcs[1], nodes='state_input')
        p[ph_name + '.states:u'] = phase.interpolate(ys=bcs[2], nodes='state_input')
        p[ph_name + '.states:v'] = phase.interpolate(ys=bcs[3], nodes='state_input')
        p[ph_name + '.states:m'] = phase.interpolate(ys=bcs[4], nodes='state_input')
        p[ph_name + '.controls:alpha'] = phase.interpolate(ys=[np.pi, np.pi*2/3], nodes='control_input')
        
        return p
                
    
class descentOptimizer2p(sstoOptimizer2p): #two phases descent trajectory with constant thrust and constrained vertical landing
    
    def set_horiz_state_options(self, phase, bcs):
        
        """
        set the state variables options for the attitude-free phase
        """
        
        phase.set_time_options(fix_initial=True, duration_scaler=self.s[0])
        phase.set_state_options('r', fix_initial=True, scaler=self.s[1], defect_scaler=self.ds[1], lower=bcs[0,1])
        phase.set_state_options('theta', fix_initial=True, scaler=self.s[2], defect_scaler=self.ds[2], lower=0.0)
        phase.set_state_options('u', fix_initial=True, scaler=self.s[3], defect_scaler=self.ds[3])
        phase.set_state_options('v', fix_initial=True, scaler=self.s[4], defect_scaler=self.ds[4], lower=0.0)
        phase.set_state_options('m', fix_initial=True, scaler=self.s[5], defect_scaler=self.ds[5],
                                lower=bcs[4,0]/100, upper=bcs[4,0])
        
        return phase
    
    def set_horiz_constraints(self, phase, bcs):
        
        """
        set constraints for the attitude-free phase
        """
        
        phase.add_boundary_constraint('v', loc='final', equals=0.0)
        
        return phase
        
    def set_vert_state_options(self, phase, bcs):
        
        """
        set the state variables options for the final vertical landing
        """
        
        phase.set_time_options(duration_scaler=self.s[0])
        phase.set_state_options('r', scaler=self.s[1], defect_scaler=self.ds[1], lower=bcs[0,2])
        phase.set_state_options('u', scaler=self.s[3], defect_scaler=self.ds[3])
        phase.set_state_options('m', scaler=self.s[5], defect_scaler=self.ds[5], lower=bcs[4,0]/100, upper=bcs[4,0])
        
        return phase
    
    def set_vert_constraints(self, phase, bcs):
        
        """
        set constraints for the final vertical landing
        """
        
        phase.add_boundary_constraint('r', loc='final', equals=bcs[0,2])
        phase.add_boundary_constraint('u', loc='final', equals=bcs[3,2])
        
        return phase
    
    def set_horiz_initial_guess(self, p, phase, bcs, t0):
        
        """
        returns a Problem class instance with the initial guess set as a linear interpolation of the BCs
        for the attitude-free phase
        """
        
        ph_name = 'trj.horiz'
        
        p.set_val(ph_name + '.t_initial', 0.0)
        p.set_val(ph_name + '.t_duration', t0[0])

        p.set_val(ph_name + '.states:r', phase.interpolate(ys=bcs[0,:2], nodes='state_input'))
        p.set_val(ph_name + '.states:theta', phase.interpolate(ys=bcs[1,:2], nodes='state_input'))
        p.set_val(ph_name + '.states:u', phase.interpolate(ys=bcs[2,:2], nodes='state_input'))
        p.set_val(ph_name + '.states:v', phase.interpolate(ys=bcs[3,:2], nodes='state_input'))
        p.set_val(ph_name + '.states:m', phase.interpolate(ys=bcs[4,:2], nodes='state_input'))
        p.set_val(ph_name + '.controls:alpha', phase.interpolate(ys=[2*np.pi, 5/6*np.pi], nodes='control_input'))
        
        return p
    
    def set_vert_initial_guess(self, p, phase, bcs, t0):
        
        """
        returns a Problem class instance with the initial guess set as a linear interpolation of the BCs
        for the final vertical landing
        """
        
        ph_name = 'trj.vert'
        
        p.set_val(ph_name + '.t_initial', t0[0])
        p.set_val(ph_name + '.t_duration', t0[1]-t0[0])

        p.set_val(ph_name + '.states:r', phase.interpolate(ys=bcs[0,1:], nodes='state_input'))
        p.set_val(ph_name + '.states:u', phase.interpolate(ys=bcs[2,1:], nodes='state_input'))
        p.set_val(ph_name + '.states:m', phase.interpolate(ys=bcs[4,1:], nodes='state_input'))
        
        return p
    
    def run_optimizer(self, F, w, bcs, t0):
                
        #OpenMDAO Problem class instance
        p = self.set_pbm_driver()
        
        #add trajectory
        trj = p.model.add_subsystem('trj', Trajectory())
        
        #horizontal powered braking
        th = self.get_transcription(self.settings['num_seg_horiz'], self.settings['transcription_order_horiz'])
        ode_class_horiz, params_horiz = self.get_ODE_class(F, w)
        horiz = Phase(transcription=th, ode_class=ode_class_horiz, ode_init_kwargs=params_horiz)
        horiz = trj.add_phase('horiz', horiz)
        horiz = self.set_horiz_state_options(horiz, bcs)
        horiz = self.set_horiz_constraints(horiz, bcs)
        horiz = self.set_horiz_controls(horiz, bcs)

        #vertical powered braking
        tv = self.get_transcription(self.settings['num_seg_vert'], self.settings['transcription_order_vert'])
        ode_class_vert, params_vert = self.get_vert_ODE_class(F, w)
        vert = Phase(transcription=tv, ode_class=ode_class_vert, ode_init_kwargs=params_vert)
        vert = trj.add_phase('vert', vert)
        vert = self.set_vert_state_options(vert, bcs)
        vert = self.set_vert_constraints(vert, bcs)

        #add design parameters
        trj = self.set_params(trj, F, w)
        
        #link the two phases
        trj.link_phases(phases=['horiz', 'vert'], vars=['r', 'u', 'm', 'time'], locs=('+-','-+'))
        
        #add the objective: maximize the final mass
        vert.add_objective('m', loc='final', scaler=-self.s[5])

        #set up the problem
        p.model.options['assembled_jac_type'] = self.settings['top_level_jacobian']
        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.setup(check=True, force_alloc_complex=True, derivatives=True)

        #provide an initial guess for all the states and control variables
        p = self.set_horiz_initial_guess(p, horiz, bcs, t0)
        p = self.set_vert_initial_guess(p, vert, bcs, t0)
        
        #run the model
        p.run_model()
        
        #check the partial derivatives defined in sstoODE and sstoODEvertical
        if self.settings['debug']:
            p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)
        
        failed = p.run_driver() #run the driver (return True if the optimization fails)
                                
        if not failed:
            self.p = p
            self.trj = trj
                            
        return failed
    
    
class descentOptimizerThrottle(descentOptimizer): #single pahse descent trajectory with varying thrust
    
    def get_ODE_class(self, F, w):
        
        """
        returns the ODE class for the current Phase
        """
        
        params = {'mu': self.const['mu'], 'F': F, 'w': w}
        
        return sstoODEthrottle, params
    
    def set_initial_guess(self, p, phase, F, w, bcs, t0):
        
        """
        returns a Problem class instance with the initial guess from a linear interpolation of the BCs
        """
        
        ph_name = phase.pathname
        
        p[ph_name + '.t_initial'] = 0.0
        p[ph_name + '.t_duration'] = t0

        p[ph_name + '.states:r'] = phase.interpolate(ys=bcs[0], nodes='state_input')
        p[ph_name + '.states:theta'] = phase.interpolate(ys=bcs[1], nodes='state_input')
        p[ph_name + '.states:u'] = phase.interpolate(ys=bcs[2], nodes='state_input')
        p[ph_name + '.states:v'] = phase.interpolate(ys=bcs[3], nodes='state_input')
        p[ph_name + '.states:m'] = phase.interpolate(ys=bcs[4], nodes='state_input')
        p[ph_name + '.controls:alpha'] = phase.interpolate(ys=[0., 0.], nodes='control_input')
        p[ph_name + '.controls:k'] = phase.interpolate(ys=[0., 1.], nodes='control_input')
        
        return p
    
    def set_controls_params(self, phase, bcs, F, w):
        
        """
        returns a Phase class instance with the controls and the design parameters set
        """
        
        phase.add_control('alpha', continuity=True, rate_continuity=False, rate2_continuity=False,
                          lower=bcs[5,0], upper=bcs[5,1]) #pi/2, 3/2*pi
        phase.add_control('k', continuity=False, rate_continuity=False, rate2_continuity=False,
                          lower=bcs[6,0], upper=bcs[6,1])

        phase.add_design_parameter('thrust', opt=False, val=F)
        phase.add_design_parameter('w', opt=False, val=w)
        
        return phase
        
    
class descentOptimizerThrottle2p(descentOptimizer2p):
    
    def get_ODE_class(self, F, w):
        
        """
        returns the ODE class for the current Phase
        """
        
        ode_class, params = descentOptimizerThrottle.get_ODE_class(self, F, w)
        
        return ode_class, params
    
    def set_horiz_controls(self, phase, bcs):
        
        """
        set the control alpha for the attitude-free phase
        """
        
        phase.add_control('alpha', continuity=True, rate_continuity=True, rate2_continuity=True,
                          lower=bcs[5,0], upper=bcs[5,1]) #pi/2, 3/2*pi
        phase.add_control('k', continuity=False, rate_continuity=False, rate2_continuity=False,
                          lower=bcs[6,0], upper=bcs[6,1]) #klim
        
        return phase
    
    def set_horiz_initial_guess(self, p, phase, bcs, t0):
        
        """
        returns a Problem class instance with the initial guess set as a linear interpolation of the BCs
        for the attitude-free phase
        """
        
        ph_name = 'trj.horiz'
        
        p.set_val(ph_name + '.t_initial', 0.0)
        p.set_val(ph_name + '.t_duration', t0[0])

        p.set_val(ph_name + '.states:r', phase.interpolate(ys=bcs[0,:2], nodes='state_input'))
        p.set_val(ph_name + '.states:theta', phase.interpolate(ys=bcs[1,:2], nodes='state_input'))
        p.set_val(ph_name + '.states:u', phase.interpolate(ys=bcs[2,:2], nodes='state_input'))
        p.set_val(ph_name + '.states:v', phase.interpolate(ys=bcs[3,:2], nodes='state_input'))
        p.set_val(ph_name + '.states:m', phase.interpolate(ys=bcs[4,:2], nodes='state_input'))
        p.set_val(ph_name + '.controls:alpha', phase.interpolate(ys=[3/2*np.pi, np.pi/2], nodes='control_input'))
        p.set_val(ph_name + '.controls:k', phase.interpolate(ys=[0.0, 1.0], nodes='control_input'))
        
        return p