# -*- coding: utf-8 -*-
"""
The script defines the classes required to solve the optimal control problem for finding
the most fuel-efficient three-dimensional ascent trajectory from the Moon surface to a specified LLO

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

from __future__ import print_function, division, absolute_import

#adjust path
import sys

if '..' not in sys.path:
    sys.path.append('..')

#import required modules
import numpy as np
#from six import iteritems

from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
from dymos import Phase, Trajectory, GaussLobatto, Radau

from Optimizers.ssto_optimizer import sstoOptimizer
from ODEs.ssto_ode3D import sstoODE_3D_group
from ODEs.ssto_ode3D_coe import sstoODE_3D_group_coe

try:
    from openmdao.api import pyOptSparseDriver
except:
    print('\npyOptSparse module not found!\n')
      
    
class sstoOptimizer_3D(sstoOptimizer):
    
    def get_ODE_class_coe(self, Isp):
        
        """
        returns the ODE class for the current Phase
        """
        
        eps = np.finfo(float).eps #smallest number such that 1.0 + eps != 1.0
        params = {'eps':eps, 'mu':self.const['mu'], 'g0':self.const['g0'], 'Isp':Isp}
        
        return sstoODE_3D_group_coe, params
    
    def get_ODE_class_he(self, Isp):
        
        """
        returns the ODE class for the current Phase
        """
        
        params = {'mu':self.const['mu'], 'g0':self.const['g0'], 'Isp':Isp}
        
        return sstoODE_3D_group, params
                
    def set_state_options(self, phase, t0):
        
        """
        returns a Phase class instance with the state options set
        """
        
        #impose fixed initial time and optional duration bounds
        if self.settings['duration_bounds']:
            phase.set_time_options(fix_initial=True, duration_scaler=self.s[0], duration_bounds=(t0[0], t0[2]))
        else:
            phase.set_time_options(fix_initial=True, duration_scaler=self.s[0])
        
        #impose ICs and bounds on the state variables
        phase.set_state_options('x', fix_initial=True, scaler=self.s[1], defect_scaler=self.ds[1])
        phase.set_state_options('y', fix_initial=True, scaler=self.s[1], defect_scaler=self.ds[1])
        phase.set_state_options('z', fix_initial=True, scaler=self.s[1], defect_scaler=self.ds[1])

        phase.set_state_options('vx', fix_initial=True, scaler=self.s[2], defect_scaler=self.ds[2])   
        phase.set_state_options('vy', fix_initial=True, scaler=self.s[2], defect_scaler=self.ds[2])   
        phase.set_state_options('vz', fix_initial=True, scaler=self.s[2], defect_scaler=self.ds[2])   
        
        m0 = self.const['m0']
        phase.set_state_options('m', fix_initial=True, scaler=self.s[3], defect_scaler=self.ds[3],  lower=m0/100, upper=m0)
                
        return phase
        
    def set_bcs_coe(self, phase, coe):
            
        """
        returns a Phase class instance with final BCs imposed on COE
        """
        
        phase.add_boundary_constraint('a', loc='final', shape=(1,), equals=coe[0])
        phase.add_boundary_constraint('e', loc='final', shape=(1,), equals=coe[1])
        phase.add_boundary_constraint('i', loc='final',shape=(1,), equals=coe[2])
        phase.add_boundary_constraint('raan', loc='final', shape=(1,), equals=coe[3])
        phase.add_boundary_constraint('w', loc='final', shape=(1,), equals=coe[4])
        
        return phase
    
    def set_path_constraints(self, phase):
        
        """
        returns a Phase class instance with path constraints imposed on r and u2
        """
       
        phase.add_path_constraint('u2', lower=1.0, upper=1.0)
        phase.add_path_constraint('r', lower=self.const['R'])
        
        return phase
    
    def set_bcs_he(self, phase, H, E):
            
        """
        returns a Phase class instance with final BCs imposed on H and E vectors
        """
        
        phase.add_boundary_constraint('hx', loc='final', shape=(1,), equals=H[0])
        phase.add_boundary_constraint('hy', loc='final', shape=(1,), equals=H[1])
        phase.add_boundary_constraint('hz', loc='final', shape=(1,), equals=H[2])
        phase.add_boundary_constraint('ex', loc='final', shape=(1,), equals=E[0])
        phase.add_boundary_constraint('ey', loc='final', shape=(1,), equals=E[1])
        phase.add_boundary_constraint('ez', loc='final',shape=(1,), equals=E[2])
        
        return phase
        
    def set_controls_params(self, phase, Isp, Tmin, Tmax):
            
        """
        returns a Phase class instance with the controls and the design parameters set
        """
        
        cont = True
        rate_cont = True
        rate2_cont = False
        
        #thrust direction unit vector
        phase.add_control('ux', continuity=cont, rate_continuity=rate_cont, rate2_continuity=rate2_cont, lower=-1.0, upper=1.0)
        phase.add_control('uy', continuity=cont, rate_continuity=rate_cont, rate2_continuity=rate2_cont, lower=-1.0, upper=1.0)       
        phase.add_control('uz', continuity=cont, rate_continuity=rate_cont, rate2_continuity=rate2_cont, lower=-1.0, upper=1.0)
        
        #thrust magnitude
        phase.add_control('T', continuity=False, rate_continuity=False, rate2_continuity=False, lower=Tmin, upper=Tmax)

        #specific impulse (constant)
        phase.add_design_parameter('Isp', opt=False, val=Isp)
        
        return phase
        
    def set_initial_guess(self, p, phase, Isp, u0, Tmin, Tmax, t0, R0, Rf, Vf):
            
        """
        set the initial guess as a linear interpolation of the BCs
        """
        
        ph_name = phase.pathname
        m0 = self.const['m0']
        
        p[ph_name + '.t_initial'] = 0.0
        p[ph_name + '.t_duration'] = t0[1]
        
        p[ph_name + '.states:x'] = phase.interpolate(ys=[R0[0], Rf[0]], nodes='state_input')
        p[ph_name + '.states:y'] = phase.interpolate(ys=[R0[1], Rf[1]], nodes='state_input')
        p[ph_name + '.states:z'] = phase.interpolate(ys=[R0[2], Rf[2]], nodes='state_input')
        p[ph_name + '.states:vx'] = phase.interpolate(ys=[0.0, Vf[0]], nodes='state_input')
        p[ph_name + '.states:vy'] = phase.interpolate(ys=[0.0, Vf[1]], nodes='state_input')
        p[ph_name + '.states:vz'] = phase.interpolate(ys=[0.0, Vf[2]], nodes='state_input')
        p[ph_name + '.states:m'] = phase.interpolate(ys=[m0, m0/100], nodes='state_input')
        
        #adjust the norm of u such that is equal to one in all the control nodes
        control_nodes = phase.options['transcription'].grid_data.subset_node_indices['control_input']
        n = np.size(control_nodes)
        u_raw = np.linspace(u0[0], u0[1], n) #u0 = [[ux0, uy0, uz0], [uxf, uyf, uzf]]
        u_norm = np.linalg.norm(u_raw, 2, axis=1, keepdims=True)
        u = u_raw/u_norm
        
        p[ph_name + '.controls:ux'] = np.reshape(u[:,0], (n,1))
        p[ph_name + '.controls:ux'] = np.reshape(u[:,1], (n,1))
        p[ph_name + '.controls:ux'] = np.reshape(u[:,2], (n,1))
        
        p[ph_name + '.controls:T'] = phase.interpolate(ys=[Tmax, Tmin], nodes='control_input')
              
        return p
        
    def run_optimizer_coe(self, Isp, Tmin, Tmax, u0, t0, R0, Rf, Vf, coe):
            
        #build an openMDAO Problem class instance and define the optimizer driver and settings
        p = self.set_pbm_driver()

        #build a Dymos Transcription and Phase instances
        t = self.get_transcription(self.settings['num_seg'], self.settings['transcription_order'])
        ode_class, params = self.get_ODE_class_coe(Isp)
        phase = Phase(transcription=t, ode_class=ode_class, ode_init_kwargs=params)
        
        #set the Dymos phase as the Group subsystem
        p.model.add_subsystem('phase0', phase)
        p.model.options['assembled_jac_type'] = self.settings['top_level_jacobian']
        p.model.linear_solver = DirectSolver(assemble_jac=True)
        
        #set the states options
        phase = self.set_state_options(phase, t0)

        #impose BCs on state variables
        phase = self.set_bcs_coe(phase, coe)
        
        #impose path constraints on r and u
        phase = self.set_path_constraints(phase)
        
        #set the controls and the design parameters
        phase = self.set_controls_params(phase, Isp, Tmin, Tmax)
        
        #objective: maximize the final mass
        phase.add_objective('m', loc='final', scaler=-self.s[3])

        #set up the problem
        p.setup(check=True, force_alloc_complex=True, derivatives=True)

        #set the initial guess
        p = self.set_initial_guess(p, phase, Isp, u0, Tmin, Tmax, t0, R0, Rf, Vf)

        #run the model
        p.run_model()
        
        #check the partial derivatives defined in the ODEs
        if self.settings['debug']:
            p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)
     
        #run the driver (returns True if the optimization fails)
        failed = p.run_driver()
        #failed = False 
        
        if not failed:
            self.p = p
            self.phase = phase
            
        return failed
    
    def run_optimizer_he(self, Isp, Tmin, Tmax, u0, t0, R0, Rf, Vf, Hf, Ef):
            
        #build an openMDAO Problem class instance and define the optimizer driver and settings
        p = self.set_pbm_driver()

        #build a Dymos Transcription and Phase instances
        t = self.get_transcription(self.settings['num_seg'], self.settings['transcription_order'])
        ode_class, params = self.get_ODE_class_he(Isp)
        phase = Phase(transcription=t, ode_class=ode_class, ode_init_kwargs=params)
        
        #set the Dymos phase as the Group subsystem
        p.model.add_subsystem('phase0', phase)
        p.model.options['assembled_jac_type'] = self.settings['top_level_jacobian']
        p.model.linear_solver = DirectSolver(assemble_jac=True)
        
        #set the states options
        phase = self.set_state_options(phase, t0)

        #impose BCs on state variables
        phase = self.set_bcs_he(phase, Hf, Ef)
        
        #impose path constraints on r and u
        phase = self.set_path_constraints(phase)
        
        #set the controls and the design parameters
        phase = self.set_controls_params(phase, Isp, Tmin, Tmax)
        
        #objective: maximize the final mass
        phase.add_objective('m', loc='final', scaler=-self.s[3])

        #set up the problem
        p.setup(check=True, force_alloc_complex=True, derivatives=True)

        #set the initial guess
        p = self.set_initial_guess(p, phase, Isp, u0, Tmin, Tmax, t0, R0, Rf, Vf)

        #run the model
        p.run_model()
        
        #check the partial derivatives defined in the ODEs
        if self.settings['debug']:
            p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)
     
        #run the driver (returns True if the optimization fails)
        failed = p.run_driver()
        #failed = False 
        
        if not failed:
            self.p = p
            self.phase = phase
            
        return failed