# -*- coding: utf-8 -*-
"""
The script defines the classes required to solve the optimal control problem for finding
the most fuel-efficient ascent trajectory from the Moon surface to a specified LLO

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
from time import time

from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
from dymos import Phase, Trajectory, GaussLobatto, Radau
from ODEs.ssto_ode import sstoODE, sstoODEvertical, sstoODEthrottle
from Utils.ssto_guess import sstoGuess

try:
    from openmdao.api import pyOptSparseDriver
except:
    print('\npyOptSparse module not found!\n')


class sstoOptimizer: #single phase with constant thrust
    
    def __init__(self, const, settings):
        
        """
        initialize the Optimizer constants, settings and scaling parameters
        """
        
        self.const = const
        self.settings = settings
        
        self.s = self.settings['scalers'] #scalers for (time, r, theta, u, v, m)
        self.ds = self.settings['defect_scalers'] #defect scalers for (time, r, theta, u, v, m)
        
    def get_pbm(self):
        
        """
        returns the OpenMDAO Problem class instance
        """
        
        return self.p
    
    def get_phase(self):
        
        """
        returns the Dymos Phase class instance
        """
        
        return self.phase
    
    def get_transcription(self, nb_seg, tr_ord):
        
        """
        returns the Transcription class instance for the current Phase
        """
        
        if self.settings['transcription'] == 'gauss-lobatto':
            t = GaussLobatto(num_segments=nb_seg, order=tr_ord, compressed=self.settings['compressed'])
        elif self.settings['transcription'] == 'radau-ps':
            t = Radau(num_segments=nb_seg, order=tr_ord, compressed=self.settings['compressed'])
            
        return t
    
    def get_ODE_class(self, F, w):
        
        """
        returns the ODE class for the current Phase with the required input arguments
        """
        
        params = {'mu':self.const['mu'], 'F':F, 'w':w}
        
        return sstoODE, params
    
    def set_pbm_driver(self):
        
        """
        returns a Problem class instance with the corresponding Driver
        """
        
        p = Problem(model=Group())
        
        try:
            p.driver = pyOptSparseDriver()
            p.driver.options['optimizer'] = self.settings['solver']
            p.driver.options['print_results'] = False
            p.driver.options['dynamic_derivs_sparsity'] = True
            
            #SLSQP specific options
            if self.settings['solver'] == 'SLSQP':
                p.driver.opt_settings['ACC'] = self.settings['tol']
                p.driver.opt_settings['MAXIT'] = self.settings['maxiter']
                p.driver.opt_settings['IPRINT'] = -1
                
            #PSQP specific options
            if self.settings['solver'] == 'PSQP':
                p.driver.opt_settings['IPRINT'] = 0
                
            #SNOPT specific settings
            if self.settings['solver'] == 'SNOPT':
                p.driver.opt_settings['Iterations limit'] = 20000
                p.driver.opt_settings['Major iterations limit'] = 5000
                p.driver.opt_settings['Minor iterations limit'] = 2000
                p.driver.opt_settings['Scale option'] = 1 
                
            print('\nSolver ' + self.settings['solver'] + ' included in pyOptSparse\n')

        except:
            p.driver = ScipyOptimizeDriver()
            p.driver.options['tol'] = self.settings['tol']
            p.driver.options['maxiter'] = self.settings['maxiter']
            p.driver.options['disp'] = self.settings['debug']
            
            print('\nThere was an error while importing ' + self.settings['solver'] + ' from pyOptSparse')
            print('Using SLSQP from Scipy instead\n')
            
        #p.driver.options['dynamic_simul_derivs'] = self.settings['dynamic_simul_derivs'] #deprecated but better than driver.declare_coloring
        p.driver.declare_coloring(show_summary=True, show_sparsity=False)
        
        return p
    
    def set_state_options(self, phase, bcs, t0):
        
        """
        returns a Phase class instance with the state options set
        """
        
        #impose fixed initial time and duration bounds
        if self.settings['duration_bounds']:
            phase.set_time_options(fix_initial=True, duration_scaler=self.s[0], duration_bounds=(t0[0], t0[2]))
        else:
            phase.set_time_options(fix_initial=True, duration_scaler=self.s[0])
        
        #impose ICs and bounds on the state variables
        phase.set_state_options('r', fix_initial=True, scaler=self.s[1], defect_scaler=self.ds[1], lower=bcs[0,0])
        phase.set_state_options('theta', fix_initial=True, scaler=self.s[2], defect_scaler=self.ds[2], lower=0.)
        phase.set_state_options('u', fix_initial=True, scaler=self.s[3], defect_scaler=self.ds[3], lower=0.)
        phase.set_state_options('v', fix_initial=True, scaler=self.s[4], defect_scaler=self.ds[4], lower=0.)
        phase.set_state_options('m', fix_initial=True, scaler=self.s[5], defect_scaler=self.ds[5],
                                lower=bcs[4,1], upper=bcs[4,0])
                
        return phase
    
    def set_constraints(self, phase, bcs):
        
        """
        returns a Phase class instance with final BCs set
        """
        
        phase.add_boundary_constraint('r', loc='final', equals=bcs[0,1])
        phase.add_boundary_constraint('u', loc='final', equals=bcs[2,1])
        phase.add_boundary_constraint('v', loc='final', equals=bcs[3,1])
        
        return phase
    
    def set_controls_params(self, phase, bcs, F, w):
        
        """
        returns a Phase class instance with the controls and the design parameters set
        """
        
        #the only control variable is the thrust direction alpha
        phase.add_control('alpha', continuity=True, rate_continuity=True, rate2_continuity=self.settings['alpha_rate2_cont'],
                          lower=bcs[5,0], upper=bcs[5,1])

        #the design parameters are the thrust magnitude and the exaust velocity
        phase.add_design_parameter('thrust', opt=False, val=F)
        phase.add_design_parameter('w', opt=False, val=w)
        
        return phase
    
    def set_initial_guess(self, p, phase, F, w, bcs, t0):
        
        """
        returns a Problem class instance with the initial guess set as a linear interpolation of the BCs
        """
                
        ph_name = phase.pathname
                
        p[ph_name + '.t_initial'] = 0.0
        p[ph_name + '.t_duration'] = t0[1]
        
        p[ph_name + '.states:r'] = phase.interpolate(ys=bcs[0], nodes='state_input')
        p[ph_name + '.states:theta'] = phase.interpolate(ys=bcs[1], nodes='state_input')
        p[ph_name + '.states:u'] = phase.interpolate(ys=bcs[2], nodes='state_input')
        p[ph_name + '.states:v'] = phase.interpolate(ys=bcs[3], nodes='state_input')
        p[ph_name + '.states:m'] = phase.interpolate(ys=bcs[4], nodes='state_input')
        p[ph_name + '.controls:alpha'] = phase.interpolate(ys=[0., 0.], nodes='control_input')
                
        return p
          
    def run_optimizer(self, F, w, bcs, t0):
        
        #build an openMDAO Problem class instance and define the optimizer driver and settings
        p = self.set_pbm_driver()

        #build a Dymos Transcription and Phase instances
        t = self.get_transcription(self.settings['num_seg'], self.settings['transcription_order'])
        ode_class, params = self.get_ODE_class(F, w)
        phase = Phase(transcription=t, ode_class=ode_class, ode_init_kwargs=params)
        
        #set the Dymos phase as the Group subsystem
        p.model.add_subsystem('phase0', phase)
        p.model.options['assembled_jac_type'] = self.settings['top_level_jacobian']
        p.model.linear_solver = DirectSolver(assemble_jac=True)
        
        #set the states options
        phase = self.set_state_options(phase, bcs, t0)

        #impose constraints on state variables
        phase = self.set_constraints(phase, bcs)
        
        #set the controls and the design parameters
        phase = self.set_controls_params(phase, bcs, F, w)
        
        #objective: maximize the final mass
        phase.add_objective('m', loc='final', scaler=-self.s[5])

        #set up the problem
        p.setup(check=True, force_alloc_complex=True, derivatives=True)

        #set the initial guess
        p = self.set_initial_guess(p, phase, F, w, bcs, t0)

        #run the model
        p.run_model()
                
        #check the partial derivatives defined in the ODEs
        if self.settings['debug']:
            t0 = time()
            p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)
            t1 = time()
            dt = t1 - t0
            print("\nTime for derivative checking: ", dt)
            print("")
                    
        #run the driver (returns True if the optimization fails)
        failed = p.run_driver()
                                
        if not failed:
            self.p = p
            self.phase = phase
            
        return failed
    
class sstoOptimizerThrottle(sstoOptimizer): #single phase with varying thrust
    
    def get_ODE_class(self, F, w):
        
        """
        returns the ODE class for the current Phase
        """
        
        params = {'mu':self.const['mu'], 'F':F, 'w':w}
        
        return sstoODEthrottle, params
    
    def set_controls_params(self, phase, bcs, F, w):
        
        """
        returns a Phase class instance with the controls and the design parameters set
        """
        
        #the contrl variables are the thrust direction alpha and the throttle k
        phase.add_control('k', continuity=False, rate_continuity=False, rate2_continuity=False, lower=bcs[6,0], upper=bcs[6,1])
        
        phase = sstoOptimizer.set_controls_params(self, phase, bcs, F, w)
        
        return phase
    
    def set_initial_guess(self, p, phase, F, w, bcs, t0):
        
        """
        returns a Problem class instance with the initial guess from sstoGuess or from a linear interpolation of the BCs
        """
        
        ph_name = phase.pathname
        
        if self.settings['acc_guess']: #accurate guess (ideal trajectory)
            
            print('\nComputing accurate initial guess')
            
            #build an sstoGuess class instance and determine the time of flight for the initial guess
            init_guess = sstoGuess(self.const, F, w)
            tof, s_t = init_guess.compute_tof()
            
            #set the time of flight of the initial guess as the Phase time of flight
            p[ph_name + '.t_initial'] = 0.0
            p[ph_name + '.t_duration'] = tof
            p.run_model()
            
            #retrieve the time vector for all the nodes
            t_all = p[ph_name + '.time.time']
            
            #determine the states and controls input nodes indices
            state_nodes = phase.options['transcription'].grid_data.subset_node_indices['state_input']
            control_nodes = phase.options['transcription'].grid_data.subset_node_indices['control_input']
            
            #determine the time vectors for the states and controls nodes
            t_control = np.take(t_all, control_nodes)
            t_state = np.take(t_all, state_nodes)
            
            #determine the indices of the states time vector elements in the controls time vector
            idx_state_control = np.nonzero(np.isin(t_control, t_state))[0]
            
            #compute the timeseries of the initial guess in the controls nodes
            s_sg, s_ht = init_guess.compute_trajectory(t=t_control)
            
            print('\nAccurate intial guess found!')
            print('Final mass: ' + str(init_guess.m_final/self.const['m0']*100) + ' %')
            print('Time of flight: ' + str(init_guess.tof) + ' s\n')
            
            #init_guess.plot_all() plot the "ideal" trajectory
            
            #set the computed timeseries as the Phase initial guess
            p[ph_name + '.states:r'] = np.reshape(np.take(init_guess.r, idx_state_control), (len(idx_state_control), 1))
            p[ph_name + '.states:theta'] = np.reshape(np.take(init_guess.theta, idx_state_control), (len(idx_state_control), 1))
            p[ph_name + '.states:u'] = np.reshape(np.take(init_guess.u, idx_state_control), (len(idx_state_control), 1))
            p[ph_name + '.states:v'] = np.reshape(np.take(init_guess.v, idx_state_control), (len(idx_state_control), 1))
            p[ph_name + '.states:m'] = np.reshape(np.take(init_guess.m, idx_state_control), (len(idx_state_control), 1))
        
            p[ph_name + '.controls:alpha'] = np.reshape(init_guess.alpha, (len(control_nodes), 1))        
            p[ph_name + '.controls:k'] = np.reshape(init_guess.k, (len(control_nodes), 1))
                        
        else: #poor guess from linear interpolation
            
            p = sstoOptimizer.set_initial_guess(self, p, phase, F, w, bcs, t0)
            p[ph_name + '.controls:k'] = phase.interpolate(ys=[1., 0.], nodes='control_input')
                    
        return p
    
        
class sstoOptimizer2p(sstoOptimizer): #two phases (vertical rise and free-attitude) with constant thrust
    
    def get_trj(self):
        
        """
        return the Dymos Trajectory class instance
        """
        
        return self.trj
        
    def set_vert_state_options(self, phase, bcs):
        
        """
        set the state variables options for the vertical rising phase
        """
        
        phase.set_time_options(fix_initial=True)
        phase.set_state_options('r', fix_initial=True, scaler=self.s[1], defect_scaler=self.ds[1], lower=bcs[0,0], upper=bcs[0,2])
        phase.set_state_options('u', fix_initial=True, lower=0.)
        phase.set_state_options('m', fix_initial=True, scaler=self.s[5], lower=bcs[4,2], upper=bcs[4,0])
        
        return phase
    
    def set_vert_constraints(self, phase, bcs):
        
        """
        set constraints for the vertical rise phase
        """
        
        if self.settings['fixed']=='time': #vertical phase ends after a fixed amount of time
            phase.set_time_options(fix_duration=True)
        elif self.settings['fixed']=='alt': #vertical phase ends after the spacecraft has reached the specified altitude
            phase.add_boundary_constraint('r', loc='final', equals=bcs[0,1])
            
        return phase
    
    def set_horiz_state_options(self, phase, bcs):
        
        """
        set the state variables options for the attitude-free phase
        """
        
        phase.set_time_options(duration_scaler=self.s[0])
        phase.set_state_options('r', scaler=self.s[1], defect_scaler=self.ds[1], lower=bcs[0,0], upper=bcs[0,2])
        phase.set_state_options('theta', fix_initial=True, scaler=self.s[2], defect_scaler=self.ds[2], lower=0.)
        phase.set_state_options('u', scaler=self.s[3], defect_scaler=self.ds[3], lower=0.)
        phase.set_state_options('v', fix_initial=True, scaler=self.s[4], defect_scaler=self.ds[4], lower=0.)
        phase.set_state_options('m', scaler=self.s[5], defect_scaler=self.ds[5], lower=bcs[4,2], upper=bcs[4,0])
        
        return phase
    
    def set_horiz_constraints(self, phase, bcs):
        
        """
        set final BCs for the attitude-free phase
        """
        
        phase.add_boundary_constraint('r', loc='final', equals=bcs[0,2])
        phase.add_boundary_constraint('u', loc='final', equals=bcs[2,2])
        phase.add_boundary_constraint('v', loc='final', equals=bcs[3,2])
        
        return phase
    
    def set_horiz_controls(self, phase, bcs):
        
        """
        set the control alpha for the attitude-free phase
        """
        
        phase.add_control('alpha', continuity=True, rate_continuity=True, rate2_continuity=True, lower=bcs[5,0], upper=bcs[5,1])
        
        return phase
    
    def set_params(self, trj, F, w):
        
        """
        set the trajectory parameters thrust and exaust velocity
        """
        
        trj.add_design_parameter('thrust', opt=False, val=F, units='N')
        trj.add_design_parameter('w', opt=False, val=w, units='m/s')
        
        return trj
    
    def set_vert_initial_guess(self, p, phase, bcs, t0):
        
        """
        returns a Problem class instance with the initial guess set as a linear interpolation of the BCs
        for the vertical rise phase
        """
                
        ph_name = 'trj.vert'
        
        p.set_val(ph_name + '.t_initial', 0.0)
        p.set_val(ph_name + '.t_duration', t0[1])

        p.set_val(ph_name + '.states:r', phase.interpolate(ys=bcs[0,:2], nodes='state_input'))
        p.set_val(ph_name + '.states:u', phase.interpolate(ys=bcs[2,:2], nodes='state_input'))
        p.set_val(ph_name + '.states:m', phase.interpolate(ys=bcs[4,:2], nodes='state_input'))
        
        return p
    
    def set_horiz_initial_guess(self, p, phase, bcs, t0):
        
        """
        returns a Problem class instance with the initial guess set as a linear interpolation of the BCs
        for the attitude-free phase
        """
                
        ph_name = 'trj.horiz'
        
        p.set_val(ph_name + '.t_initial', t0[0])
        p.set_val(ph_name + '.t_duration', t0[2]-t0[1])

        p.set_val(ph_name + '.states:r', phase.interpolate(ys=bcs[0,1:], nodes='state_input'))
        p.set_val(ph_name + '.states:theta', phase.interpolate(ys=bcs[1,1:], nodes='state_input'))
        p.set_val(ph_name + '.states:u', phase.interpolate(ys=bcs[2,1:], nodes='state_input'))
        p.set_val(ph_name + '.states:v', phase.interpolate(ys=bcs[3,1:], nodes='state_input'))
        p.set_val(ph_name + '.states:m', phase.interpolate(ys=bcs[4,1:], nodes='state_input'))
        p.set_val(ph_name + '.controls:alpha', phase.interpolate(ys=[np.pi/2, 0.], nodes='control_input'))
        
        return p
    
    def get_vert_ODE_class(self, F, w):
                
        """
        returns the ODE class for the vertical rise Phase with the required input arguments
        """
        
        params = {'mu':self.const['mu'], 'F':F, 'w':w}
        
        return sstoODEvertical, params
        
    def run_optimizer(self, F, w, bcs, t0):
        
        #build an openMDAO Problem class instance and define the optimizer driver and settings
        p = self.set_pbm_driver()
        
        #add a trajectory
        trj = p.model.add_subsystem('trj', Trajectory())

        #add vertical phase
        tv = self.get_transcription(self.settings['num_seg_vert'], self.settings['transcription_order_vert'])
        ode_class_vert, params_vert = self.get_vert_ODE_class(F, w)
        vert = Phase(transcription=tv, ode_class=ode_class_vert, ode_init_kwargs=params_vert)
        vert = trj.add_phase('vert', vert)
        vert = self.set_vert_state_options(vert, bcs)
        vert = self.set_vert_constraints(vert, bcs)
        
        #add attitude-free phase
        th = self.get_transcription(self.settings['num_seg_horiz'], self.settings['transcription_order_horiz'])
        ode_class_horiz, params_horiz = self.get_ODE_class(F, w)
        horiz = Phase(transcription=th, ode_class=ode_class_horiz, ode_init_kwargs=params_horiz)
        horiz = trj.add_phase('horiz', horiz)
        horiz = self.set_horiz_state_options(horiz, bcs)
        horiz = self.set_horiz_constraints(horiz, bcs)
        horiz = self.set_horiz_controls(horiz, bcs)
        
        #add design parameters
        trj = self.set_params(trj, F, w)
        
        #link the two phases
        trj.link_phases(phases=['vert', 'horiz'], vars=['r', 'u', 'm', 'time'], locs=('+-', '-+'))
        
        #objective: maximize the final mass
        horiz.add_objective('m', loc='final', scaler=-self.s[5])
        
        #set up the problem
        p.model.options['assembled_jac_type'] = self.settings['top_level_jacobian']
        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.setup(check=True, force_alloc_complex=True, derivatives=True)
        
        #set the initial guess
        p = self.set_vert_initial_guess(p, vert, bcs, t0)
        p = self.set_horiz_initial_guess(p, horiz, bcs, t0)

        #run the model
        p.run_model()
                
        #check the partial derivatives defined in the ODEs
        if self.settings['debug']:
            p.check_partials(method='cs', compact_print=True, show_only_incorrect=True)
                    
        #run the driver (returns True if the optimization fails)
        failed = p.run_driver()
        
        if not failed:
            self.p = p
            self.trj = trj
            
        return failed
