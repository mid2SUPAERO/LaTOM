# -*- coding: utf-8 -*-
"""
The script defines the class required to solve the optimal control problem for finding
the most fuel-efficient ascent trajectory from the Moon surface to a specified LLO with
variable thrust and constraint on the minimum safe altitude

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
from six import iteritems

from ODEs.ssto_ode_constraint import sstoODEconstraint
from Optimizers.ssto_optimizer import sstoOptimizerThrottle


class sstoOptConstraint(sstoOptimizerThrottle):
        
    def set_initial_guess(self, p, phase, F, w, bcs, t0):
        
        """
        returns a Problem class instance with the initial guess from one of the followings:
            unconstrained trajectory
            ideal trajectory (sstoGuess)
            linear interpolation of the BCs
        """
        
        if self.settings['run_unconstrained']: #solve the problem without constraint and set the solution as initial guess
        
            settings_unconstrained = self.settings.copy()
            settings_unconstrained['solver'] = settings_unconstrained['solver_unconstrained']
            settings_unconstrained.pop('solver_unconstrained')
            settings_unconstrained.pop('run_unconstrained')
        
            print('\nSolving the unconstrained optimal trajectory')
        
            opt_unconstrained = sstoOptimizerThrottle(self.const, settings_unconstrained)
            failed = opt_unconstrained.run_optimizer(F, w, bcs, t0)
        
            if not failed:
                ph_unconstrained = opt_unconstrained.get_phase()
                op_dict = dict([(name, options) for (name, options) in ph_unconstrained.list_outputs(units=True, out_stream=None)])
                ph_path = ph_unconstrained.pathname + '.' if ph_unconstrained.pathname else ''

                #initial state values
                for name in ph_unconstrained.state_options:
                    op = op_dict['{0}indep_states.states:{1}'.format(ph_path, name)]
                    p['{0}states:{1}'.format(ph_path, name)][...] = op['value']

                #control values
                for name, options in iteritems(ph_unconstrained.control_options):
                    op = op_dict['{0}control_group.indep_controls.controls:{1}'.format(ph_path, name)]
                    p['{0}controls:{1}'.format(ph_path, name)][...] = op['value']
                
                #time
                p['{0}t_initial'.format(ph_path)] = 0.0
                td = op_dict['{0}time_extents.t_duration'.format(ph_path)]
                p['{0}t_duration'.format(ph_path)] = td['value'][0]
            
                print('\nUnconstrained trajectory found and set as initial guess!')
                print('Time of flight: ' + str(td['value'][0]) + 's')
                print('Final mass: ' + str(p['{0}states:m'.format(ph_path)][-1][0]) + ' kg\n')
                
            else:
                p = sstoOptimizerThrottle.set_initial_guess(self, p, phase, F, w, bcs, t0)
                print('\nOptimization of the unconstrained phase failed!')
            
        else:
            p = sstoOptimizerThrottle.set_initial_guess(self, p, phase, F, w, bcs, t0)
            print('Initial guess set without solving the unconstrained trajectory!')
        
        return p
    
    def get_ODE_class(self, F, w):
        
        """
        returns the ODE class for the current Phase
        """
        
        params = {'mu': self.const['mu'], 'F': F, 'w': w, 'R':self.const['R'], 'hmin':self.const['hmin'], 'mc':self.const['mc']}
        
        return sstoODEconstraint, params
    
    def set_constraints(self, phase, bcs):
        
        """
        returns a Phase class instance with boundary and path constraints set
        """
        
        phase = sstoOptimizerThrottle.set_constraints(self, phase, bcs)
        
        #impose the difference between the spacecraft and the minimum safe distance from the center of the Moon to be greater than zero
        phase.add_path_constraint('dist_safe', lower=0.) 
        
        return phase