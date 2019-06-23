# -*- coding: utf-8 -*-
"""
The script defines the equations of motion (ODEs) and their partial derivatives required
to describe a two-dimensional ascent or descent trajectory from or to the Moon surface

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent
from dymos import declare_time, declare_state, declare_parameter

#sstoODE: ODEs with constant thrust

@declare_time(units='s')

@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('theta', rate_source='thetadot', units='rad')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')

@declare_parameter('alpha', targets=['alpha'], units='rad')
@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')

class sstoODE(ExplicitComponent):
    
    def initialize(self):
        
        """
        initialize an instance specifying the followings:
            num_nodes:  number of discrete points in which the constraint has to be evaluated
            mu:         lunar gravitational parameter (m^3/s^2)
            F:          thrust (N)
            w:          exaust velocity (m/s)
        """
        
        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', types=float)
        self.options.declare('F', types=float)
        self.options.declare('w', types=float)
        
    def setup(self):
        
        """
        define the equation inputs, outputs and jacobian components
        """
        
        nn = self.options['num_nodes']
        F = self.options['F']
        w = self.options['w']

        #inputs (ODE right-hand side)
        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('v', val=np.zeros(nn), desc='tangential velocity', units='m/s')
        self.add_input('m', val=np.zeros(nn), desc='mass', units='kg')
        
        self.add_input('alpha', val=np.zeros(nn), desc='thrust direction', units='rad')
        self.add_input('thrust', val=F*np.ones(nn), desc='thrust', units='N')
        self.add_input('w', val=w*np.ones(nn), desc='exaust velocity', units='m/s')

        #outputs (ODE left-hand side)
        self.add_output('rdot', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_output('thetadot', val=np.zeros(nn), desc='angular rate', units='rad/s')
        self.add_output('udot', val=np.zeros(nn), desc='radial acceleration', units='m/s**2')
        self.add_output('vdot', val=np.zeros(nn), desc='tangential acceleration', units='m/s**2')
        self.add_output('mdot', val=np.zeros(nn), desc='mass rate', units='kg/s')

        #partial derivatives of the ODE outputs respect to the ODE inputs
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='rdot', wrt='u', rows=ar, cols=ar, val=1)
        
        self.declare_partials(of='thetadot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='thetadot', wrt='v', rows=ar, cols=ar)
        
        self.declare_partials(of='udot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='thrust', rows=ar, cols=ar)
        
        self.declare_partials(of='vdot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='thrust', rows=ar, cols=ar)
        
        self.declare_partials(of='mdot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='mdot', wrt='w', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        """
        compute the equation outputs
        """
        
        mu = self.options['mu']
        
        r = inputs['r']
        u = inputs['u']
        v = inputs['v']
        m = inputs['m']
        alpha = inputs['alpha']
        thrust = inputs['thrust']
        w = inputs['w']
                
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        outputs['rdot'] = u
        outputs['thetadot'] = v/r
        outputs['udot'] = -mu/r**2 + v**2/r + (thrust/m)*sin_alpha
        outputs['vdot'] = -u*v/r + (thrust/m)*cos_alpha
        outputs['mdot'] = -thrust/w
        
    def compute_partials(self, inputs, jacobian):
        
        """
        compute the jacobian components
        """
        
        mu = self.options['mu']
        
        r = inputs['r']
        u = inputs['u']
        v = inputs['v']
        m = inputs['m']
        alpha = inputs['alpha']
        thrust = inputs['thrust']
        w = inputs['w']
        
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        jacobian['thetadot', 'r'] = -v/r**2
        jacobian['thetadot', 'v'] = 1/r

        jacobian['udot', 'r'] = 2*mu/r**3 - (v/r)**2
        jacobian['udot', 'v'] = 2*v/r
        jacobian['udot', 'm'] = -(thrust/m**2)*sin_alpha
        jacobian['udot', 'alpha'] = (thrust/m)*cos_alpha
        jacobian['udot', 'thrust'] = sin_alpha/m

        jacobian['vdot', 'r'] = u*v/r**2
        jacobian['vdot', 'u'] = -v/r
        jacobian['vdot', 'v'] = -u/r
        jacobian['vdot', 'm'] = -(thrust/m**2)*cos_alpha
        jacobian['vdot', 'alpha'] = -(thrust/m)*sin_alpha
        jacobian['vdot', 'thrust'] = cos_alpha/m
        
        jacobian['mdot', 'thrust'] = -1/w
        jacobian['mdot', 'w'] = thrust/w**2
        
        
#sstoODEvertical: ODEs for vertical rising/descent phase (zero tangential velocity and 90 deg thrust direction)
        
@declare_time(units='s')

@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')

@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')

class sstoODEvertical(ExplicitComponent):
    
    def initialize(self):
        
        """
        initialize an instance specifying the followings:
            num_nodes:  number of discrete points in which the constraint has to be evaluated
            mu:         lunar gravitational parameter (m^3/s^2)
            F:          thrust (N)
            w:          exaust velocity (m/s)
        """
        
        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', types=float)
        self.options.declare('F', types=float)
        self.options.declare('w', types=float)
        
    def setup(self):
        
        """
        define the equation inputs, outputs and jacobian components
        """
        
        nn = self.options['num_nodes']
        F = self.options['F']
        w = self.options['w']

        #inputs (ODE right-hand side)
        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('m', val=np.zeros(nn), desc='mass', units='kg')
        
        self.add_input('thrust', val=F*np.ones(nn), desc='thrust', units='N')
        self.add_input('w', val=w*np.ones(nn), desc='exaust velocity', units='m/s')

        #outputs (ODE left-hand side)
        self.add_output('rdot', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_output('udot', val=np.zeros(nn), desc='radial acceleration', units='m/s**2')
        self.add_output('mdot', val=np.zeros(nn), desc='mass rate', units='kg/s')

        #partial derivatives of the ODE outputs respect to the ODE inputs
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='rdot', wrt='u', rows=ar, cols=ar, val=1.0)
        
        self.declare_partials(of='udot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='thrust', rows=ar, cols=ar)
                
        self.declare_partials(of='mdot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='mdot', wrt='w', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        """
        compute the equation outputs
        """
        
        mu = self.options['mu']
        
        r = inputs['r']
        u = inputs['u']
        m = inputs['m']
        thrust = inputs['thrust']
        w = inputs['w']

        outputs['rdot'] = u
        outputs['udot'] = -mu/r**2 + thrust/m
        outputs['mdot'] = -thrust/w
        
    def compute_partials(self, inputs, jacobian):
        
        """
        compute the jacobian components
        """
        
        mu = self.options['mu']
        
        r = inputs['r']
        m = inputs['m']
        thrust = inputs['thrust']
        w = inputs['w']

        jacobian['udot', 'r'] = 2*mu/r**3
        jacobian['udot', 'm'] = -thrust/m**2
        jacobian['udot', 'thrust'] = 1/m
        
        jacobian['mdot', 'thrust'] = -1/w
        jacobian['mdot', 'w'] = thrust/w**2
        
        
#sstoODEthrottle: ODEs with variable thrust (throttle k)

@declare_time(units='s')

@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('theta', rate_source='thetadot', units='rad')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')

@declare_parameter('alpha', targets=['alpha'], units='rad')
@declare_parameter('k', targets=['k'])
@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')

class sstoODEthrottle(ExplicitComponent):
    
    def initialize(self):
        
        """
        initialize an instance specifying the followings:
            num_nodes:  number of discrete points in which the constraint has to be evaluated
            mu:         lunar gravitational parameter (m^3/s^2)
            F:          thrust (N)
            w:          exaust velocity (m/s)
        """
        
        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', types=float)
        self.options.declare('F', types=float)
        self.options.declare('w', types=float)
        
    def setup(self):
        
        """
        define the equation inputs, outputs and jacobian components
        """
        
        nn = self.options['num_nodes']
        F = self.options['F']
        w = self.options['w']

        #inputs (ODE right-hand side)
        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('v', val=np.zeros(nn), desc='tangential velocity', units='m/s')
        self.add_input('m', val=np.zeros(nn), desc='mass', units='kg')
        
        self.add_input('alpha', val=np.zeros(nn), desc='thrust direction', units='rad')
        self.add_input('k', val=np.zeros(nn), desc='throttle')
        self.add_input('thrust', val=F*np.ones(nn), desc='thrust', units='N')
        self.add_input('w', val=w*np.ones(nn), desc='exaust velocity', units='m/s')

        #outputs (ODE left-hand side)
        self.add_output('rdot', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_output('thetadot', val=np.zeros(nn), desc='angular rate', units='rad/s')
        self.add_output('udot', val=np.zeros(nn), desc='radial acceleration', units='m/s**2')
        self.add_output('vdot', val=np.zeros(nn), desc='tangential acceleration', units='m/s**2')
        self.add_output('mdot', val=np.zeros(nn), desc='mass rate', units='kg/s')

        #partial derivatives of the ODE outputs respect to the ODE inputs
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='rdot', wrt='u', rows=ar, cols=ar, val=1)
        
        self.declare_partials(of='thetadot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='thetadot', wrt='v', rows=ar, cols=ar)
        
        self.declare_partials(of='udot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='k', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='thrust', rows=ar, cols=ar)
        
        self.declare_partials(of='vdot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='k', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='thrust', rows=ar, cols=ar)
        
        self.declare_partials(of='mdot', wrt='k', rows=ar, cols=ar)
        self.declare_partials(of='mdot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='mdot', wrt='w', rows=ar, cols=ar)
                
    def compute(self, inputs, outputs):
        
        """
        compute the equation outputs
        """
        
        mu = self.options['mu']
        
        r = inputs['r']
        u = inputs['u']
        v = inputs['v']
        m = inputs['m']
        alpha = inputs['alpha']
        k = inputs['k']
        thrust = inputs['thrust']
        w = inputs['w']
        
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        outputs['rdot'] = u
        outputs['thetadot'] = v/r
        outputs['udot'] = -mu/r**2 + v**2/r + (thrust*k/m)*sin_alpha
        outputs['vdot'] = -u*v/r + (thrust*k/m)*cos_alpha
        outputs['mdot'] = -thrust*k/w
        
    def compute_partials(self, inputs, jacobian):
        
        """
        compute the jacobian components
        """
        
        mu = self.options['mu']
        
        r = inputs['r']
        u = inputs['u']
        v = inputs['v']
        m = inputs['m']
        alpha = inputs['alpha']
        k = inputs['k']
        thrust = inputs['thrust']
        w = inputs['w']        
        
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        jacobian['thetadot', 'r'] = -v/r**2
        jacobian['thetadot', 'v'] = 1/r

        jacobian['udot', 'r'] = 2*mu/r**3 - (v/r)**2
        jacobian['udot', 'v'] = 2*v/r
        jacobian['udot', 'm'] = -(thrust*k/m**2)*sin_alpha
        jacobian['udot', 'alpha'] = (thrust*k/m)*cos_alpha
        jacobian['udot', 'k'] = (thrust/m)*sin_alpha
        jacobian['udot', 'thrust'] = (k/m)*sin_alpha

        jacobian['vdot', 'r'] = u*v/r**2
        jacobian['vdot', 'u'] = -v/r
        jacobian['vdot', 'v'] = -u/r
        jacobian['vdot', 'm'] = -(thrust*k/m**2)*cos_alpha
        jacobian['vdot', 'alpha'] = -(thrust*k/m)*sin_alpha
        jacobian['vdot', 'k'] = (thrust/m)*cos_alpha
        jacobian['vdot', 'thrust'] = (k/m)*cos_alpha
        
        jacobian['mdot', 'k'] = -thrust/w
        jacobian['mdot', 'thrust'] = -k/w
        jacobian['mdot', 'w'] = (thrust*k)/w**2